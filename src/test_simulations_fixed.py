#!/usr/bin/env python3
"""
test_simulations.py
===================

Validation/simulation suite for the radioprotection transport library.

The suite is deliberately built around the public classes already present in
the project:

    - IndoorsDiffusionAdvectionDecay
    - OutdoorsDiffusionAdvectionDecay
    - UniformField
    - ShearField
    - GustField
    - VortexField
    - HRTM

It creates:
    * 24 principal indoor/outdoor cases (8 scenarios x 3 radionuclides)
    * analytical radioactive-decay checks for Tc-99m, I-131 and F-18
    * optional HRTM/dosimetry checks
    * CSV and JSON summaries
    * optional plots

IMPORTANT:
The present DiffusionAdvectionDecay implementation continuously injects the
configured source at every time step. Therefore the decay-validation tests
explicitly set emission_rate=0 and initialise the concentration field
manually. This tests the decay term without changing the library itself.

The script is intended to be run from the project environment, e.g.:

    python test_simulations.py

or:

    python test_simulations.py --output-dir validation_results

For a faster smoke test:

    python test_simulations.py --smoke

For HRTM as well:

    python test_simulations.py --with-hrtm

For plots:

    python test_simulations.py --plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

# Project imports. These are intentionally kept aligned with the package
# exports shown in radioprotection/__init__.py.
from radioprotection import (
    IndoorsDiffusionAdvectionDecay,
    OutdoorsDiffusionAdvectionDecay,
    UniformField,
    ShearField,
    GustField,
    VortexField,
    HRTM,
)

from radioprotection.utils import lambda_for_species


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RADIONUCLIDES = {
    "Tc-99m": {
        "half_life_s": 6.02 * 3600.0,
        "label": "Technetium-99m",
    },
    "I-131": {
        "half_life_s": 8.04 * 24.0 * 3600.0,
        "label": "Iodine-131",
    },
    "F-18": {
        "half_life_s": 109.77 * 60.0,
        "label": "Fluorine-18",
    },
}


@dataclass
class SuiteConfig:
    output_dir: str = "validation_results"
    plots: bool = False
    with_hrtm: bool = False
    smoke: bool = False
    verbose: bool = True


@dataclass
class Result:
    test_id: str
    scenario: str
    environment: str
    species: str
    status: str
    runtime_s: float
    n_steps: int = 0
    final_time_s: float = 0.0
    max_initial: float = float("nan")
    max_final: float = float("nan")
    total_initial: float = float("nan")
    total_final: float = float("nan")
    centre_initial_x: float = float("nan")
    centre_final_x: float = float("nan")
    centre_initial_y: float = float("nan")
    centre_final_y: float = float("nan")
    centre_initial_z: float = float("nan")
    centre_final_z: float = float("nan")
    metric_name: str = ""
    metric_value: float = float("nan")
    metric_limit: float = float("nan")
    notes: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

def volume_element(d: tuple[float, float, float, float]) -> float:
    return d[0] * d[1] * d[2]


def total_activity(field: np.ndarray, d: tuple[float, float, float, float]) -> float:
    """
    Integrate C over volume.

    Note:
    The library labels concentration as Bq/m^3. The integral is therefore
    expressed in Bq.
    """
    return float(np.sum(field) * volume_element(d))


def concentration_centroid(
    field: np.ndarray,
    d: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """
    Concentration-weighted centroid in physical coordinates.
    """
    total = np.sum(field)
    if total <= 0:
        return float("nan"), float("nan"), float("nan")

    ix, iy, iz = np.indices(field.shape)

    weights = field
    x = float(np.sum(ix * d[0] * weights) / total)
    y = float(np.sum(iy * d[1] * weights) / total)
    z = float(np.sum(iz * d[2] * weights) / total)

    return x, y, z


def field_at_time(fields: Dict[float, np.ndarray], time_s: float) -> np.ndarray:
    """
    Retrieve the nearest saved field.

    The library saves only every N iterations, so exact target times are not
    necessarily present.
    """
    if not fields:
        raise RuntimeError("No saved concentration fields were produced.")

    key = min(fields.keys(), key=lambda t: abs(t - time_s))
    return fields[key]


def safe_relative_error(numerical: float, analytical: float) -> float:
    scale = max(abs(analytical), 1e-30)
    return abs(numerical - analytical) / scale


def print_result(result: Result) -> None:
    if result.status == "PASS":
        marker = "PASS"
    elif result.status == "FAIL":
        marker = "FAIL"
    else:
        marker = "SKIP"

    print(
        f"[{marker:4s}] {result.test_id:24s} "
        f"{result.species:7s} "
        f"{result.metric_name:28s} "
        f"{result.metric_value:.4e}"
    )


# ---------------------------------------------------------------------------
# Configuration used by the simulations
# ---------------------------------------------------------------------------

def simulation_parameters(smoke: bool = False) -> dict[str, Any]:
    """
    Keep the production validation small enough to run comfortably while
    retaining physical dimensions and the stability margin.

    The library itself checks diffusion/CFL stability in indoor simulations
    and the wind-field classes check CFL for outdoor simulations.
    """
    if smoke:
        return {
            "grid_shape": (12, 12, 8),
            "d": (1.0, 1.0, 1.0, 1.0),
            "D": (1e-3, 1e-3, 1e-3),
            "total_time": 120.0,
            "save_every": 20,
            "emission_rate": 100.0,
        }

    return {
        "grid_shape": (20, 20, 12),
        "d": (1.0, 1.0, 1.0, 1.0),
        "D": (1e-3, 1e-3, 1e-3),
        "total_time": 600.0,
        "save_every": 30,
        "emission_rate": 100.0,
    }


def central_source(grid_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        grid_shape[0] // 2,
        grid_shape[1] // 2,
        grid_shape[2] // 2,
    )


# ---------------------------------------------------------------------------
# Analytical decay validation
# ---------------------------------------------------------------------------

def run_decay_validation(config: SuiteConfig) -> list[Result]:
    """Validate radioactive decay using the public indoor class only."""
    results: list[Result] = []
    dt = 60.0

    for species, info in RADIONUCLIDES.items():
        start = time.perf_counter()
        try:
            grid_shape = (6, 6, 6)
            d = (1.0, 1.0, 1.0, dt)
            total_time = info["half_life_s"]

            # The base DiffusionAdvectionDecay class is intentionally not
            # imported. The validation uses the public indoor implementation.
            # A tiny positive D avoids the D=0 denominator in the indoor wall
            # boundary calculation while making diffusion negligible.
            sim = IndoorsDiffusionAdvectionDecay(
                grid_shape=grid_shape,
                d=d,
                total_time=total_time,
                diffusion_coefficient=(1e-12, 1e-12, 1e-12),
                species_name=species,
                source_positions=[],
                emission_rate=0.0,
                wall_deposition=0.0,
                inlet_regions=[],
                outlet_regions=[],
                inlet_wind_velocity=0.0,
                outlet_wind_velocity=0.0,
                inlet_concentration=0.0,
            )

            c0 = 1.0
            sim._concentration[:] = c0
            n_steps = int(total_time / dt)
            fields = sim.run(save_every_X_iteration=max(1, n_steps))
            final = field_at_time(fields, total_time)
            numerical = float(np.mean(final))
            analytical = c0 * math.exp(-lambda_for_species(species) * total_time)
            error = safe_relative_error(numerical, analytical)
            relative_to_half = abs(numerical - 0.5) / 0.5
            status = "PASS" if relative_to_half <= 0.01 else "FAIL"

            results.append(Result(
                test_id="DECAY_01",
                scenario="Pure radioactive decay",
                environment="indoor",
                species=species,
                status=status,
                runtime_s=time.perf_counter() - start,
                n_steps=n_steps,
                final_time_s=max(fields.keys()),
                max_initial=c0,
                max_final=float(np.max(final)),
                total_initial=total_activity(np.full(grid_shape, c0), d),
                total_final=total_activity(final, d),
                metric_name="relative error at half-life",
                metric_value=error,
                metric_limit=0.01,
                notes=(f"Analytical={analytical:.8e}; "
                       f"numerical={numerical:.8e}; "
                       f"|C/C0-0.5|/0.5={relative_to_half:.3e}"),
            ))
        except Exception as exc:
            results.append(Result(
                test_id="DECAY_01",
                scenario="Pure radioactive decay",
                environment="indoor",
                species=species,
                status="FAIL",
                runtime_s=time.perf_counter() - start,
                metric_name="execution",
                metric_value=float("nan"),
                metric_limit=0.0,
                error=f"{type(exc).__name__}: {exc}",
            ))
    return results


# ---------------------------------------------------------------------------
# Generic simulation execution
# ---------------------------------------------------------------------------

def execute_simulation(
    test_id: str,
    scenario: str,
    environment: str,
    species: str,
    sim_factory: Callable[[str], Any],
    params: dict[str, Any],
    metric: Callable[[Any, Dict[float, np.ndarray]], tuple[str, float, float, str]],
) -> Result:
    start = time.perf_counter()

    try:
        sim = sim_factory(species)
        fields = sim.run(save_every_X_iteration=params["save_every"])

        initial = field_at_time(fields, 0.0)
        final_time = max(fields.keys())
        final = fields[final_time]

        d = params["d"]

        xi, yi, zi = concentration_centroid(initial, d)
        xf, yf, zf = concentration_centroid(final, d)

        metric_name, metric_value, metric_limit, notes = metric(
            sim, fields
        )

        passed = bool(np.isfinite(metric_value) and metric_value <= metric_limit)

        result = Result(
            test_id=test_id,
            scenario=scenario,
            environment=environment,
            species=species,
            status="PASS" if passed else "FAIL",
            runtime_s=time.perf_counter() - start,
            n_steps=int(params["total_time"] / params["d"][3]),
            final_time_s=final_time,
            max_initial=float(np.max(initial)),
            max_final=float(np.max(final)),
            total_initial=total_activity(initial, d),
            total_final=total_activity(final, d),
            centre_initial_x=xi,
            centre_final_x=xf,
            centre_initial_y=yi,
            centre_final_y=yf,
            centre_initial_z=zi,
            centre_final_z=zf,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_limit=metric_limit,
            notes=notes,
        )
        return result

    except Exception as exc:
        return Result(
            test_id=test_id,
            scenario=scenario,
            environment=environment,
            species=species,
            status="FAIL",
            runtime_s=time.perf_counter() - start,
            metric_name="execution",
            metric_value=float("nan"),
            metric_limit=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Metrics for principal cases
# ---------------------------------------------------------------------------

def metric_finite_nonnegative(
    _sim: Any,
    fields: Dict[float, np.ndarray],
) -> tuple[str, float, float, str]:
    all_values = np.concatenate([f.ravel() for f in fields.values()])

    if not np.all(np.isfinite(all_values)):
        return "finite/non-negative field", float("inf"), 0.0, "NaN/Inf detected"

    minimum = float(np.min(all_values))

    if minimum < -1e-12:
        return "finite/non-negative field", abs(minimum), 1e-12, "Negative concentration"

    return "finite/non-negative field", abs(minimum), 1e-12, "OK"


def metric_centroid_shift_x(
    sim: Any,
    fields: Dict[float, np.ndarray],
) -> tuple[str, float, float, str]:
    d = sim._d
    initial = field_at_time(fields, 0.0)
    final = field_at_time(fields, max(fields.keys()))

    xi, _, _ = concentration_centroid(initial, d)
    xf, _, _ = concentration_centroid(final, d)

    return (
        "|centroid_x - initial_x|",
        abs(xf - xi),
        1e9,
        f"initial={xi:.4f} m; final={xf:.4f} m",
    )


def metric_mass_not_nan(
    sim: Any,
    fields: Dict[float, np.ndarray],
) -> tuple[str, float, float, str]:
    d = sim._d
    values = np.array([total_activity(f, d) for f in fields.values()])

    if not np.all(np.isfinite(values)):
        return "finite integrated activity", float("inf"), 0.0, "NaN/Inf detected"

    return (
        "finite integrated activity",
        0.0,
        0.0,
        f"range={np.min(values):.4e}..{np.max(values):.4e} Bq",
    )


def metric_outdoor_advection(
    sim: Any,
    fields: Dict[float, np.ndarray],
    expected_velocity: float,
) -> tuple[str, float, float, str]:
    d = sim._d
    initial = field_at_time(fields, 0.0)
    final_time = max(fields.keys())
    final = fields[final_time]

    xi, _, _ = concentration_centroid(initial, d)
    xf, _, _ = concentration_centroid(final, d)

    expected = xi + expected_velocity * final_time

    # Because the finite grid has boundaries, use the expected displacement
    # only when the plume remains inside the computational domain.
    numerical_shift = xf - xi
    expected_shift = expected - xi

    error = abs(numerical_shift - expected_shift)

    return (
        "absolute centroid advection error [m]",
        error,
        2.5,
        f"numerical shift={numerical_shift:.4f} m; "
        f"expected shift={expected_shift:.4f} m",
    )


def metric_indoor_ventilation(
    sim: Any,
    fields: Dict[float, np.ndarray],
) -> tuple[str, float, float, str]:
    d = sim._d

    initial = field_at_time(fields, 0.0)
    final = field_at_time(fields, max(fields.keys()))

    a0 = total_activity(initial, d)
    af = total_activity(final, d)

    # The metric is a qualitative numerical sanity condition:
    # ventilation/deposition/decay should not create activity.
    ratio = af / max(a0 + 1.0, 1.0)

    # Since the source is continuous in the current library, this is only a
    # sanity bound, not a physical conservation assertion.
    return (
        "final/initial integrated-activity sanity ratio",
        ratio,
        1e9,
        f"initial={a0:.4e} Bq; final={af:.4e} Bq",
    )


# ---------------------------------------------------------------------------
# Main 24-case suite
# ---------------------------------------------------------------------------

def build_cases(params: dict[str, Any]) -> list[tuple[str, str, str, Callable, Callable]]:
    N = params["grid_shape"]
    d = params["d"]
    D = params["D"]
    source = central_source(N)
    Q = params["emission_rate"]

    def indoor_closed(species: str):
        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q,
            wall_deposition=0.0,
        )

    def indoor_controlled_vent(species: str):
        inlet = [{
            "wall": "xmin",
            "y": (N[1] // 3, 2 * N[1] // 3),
            "z": (N[2] // 3, 2 * N[2] // 3),
        }]
        outlet = [{
            "wall": "xmax",
            "y": (N[1] // 3, 2 * N[1] // 3),
            "z": (N[2] // 3, 2 * N[2] // 3),
        }]

        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q,
            wall_deposition=1e-5,
            inlet_regions=inlet,
            outlet_regions=outlet,
            inlet_wind_velocity=0.05,
            outlet_wind_velocity=0.05,
            inlet_concentration=0.0,
        )

    def indoor_uncontrolled_vent(species: str):
        # Multiple openings intentionally stress the boundary-mask handling.
        inlet = [
            {
                "wall": "xmin",
                "y": (2, N[1] // 2),
                "z": (2, N[2] - 2),
            },
            {
                "wall": "ymin",
                "x": (N[0] // 2, N[0] - 2),
                "z": (2, N[2] - 2),
            },
        ]
        outlet = [
            {
                "wall": "xmax",
                "y": (N[1] // 2, N[1] - 2),
                "z": (2, N[2] - 2),
            },
            {
                "wall": "ymax",
                "x": (2, N[0] // 2),
                "z": (2, N[2] - 2),
            },
        ]

        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q * 2.0,
            wall_deposition=2e-5,
            inlet_regions=inlet,
            outlet_regions=outlet,
            inlet_wind_velocity=0.05,
            outlet_wind_velocity=0.05,
            inlet_concentration=0.0,
        )

    def indoor_high_release(species: str):
        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q * 5.0,
            wall_deposition=2e-5,
        )

    def outdoor_uniform(species: str):
        wind = UniformField(
            grid_shape=N,
            initial_velocity=(0.05, 0.0, 0.0),
        )
        return OutdoorsDiffusionAdvectionDecay(
            wind_model=wind,
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q,
        )

    def outdoor_shear(species: str):
        wind = ShearField(
            grid_shape=N,
            Uref=0.05,
            zref=max(N[2], 2),
            alpha=0.20,
        )
        return OutdoorsDiffusionAdvectionDecay(
            wind_model=wind,
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q,
        )

    def outdoor_gust(species: str):
        wind = GustField(
            grid_shape=N,
            Umean=0.04,
            amplitude=0.01,
            period=120.0,
        )
        return OutdoorsDiffusionAdvectionDecay(
            wind_model=wind,
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q,
        )

    def outdoor_vortex(species: str):
        wind = VortexField(
            grid_shape=N,
            omega=0.001,
        )
        return OutdoorsDiffusionAdvectionDecay(
            wind_model=wind,
            grid_shape=N,
            d=d,
            total_time=params["total_time"],
            diffusion_coefficient=D,
            species_name=species,
            source_positions=[source],
            emission_rate=Q,
        )

    return [
        (
            "INDOOR_01",
            "Indoor controlled: closed room",
            "indoor",
            indoor_closed,
            metric_finite_nonnegative,
        ),
        (
            "INDOOR_02",
            "Indoor controlled: known ventilation",
            "indoor",
            indoor_controlled_vent,
            metric_finite_nonnegative,
        ),
        (
            "INDOOR_03",
            "Indoor uncontrolled: multiple openings",
            "indoor",
            indoor_uncontrolled_vent,
            metric_finite_nonnegative,
        ),
        (
            "INDOOR_04",
            "Indoor uncontrolled: high release",
            "indoor",
            indoor_high_release,
            metric_finite_nonnegative,
        ),
        (
            "OUTDOOR_01",
            "Outdoor controlled: uniform wind",
            "outdoor",
            outdoor_uniform,
            lambda sim, fields: metric_outdoor_advection(
                sim, fields, expected_velocity=0.05
            ),
        ),
        (
            "OUTDOOR_02",
            "Outdoor controlled: shear wind",
            "outdoor",
            outdoor_shear,
            metric_finite_nonnegative,
        ),
        (
            "OUTDOOR_03",
            "Outdoor uncontrolled: gusts",
            "outdoor",
            outdoor_gust,
            metric_finite_nonnegative,
        ),
        (
            "OUTDOOR_04",
            "Outdoor uncontrolled: vortex",
            "outdoor",
            outdoor_vortex,
            metric_finite_nonnegative,
        ),
    ]


def run_principal_suite(config: SuiteConfig) -> list[Result]:
    params = simulation_parameters(config.smoke)
    cases = build_cases(params)
    results: list[Result] = []

    for test_id, scenario, environment, factory, metric in cases:
        for species in RADIONUCLIDES:
            result = execute_simulation(
                test_id=test_id,
                scenario=scenario,
                environment=environment,
                species=species,
                sim_factory=factory,
                params=params,
                metric=metric,
            )
            results.append(result)
            print_result(result)

    return results


# ---------------------------------------------------------------------------
# HRTM validation
# ---------------------------------------------------------------------------

def run_hrtm_validation(
    config: SuiteConfig,
    simulation_results: list[Result],
) -> list[Result]:
    """
    Run one representative indoor simulation per radionuclide and pass its
    saved concentration fields to HRTM.

    HRTM is kept optional because it depends on the project's external
    dosimetric data files being available in the configured package/data path.
    """
    results: list[Result] = []

    params = simulation_parameters(config.smoke)
    N = params["grid_shape"]
    source = central_source(N)

    for species in RADIONUCLIDES:
        start = time.perf_counter()

        try:
            sim = IndoorsDiffusionAdvectionDecay(
                grid_shape=N,
                d=params["d"],
                total_time=params["total_time"],
                diffusion_coefficient=params["D"],
                species_name=species,
                source_positions=[source],
                emission_rate=params["emission_rate"],
                wall_deposition=1e-5,
            )
            fields = sim.run(save_every_X_iteration=params["save_every"])

            hrtm = HRTM(
                sim,
                population_type="public",
                age_group="adult",
                gender="male",
                physical_activity="sitting",
                absorption="F",
            )

            dose_fields = hrtm.effective_dose_commitment()

            arrays = [np.asarray(f) for f in dose_fields.values()]
            finite = all(np.all(np.isfinite(a)) for a in arrays)
            nonnegative = all(np.all(a >= 0) for a in arrays)

            if not arrays:
                status = "FAIL"
                metric_value = float("inf")
                notes = "HRTM returned no saved dose fields."
            elif not finite or not nonnegative:
                status = "FAIL"
                metric_value = float("inf")
                notes = "Dose field contains NaN/Inf or negative values."
            else:
                status = "PASS"
                metric_value = 0.0
                notes = (
                    f"{len(arrays)} dose fields generated; "
                    f"max dose={max(float(np.max(a)) for a in arrays):.4e}"
                )

            results.append(
                Result(
                    test_id="HRTM_01",
                    scenario="Indoor concentration -> HRTM effective dose",
                    environment="indoor",
                    species=species,
                    status=status,
                    runtime_s=time.perf_counter() - start,
                    n_steps=int(params["total_time"] / params["d"][3]),
                    final_time_s=max(fields.keys()),
                    metric_name="finite/non-negative dose field",
                    metric_value=metric_value,
                    metric_limit=0.0,
                    notes=notes,
                )
            )

        except Exception as exc:
            results.append(
                Result(
                    test_id="HRTM_01",
                    scenario="Indoor concentration -> HRTM effective dose",
                    environment="indoor",
                    species=species,
                    status="FAIL",
                    runtime_s=time.perf_counter() - start,
                    metric_name="execution",
                    metric_value=float("nan"),
                    metric_limit=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return results


# ---------------------------------------------------------------------------
# CSV / JSON reporting
# ---------------------------------------------------------------------------

def save_results(results: Iterable[Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in results]

    json_path = output_dir / "validation_results.json"
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "validation_results.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    passed = sum(r.status == "PASS" for r in results)
    failed = sum(r.status == "FAIL" for r in results)
    skipped = sum(r.status == "SKIP" for r in results)

    summary = {
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": passed / len(rows) if rows else 0.0,
    }

    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nValidation summary")
    print("------------------")
    print(f"Total : {summary['total']}")
    print(f"PASS  : {summary['passed']}")
    print(f"FAIL  : {summary['failed']}")
    print(f"SKIP  : {summary['skipped']}")
    print(f"Rate  : {100.0 * summary['pass_rate']:.1f}%")
    print(f"\nResults written to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------

def make_plots(results: list[Result], output_dir: Path) -> None:
    """
    Generate a compact diagnostic plot from the result table.

    No simulation field is re-run here. This keeps reporting separate from
    numerical execution.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; plots skipped.")
        return

    # Runtime by test/species.
    labels = [f"{r.test_id}\n{r.species}" for r in results]
    runtime = [r.runtime_s for r in results]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(results)), runtime)
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Validation case")
    ax.set_title("Validation-suite runtime")
    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_runtime.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> SuiteConfig:
    parser = argparse.ArgumentParser(
        description="Run the radioprotection diffusion/advection/decay validation suite."
    )

    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Directory for CSV/JSON reports and optional plots.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate diagnostic plots.",
    )
    parser.add_argument(
        "--with-hrtm",
        action="store_true",
        help="Also run the HRTM/dose validation.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a smaller/faster grid and shorter simulations.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    args = parser.parse_args()

    return SuiteConfig(
        output_dir=args.output_dir,
        plots=args.plots,
        with_hrtm=args.with_hrtm,
        smoke=args.smoke,
        verbose=not args.quiet,
    )


def main() -> int:
    config = parse_args()
    output_dir = Path(config.output_dir)

    print("=" * 78)
    print("Radioprotection transport validation suite")
    print("=" * 78)
    print("Radionuclides:", ", ".join(RADIONUCLIDES))
    print("Principal cases: 8 scenarios x 3 radionuclides = 24")
    print("HRTM:", "enabled" if config.with_hrtm else "disabled")
    print("Mode:", "smoke" if config.smoke else "validation")
    print()

    all_results: list[Result] = []

    print("1/3 - Analytical radioactive-decay tests")
    decay_results = run_decay_validation(config)
    all_results.extend(decay_results)
    for result in decay_results:
        print_result(result)

    print("\n2/3 - Indoor/outdoor transport suite")
    principal_results = run_principal_suite(config)
    all_results.extend(principal_results)

    if config.with_hrtm:
        print("\n3/3 - HRTM/dosimetry tests")
        hrtm_results = run_hrtm_validation(config, principal_results)
        all_results.extend(hrtm_results)
        for result in hrtm_results:
            print_result(result)
    else:
        print("\n3/3 - HRTM/dosimetry tests skipped")
        print("Use --with-hrtm when the HRTM data files are available.")

    save_results(all_results, output_dir)

    if config.plots:
        make_plots(all_results, output_dir)
        print(f"Plot written to: {(output_dir / 'validation_runtime.png').resolve()}")

    failed = [r for r in all_results if r.status == "FAIL"]

    print("\nFailures")
    print("--------")
    if not failed:
        print("No failures reported.")
    else:
        for r in failed:
            print(
                f"{r.test_id} / {r.species}: "
                f"{r.error or r.notes}"
            )

    # Return a non-zero exit code if a validation case failed.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
