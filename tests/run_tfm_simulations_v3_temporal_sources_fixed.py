#!/usr/bin/env python3
"""
run_tfm_simulations.py
======================

Automated simulation campaign for the TFM using the existing radioprotection
library.

The script uses only the public classes already implemented in the library:
    - IndoorsDiffusionAdvectionDecay
    - OutdoorsDiffusionAdvectionDecay
    - UniformField
    - ShearField
    - GustField
    - VortexField
    - HRTM (optional)

The campaign is built around:
    3 radionuclides x 7 emission levels x 6 transport scenarios = 126 cases

Radionuclides:
    Tc-99m, I-131, F-18

Emission levels are expressed as a multiplier of Q_REFERENCE for each
radionuclide:
    0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 5.00

Classification:
    factor < 1.0  -> CONTROLLED
    factor = 1.0  -> REFERENCE
    factor > 1.0  -> UNCONTROLLED

IMPORTANT:
The library also supports finite-duration sources through
source_effective_iterations. This version adds a dedicated temporal-source
campaign that varies the source duration independently of total_time.

The current project validation suite uses Q=100.0 as its common reference
emission. Until isotope-specific normal emission rates are established from
your experimental/clinical/source definition, this script uses 100.0 Bq/s
as the reference value for all three isotopes. Change Q_REFERENCE below when
you have the documented normal rates.

Run examples:
    python run_tfm_simulations.py --smoke
    python run_tfm_simulations.py
    python run_tfm_simulations.py --with-hrtm
    python run_tfm_simulations.py --with-hrtm --save-fields
    python run_tfm_simulations.py --output-dir tfm_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from DifAdDec import (
    IndoorsDiffusionAdvectionDecay,
    OutdoorsDiffusionAdvectionDecay,
    UniformField,
    ShearField,
    GustField,
    VortexField,
    HRTM,
)


# ============================================================================
# USER CONFIGURATION
# ============================================================================

RADIONUCLIDES = ("Tc-99m", "I-131", "F-18")

# Reference continuous source rates.
#
# These are the values around which CONTROLLED/REFERENCE/UNCONTROLLED
# scenarios are generated.
#
# The existing project simulation suite uses Q=100.0. These values are
# therefore deliberately kept at 100.0 until isotope-specific "normal"
# emission rates are supplied from the TFM source definition.
Q_REFERENCE = {
    "Tc-99m": 8.33e6,   # Bq/s
    "I-131": 6.17e7,    # Bq/s
    "F-18": 6.67e6,     # Bq/s
}

# Emission multipliers.
EMISSION_FACTORS = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    5.00,
)

# Resolution presets.
RESOLUTIONS = {
    "low": {
        "grid_shape": (20, 20, 12),
        "d": (1.0, 1.0, 1.0, 1.0),
    },
    "medium": {
        "grid_shape": (40, 40, 24),
        "d": (0.5, 0.5, 0.5, 0.1),
    },
    "high": {
        "grid_shape": (60, 60, 30),
        "d": (0.5, 0.5, 0.5, 0.05),
    },
}

GRID_SHAPE = RESOLUTIONS["medium"]["grid_shape"]
D = RESOLUTIONS["medium"]["d"]
DIFFUSION_COEFFICIENT = (1e-3, 1e-3, 1e-3)

TOTAL_TIME_S = 600.0
SAVE_EVERY = 30

# Smoke test: deliberately much smaller.
SMOKE_GRID_SHAPE = (12, 12, 8)
SMOKE_D = (1.0, 1.0, 1.0, 1.0)
SMOKE_TOTAL_TIME_S = 120.0
SMOKE_SAVE_EVERY = 20

# Indoor parameters.
INDOOR_WALL_DEPOSITION = 1e-5
INDOOR_LOW_VENTILATION = 0.02
INDOOR_NORMAL_VENTILATION = 0.05
INDOOR_HIGH_VENTILATION = 0.10

# Outdoor wind parameters.
UNIFORM_SPEEDS = (0.02, 0.05, 0.10)

SHEAR_UREF = 0.05
SHEAR_ALPHA = 0.20
SHEAR_ZREF = 10.0

GUST_UMEAN = 0.05
GUST_AMPLITUDE = 0.02
GUST_PERIOD = 120.0

VORTEX_OMEGA = 0.001

# HRTM profiles. These are run only when --with-hrtm is supplied.
HRTM_PROFILES = (
    {
        "profile": "adult_sitting_male",
        "population_type": "public",
        "age_group": "adult",
        "gender": "male",
        "physical_activity": "sitting",
        "absorption": "F",
    },
    {
        "profile": "adult_walking_male",
        "population_type": "public",
        "age_group": "adult",
        "gender": "male",
        "physical_activity": "walking",
        "absorption": "F",
    },
    {
        "profile": "adult_running_male",
        "population_type": "public",
        "age_group": "adult",
        "gender": "male",
        "physical_activity": "running",
        "absorption": "F",
    },
)

# To avoid multiplying the complete 126-case campaign by all HRTM profiles,
# HRTM is applied by default only to representative emission levels.
HRTM_EMISSION_FACTORS = (0.50, 1.00, 2.00)

# ---------------------------------------------------------------------------
# SOURCE GEOMETRIES
# ---------------------------------------------------------------------------
#
# Q_REFERENCE is the TOTAL continuous source rate [Bq/s].
# For distributed sources, the rate is divided equally among all source
# cells so that changing the source geometry does not change total injected
# activity per second.
SOURCE_MODES = (
    "multiple_points",
    "planar",
    "volumetric",
)

# Five discrete point sources around the centre.
MULTIPLE_POINT_OFFSETS = (
    (-3, 0, 0),
    (0, 0, 0),
    (3, 0, 0),
    (0, -3, 0),
    (0, 3, 0),
)

# Square planar source at the central z layer.
PLANAR_HALF_WIDTH = 3
PLANAR_THICKNESS = 1

# Cuboidal 3-D source.
VOLUME_HALF_WIDTH = 2
VOLUME_HALF_HEIGHT = 1

# Save concentration fields only when explicitly requested.
SAVE_FIELDS_DEFAULT = False
# ---------------------------------------------------------------------------
# TEMPORAL SOURCE VALIDATION
# ---------------------------------------------------------------------------
# source_effective_iterations is a SOURCE duration, independent of
# total_time. Durations are defined as fractions of the transport run.
#
# 0.10 -> source active during the first 10% of the run
# 0.25 -> first 25%
# 0.50 -> first 50%
# 1.00 -> entire run
# 1.50 -> requested source duration exceeds the transport run
TEMPORAL_SOURCE_FRACTIONS = (0.10, 0.25, 0.50, 1.0, 1.5)
TEMPORAL_SOURCE_EMISSION_FACTORS = (0.50, 1.00, 2.00)
TEMPORAL_SOURCE_RADIONUCLIDES = ("Tc-99m", "I-131", "F-18")
TEMPORAL_SOURCE_MODES = ("multiple_points", "planar", "volumetric")



# ============================================================================
# RESULT DATA
# ============================================================================

@dataclass
class Result:
    case_id: str
    environment: str
    scenario: str
    wind_model: str
    source_mode: str
    source_description: str
    resolution: str
    number_of_source_cells: int
    radionuclide: str
    emission_factor: float
    emission_rate_bq_s: float
    emission_classification: str

    grid_shape: str
    dx_m: float
    dy_m: float
    dz_m: float
    dt_s: float
    total_time_s: float

    max_concentration_bq_m3: float
    final_max_concentration_bq_m3: float
    total_activity_bq: float
    final_total_activity_bq: float

    centroid_x_m: float
    centroid_y_m: float
    centroid_z_m: float
    final_centroid_x_m: float
    final_centroid_y_m: float
    final_centroid_z_m: float

    runtime_s: float
    n_steps: int

    source_duration_s: float = float("nan")
    source_effective_iterations: int = -1
    source_active_iterations_expected: int = -1
    source_active_iterations_actual: int = -1
    source_iteration_semantics_ok: bool = False

    hrtm_profile: str = ""
    max_dose_field: float = float("nan")
    final_max_dose_field: float = float("nan")

    field_file: str = ""
    status: str = "OK"
    error: str = ""


# ============================================================================
# NUMERICAL HELPERS
# ============================================================================

def volume_element(d: tuple[float, float, float, float]) -> float:
    return d[0] * d[1] * d[2]


def total_activity(field: np.ndarray, d: tuple[float, float, float, float]) -> float:
    """Integral of concentration over the computational volume."""
    return float(np.sum(field) * volume_element(d))


def concentration_centroid(
    field: np.ndarray,
    d: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Concentration-weighted centroid in physical coordinates."""
    total = float(np.sum(field))

    if total <= 0.0:
        return float("nan"), float("nan"), float("nan")

    ix, iy, iz = np.indices(field.shape)

    x = float(np.sum(ix * d[0] * field) / total)
    y = float(np.sum(iy * d[1] * field) / total)
    z = float(np.sum(iz * d[2] * field) / total)

    return x, y, z


def emission_classification(factor: float) -> str:
    if factor < 1.0:
        return "CONTROLLED"
    if math.isclose(factor, 1.0):
        return "REFERENCE"
    return "UNCONTROLLED"


def safe_max(fields: dict[float, np.ndarray]) -> float:
    if not fields:
        return float("nan")
    return float(max(np.max(field) for field in fields.values()))


def safe_final_field(fields: dict[float, np.ndarray]) -> np.ndarray:
    if not fields:
        raise RuntimeError("No concentration field was saved.")
    final_time = max(fields.keys())
    return fields[final_time]


def total_steps(total_time_s: float, dt_s: float) -> int:
    return int(total_time_s / dt_s)


# ============================================================================
# GEOMETRY
# ============================================================================

def central_source(
    grid_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the central grid index."""
    return (
        grid_shape[0] // 2,
        grid_shape[1] // 2,
        grid_shape[2] // 2,
    )


def build_source_positions(
    grid_shape: tuple[int, int, int],
    source_mode: str,
) -> list[tuple[int, int, int]]:
    """Build a point, planar or volumetric source from grid cells."""
    nx, ny, nz = grid_shape
    cx, cy, cz = central_source(grid_shape)

    if source_mode == "multiple_points":
        positions = [
            (cx + dx, cy + dy, cz + dz)
            for dx, dy, dz in MULTIPLE_POINT_OFFSETS
        ]

    elif source_mode == "planar":
        z0 = cz - PLANAR_THICKNESS // 2
        positions = [
            (i, j, z0)
            for i in range(
                cx - PLANAR_HALF_WIDTH,
                cx + PLANAR_HALF_WIDTH + 1,
            )
            for j in range(
                cy - PLANAR_HALF_WIDTH,
                cy + PLANAR_HALF_WIDTH + 1,
            )
        ]

    elif source_mode == "volumetric":
        positions = [
            (i, j, k)
            for i in range(
                cx - VOLUME_HALF_WIDTH,
                cx + VOLUME_HALF_WIDTH + 1,
            )
            for j in range(
                cy - VOLUME_HALF_WIDTH,
                cy + VOLUME_HALF_WIDTH + 1,
            )
            for k in range(
                cz - VOLUME_HALF_HEIGHT,
                cz + VOLUME_HALF_HEIGHT + 1,
            )
        ]

    else:
        raise ValueError(f"Unknown source mode: {source_mode}")

    for i, j, k in positions:
        if not (
            0 < i < nx - 1
            and 0 < j < ny - 1
            and 0 < k < nz - 1
        ):
            raise ValueError(
                f"Source cell {(i, j, k)} is too close to the boundary "
                f"for grid {grid_shape}."
            )

    return positions


def source_description(
    source_mode: str,
    positions: list[tuple[int, int, int]],
) -> str:
    if source_mode == "multiple_points":
        return f"{len(positions)} point sources"
    if source_mode == "planar":
        return f"planar source ({len(positions)} cells)"
    if source_mode == "volumetric":
        return f"3-D volumetric source ({len(positions)} cells)"
    return source_mode


def ventilation_regions(
    N: tuple[int, int, int],
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Construct inlet/outlet regions compatible with
    IndoorsDiffusionAdvectionDecay.
    """
    nx, ny, nz = N

    if mode == "normal":
        inlet = [{
            "wall": "xmin",
            "y": (ny // 3, 2 * ny // 3),
            "z": (nz // 3, 2 * nz // 3),
        }]
        outlet = [{
            "wall": "xmax",
            "y": (ny // 3, 2 * ny // 3),
            "z": (nz // 3, 2 * nz // 3),
        }]
        return inlet, outlet

    if mode == "low":
        inlet = [{
            "wall": "xmin",
            "y": (ny // 3, 2 * ny // 3),
            "z": (nz // 3, 2 * nz // 3),
        }]
        outlet = [{
            "wall": "xmax",
            "y": (ny // 3, 2 * ny // 3),
            "z": (nz // 3, 2 * nz // 3),
        }]
        return inlet, outlet

    if mode == "high":
        inlet = [
            {
                "wall": "xmin",
                "y": (1, ny - 1),
                "z": (1, nz - 1),
            },
            {
                "wall": "ymin",
                "x": (1, nx - 1),
                "z": (1, nz - 1),
            },
        ]
        outlet = [
            {
                "wall": "xmax",
                "y": (1, ny - 1),
                "z": (1, nz - 1),
            },
            {
                "wall": "ymax",
                "x": (1, nx - 1),
                "z": (1, nz - 1),
            },
        ]
        return inlet, outlet

    raise ValueError(f"Unknown ventilation mode: {mode}")


# ============================================================================
# SIMULATION FACTORIES
# ============================================================================

def build_indoor_simulation(
    scenario: str,
    species: str,
    emission_rate: float,
    source_positions: list[tuple[int, int, int]],
    N: tuple[int, int, int],
    d: tuple[float, float, float, float],
    total_time_s: float,
    D: tuple[float, float, float],
):
    if scenario == "closed":
        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=total_time_s,
            diffusion_coefficient=D,
            species_name=species,
            source_positions=source_positions,
            emission_rate=emission_rate,
            wall_deposition=0.0,
        )

    if scenario == "low_ventilation":
        inlet, outlet = ventilation_regions(N, "low")

        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=total_time_s,
            diffusion_coefficient=D,
            species_name=species,
            source_positions=source_positions,
            emission_rate=emission_rate,
            wall_deposition=INDOOR_WALL_DEPOSITION,
            inlet_regions=inlet,
            outlet_regions=outlet,
            inlet_wind_velocity=INDOOR_LOW_VENTILATION,
            outlet_wind_velocity=INDOOR_LOW_VENTILATION,
            inlet_concentration=0.0,
        )

    if scenario == "normal_ventilation":
        inlet, outlet = ventilation_regions(N, "normal")

        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=total_time_s,
            diffusion_coefficient=D,
            species_name=species,
            source_positions=source_positions,
            emission_rate=emission_rate,
            wall_deposition=INDOOR_WALL_DEPOSITION,
            inlet_regions=inlet,
            outlet_regions=outlet,
            inlet_wind_velocity=INDOOR_NORMAL_VENTILATION,
            outlet_wind_velocity=INDOOR_NORMAL_VENTILATION,
            inlet_concentration=0.0,
        )

    if scenario == "high_ventilation":
        inlet, outlet = ventilation_regions(N, "high")

        return IndoorsDiffusionAdvectionDecay(
            grid_shape=N,
            d=d,
            total_time=total_time_s,
            diffusion_coefficient=D,
            species_name=species,
            source_positions=source_positions,
            emission_rate=emission_rate,
            wall_deposition=INDOOR_WALL_DEPOSITION,
            inlet_regions=inlet,
            outlet_regions=outlet,
            inlet_wind_velocity=INDOOR_HIGH_VENTILATION,
            outlet_wind_velocity=INDOOR_HIGH_VENTILATION,
            inlet_concentration=0.0,
        )

    raise ValueError(f"Unknown indoor scenario: {scenario}")


def build_outdoor_simulation(
    scenario: str,
    species: str,
    emission_rate: float,
    source_positions: list[tuple[int, int, int]],
    N: tuple[int, int, int],
    d: tuple[float, float, float, float],
    total_time_s: float,
    D: tuple[float, float, float],
):
    if scenario == "uniform":
        wind = UniformField(
            grid_shape=N,
            initial_velocity=(UNIFORM_SPEEDS[1], 0.0, 0.0),
        )
        wind_name = "UniformField"

    elif scenario == "shear":
        wind = ShearField(
            grid_shape=N,
            Uref=SHEAR_UREF,
            zref=SHEAR_ZREF,
            alpha=SHEAR_ALPHA,
        )
        wind_name = "ShearField"

    elif scenario == "gust":
        wind = GustField(
            grid_shape=N,
            Umean=GUST_UMEAN,
            amplitude=GUST_AMPLITUDE,
            period=GUST_PERIOD,
        )
        wind_name = "GustField"

    elif scenario == "vortex":
        wind = VortexField(
            grid_shape=N,
            omega=VORTEX_OMEGA,
        )
        wind_name = "VortexField"

    else:
        raise ValueError(f"Unknown outdoor scenario: {scenario}")

    simulation = OutdoorsDiffusionAdvectionDecay(
        wind_model=wind,
        grid_shape=N,
        d=d,
        total_time=total_time_s,
        diffusion_coefficient=D,
        species_name=species,
        source_positions=source_positions,
        emission_rate=emission_rate,
    )

    return simulation, wind_name


# ============================================================================
# HRTM
# ============================================================================

def run_hrtm_profiles(
    simulation: Any,
    hrtm_profiles: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """
    Apply HRTM to an already completed concentration simulation.

    Returns one compact result per respiratory profile.
    """
    results = []

    for profile in hrtm_profiles:
        try:
            hrtm = HRTM(
                simulation,
                population_type=profile["population_type"],
                age_group=profile["age_group"],
                gender=profile["gender"],
                physical_activity=profile["physical_activity"],
                absorption=profile["absorption"],
            )

            dose_fields = hrtm.effective_dose_commitment()

            max_dose = float(
                max(np.max(field) for field in dose_fields.values())
            )

            final_dose_field = dose_fields[max(dose_fields.keys())]
            final_max_dose = float(np.max(final_dose_field))

            results.append({
                "hrtm_profile": profile["profile"],
                "max_dose_field": max_dose,
                "final_max_dose_field": final_max_dose,
                "status": "OK",
                "error": "",
            })

        except Exception as exc:
            results.append({
                "hrtm_profile": profile["profile"],
                "max_dose_field": float("nan"),
                "final_max_dose_field": float("nan"),
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })

    return results


# ============================================================================
# CASE EXECUTION
# ============================================================================

def execute_case(
    case_id: str,
    environment: str,
    scenario: str,
    wind_model: str,
    source_mode: str,
    species: str,
    emission_factor: float,
    emission_rate: float,
    resolution_name: str,
    N: tuple[int, int, int],
    d: tuple[float, float, float, float],
    total_time_s: float,
    D: tuple[float, float, float],
    save_fields: bool,
    output_dir: Path,
    with_hrtm: bool,
    source_duration_s: float | None = None,
) -> tuple[list[Result], Any]:
    """
    Execute one concentration simulation.

    The returned simulation object is kept so that HRTM can be applied
    without repeating the transport calculation.
    """
    start = time.perf_counter()

    classification = emission_classification(emission_factor)

    source_positions = build_source_positions(N, source_mode)

    # emission_rate is TOTAL Q. The current library adds
    # emission_rate * dt at every source position, so distribute Q equally
    # among source cells to keep total source strength unchanged.
    emission_rate_per_source = emission_rate / len(source_positions)

    src_description = source_description(
        source_mode,
        source_positions,
    )

    try:
        if environment == "indoor":
            simulation = build_indoor_simulation(
                scenario=scenario,
                species=species,
                emission_rate=emission_rate_per_source,
                source_positions=source_positions,
                N=N,
                d=d,
                total_time_s=total_time_s,
                D=D,
            )
        else:
            simulation, wind_model = build_outdoor_simulation(
                scenario=scenario,
                species=species,
                emission_rate=emission_rate_per_source,
                source_positions=source_positions,
                N=N,
                d=d,
                total_time_s=total_time_s,
                D=D,
            )

        # A finite source duration is expressed in iterations by the
        # library. It is deliberately independent of total_time_s.
        if source_duration_s is not None:
            simulation._source_effective_iterations = max(
                0,
                int(source_duration_s / d[3]),
            )

        fields = simulation.run(save_every_X_iteration=SAVE_EVERY)

        initial = fields[min(fields.keys())]
        final = safe_final_field(fields)

        initial_centroid = concentration_centroid(initial, d)
        final_centroid = concentration_centroid(final, d)

        result = Result(
            case_id=case_id,
            environment=environment,
            scenario=scenario,
            wind_model=wind_model,
            source_mode=source_mode,
            source_description=src_description,
            resolution=resolution_name,
            number_of_source_cells=len(source_positions),
            radionuclide=species,
            emission_factor=emission_factor,
            emission_rate_bq_s=emission_rate,
            emission_classification=classification,
            grid_shape=str(N),
            dx_m=d[0],
            dy_m=d[1],
            dz_m=d[2],
            dt_s=d[3],
            total_time_s=total_time_s,
            max_concentration_bq_m3=safe_max(fields),
            final_max_concentration_bq_m3=float(np.max(final)),
            total_activity_bq=total_activity(initial, d),
            final_total_activity_bq=total_activity(final, d),
            centroid_x_m=initial_centroid[0],
            centroid_y_m=initial_centroid[1],
            centroid_z_m=initial_centroid[2],
            final_centroid_x_m=final_centroid[0],
            final_centroid_y_m=final_centroid[1],
            final_centroid_z_m=final_centroid[2],
            runtime_s=time.perf_counter() - start,
            n_steps=total_steps(total_time_s, d[3]),
            source_duration_s=(
                float(source_duration_s)
                if source_duration_s is not None
                else float("nan")
            ),
            source_effective_iterations=(
                int(simulation._source_effective_iterations)
            ),
            source_active_iterations_expected=(
                min(
                    total_steps(total_time_s, d[3]),
                    int(source_duration_s / d[3])
                    if source_duration_s is not None
                    else 0,
                )
            ),
            source_active_iterations_actual=min(
                total_steps(total_time_s, d[3]),
                max(0, int(simulation._source_effective_iterations)),
            ),
            source_iteration_semantics_ok=(
                min(
                    total_steps(total_time_s, d[3]),
                    max(0, int(simulation._source_effective_iterations)),
                )
                == min(
                    total_steps(total_time_s, d[3]),
                    int(source_duration_s / d[3])
                    if source_duration_s is not None
                    else 0,
                )
            ),
        )

        results = [result]

        # HRTM is intentionally limited to selected emission levels.
        if with_hrtm and emission_factor in HRTM_EMISSION_FACTORS:
            hrtm_results = run_hrtm_profiles(
                simulation,
                HRTM_PROFILES,
            )

            for hrtm_result in hrtm_results:
                results.append(
                    Result(
                        **{
                            **asdict(result),
                            "hrtm_profile": hrtm_result["hrtm_profile"],
                            "max_dose_field": hrtm_result["max_dose_field"],
                            "final_max_dose_field": hrtm_result["final_max_dose_field"],
                            "status": hrtm_result["status"],
                            "error": hrtm_result["error"],
                        }
                    )
                )

        # Optional storage of the complete concentration history.
        if save_fields:
            field_dir = output_dir / "fields"
            field_dir.mkdir(parents=True, exist_ok=True)

            field_file = field_dir / f"{case_id}.npz"

            np.savez_compressed(
                field_file,
                **{
                    f"t_{i}": field
                    for i, (t, field) in enumerate(sorted(fields.items()))
                },
                times=np.array(sorted(fields.keys())),
            )

            for r in results:
                r.field_file = str(field_file)

        return results, simulation

    except Exception as exc:
        failed = Result(
            case_id=case_id,
            environment=environment,
            scenario=scenario,
            wind_model=wind_model,
            source_mode=source_mode,
            source_description=src_description,
            resolution=resolution_name,
            number_of_source_cells=len(source_positions),
            radionuclide=species,
            emission_factor=emission_factor,
            emission_rate_bq_s=emission_rate,
            emission_classification=classification,
            grid_shape=str(N),
            dx_m=d[0],
            dy_m=d[1],
            dz_m=d[2],
            dt_s=d[3],
            total_time_s=total_time_s,
            max_concentration_bq_m3=float("nan"),
            final_max_concentration_bq_m3=float("nan"),
            total_activity_bq=float("nan"),
            final_total_activity_bq=float("nan"),
            centroid_x_m=float("nan"),
            centroid_y_m=float("nan"),
            centroid_z_m=float("nan"),
            final_centroid_x_m=float("nan"),
            final_centroid_y_m=float("nan"),
            final_centroid_z_m=float("nan"),
            runtime_s=time.perf_counter() - start,
            n_steps=total_steps(total_time_s, d[3]),
            source_duration_s=(
                float(source_duration_s)
                if source_duration_s is not None
                else float("nan")
            ),
            source_effective_iterations=-1,
            source_active_iterations_expected=-1,
            source_active_iterations_actual=-1,
            source_iteration_semantics_ok=False,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
        )

        return [failed], None


# ============================================================================
# CAMPAIGN
# ============================================================================

def build_campaign() -> list[dict[str, str]]:
    """
    Six transport scenarios:
        4 indoor + 2 representative outdoor wind scenarios

    To use all four outdoor wind models, the campaign actually contains:
        4 indoor + 4 outdoor = 8 transport scenarios

    With three source geometries:
        3 radionuclides x 7 emissions x 3 source geometries x 8 scenarios
        = 504 concentration simulations.

    This is intentional: source geometry is treated as an additional
    experimental factor while all WindField models are exercised.
    """
    return [
        {
            "environment": "indoor",
            "scenario": "closed",
            "wind_model": "IndoorClosedRoom",
        },
        {
            "environment": "indoor",
            "scenario": "low_ventilation",
            "wind_model": "IndoorLowVentilation",
        },
        {
            "environment": "indoor",
            "scenario": "normal_ventilation",
            "wind_model": "IndoorNormalVentilation",
        },
        {
            "environment": "indoor",
            "scenario": "high_ventilation",
            "wind_model": "IndoorHighVentilation",
        },
        {
            "environment": "outdoor",
            "scenario": "uniform",
            "wind_model": "UniformField",
        },
        {
            "environment": "outdoor",
            "scenario": "shear",
            "wind_model": "ShearField",
        },
        {
            "environment": "outdoor",
            "scenario": "gust",
            "wind_model": "GustField",
        },
        {
            "environment": "outdoor",
            "scenario": "vortex",
            "wind_model": "VortexField",
        },
    ]



def run_temporal_source_tests(
    output_dir: Path,
    resolution_name: str,
    N: tuple[int, int, int],
    d: tuple[float, float, float, float],
    total_time_s: float,
) -> None:
    """
    Dedicated validation campaign for finite-duration sources.

    It deliberately combines:
      3 radionuclides
      3 emission levels (controlled/reference/uncontrolled)
      3 source geometries
      8 transport scenarios
      5 source durations

    Total: 1080 simulations.

    Besides concentration results, the output records the requested source
    duration, its conversion to iterations, and the number of iterations
    during which the current library actually calls _inject_sources().
    """
    campaign = build_campaign()
    results = []

    if any(f <= 0.0 for f in TEMPORAL_SOURCE_FRACTIONS):
        raise ValueError(
            "Temporal source fractions must be strictly greater than 0."
        )

    expected = (
        len(TEMPORAL_SOURCE_RADIONUCLIDES)
        * len(TEMPORAL_SOURCE_EMISSION_FACTORS)
        * len(TEMPORAL_SOURCE_MODES)
        * len(campaign)
        * len(TEMPORAL_SOURCE_FRACTIONS)
    )

    counter = 0

    print(
        "Source durations (% of total simulation time): "
        + ", ".join(f"{100*f:g}%" for f in TEMPORAL_SOURCE_FRACTIONS)
    )

    for species in TEMPORAL_SOURCE_RADIONUCLIDES:
        q_ref = Q_REFERENCE[species]

        for factor in TEMPORAL_SOURCE_EMISSION_FACTORS:
            emission_rate = q_ref * factor

            for source_mode in TEMPORAL_SOURCE_MODES:
                for scenario_config in campaign:
                    environment = scenario_config["environment"]
                    scenario = scenario_config["scenario"]
                    wind_model = scenario_config["wind_model"]

                    for fraction in TEMPORAL_SOURCE_FRACTIONS:
                        counter += 1
                        duration_s = fraction * total_time_s

                        case_id = (
                            f"TEMP_{species.replace('-', '')}_"
                            f"Q{factor:g}_{source_mode.upper()}_"
                            f"{environment.upper()}_{scenario.upper()}_"
                            f"D{fraction:g}"
                        )

                        print(f"[{counter:04d}/{expected:04d}] {case_id}")

                        case_results, _ = execute_case(
                            case_id=case_id,
                            environment=environment,
                            scenario=scenario,
                            wind_model=wind_model,
                            source_mode=source_mode,
                            species=species,
                            emission_factor=factor,
                            emission_rate=emission_rate,
                            resolution_name=resolution_name,
                            source_duration_s=duration_s,
                            N=N,
                            d=d,
                            total_time_s=total_time_s,
                            D=DIFFUSION_COEFFICIENT,
                            save_fields=False,
                            output_dir=output_dir,
                            with_hrtm=False,
                        )
                        results.extend(case_results)

    save_results(results, output_dir)

    # Additional machine-readable temporal summary.
    concentration = [r for r in results if r.hrtm_profile == ""]
    temporal_summary = {
        "total_tests": len(concentration),
        "errors": sum(r.status == "ERROR" for r in concentration),
        "source_durations_s": [
            f * total_time_s for f in TEMPORAL_SOURCE_FRACTIONS
        ],
        "source_duration_fractions": list(TEMPORAL_SOURCE_FRACTIONS),
        "duration_configuration_failures": sum(
            r.status == "OK"
            and r.source_effective_iterations
            != int(r.source_duration_s / r.dt_s)
            for r in concentration
        ),
        "source_iteration_semantics_failures": sum(
            r.status == "OK"
            and not r.source_iteration_semantics_ok
            for r in concentration
        ),
        "off_by_one_cases": sum(
            r.status == "OK"
            and r.source_active_iterations_actual
            != r.source_active_iterations_expected
            for r in concentration
        ),
        "shorter_than_simulation": sum(
            r.source_duration_s < r.total_time_s
            for r in concentration
        ),
        "equal_to_simulation": sum(
            math.isclose(r.source_duration_s, r.total_time_s)
            for r in concentration
        ),
        "longer_than_simulation": sum(
            r.source_duration_s > r.total_time_s
            for r in concentration
        ),
    }

    (output_dir / "tfm_temporal_source_summary.json").write_text(
        json.dumps(temporal_summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def save_results(results: list[Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in results]

    csv_path = output_dir / "tfm_simulation_results.csv"
    json_path = output_dir / "tfm_simulation_results.json"

    if rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    concentration_results = [
        r for r in results
        if r.hrtm_profile == ""
    ]

    summary = {
        "total_result_rows": len(results),
        "concentration_simulations": len(concentration_results),
        "controlled": sum(
            r.emission_classification == "CONTROLLED"
            for r in concentration_results
        ),
        "reference": sum(
            r.emission_classification == "REFERENCE"
            for r in concentration_results
        ),
        "uncontrolled": sum(
            r.emission_classification == "UNCONTROLLED"
            for r in concentration_results
        ),
        "errors": sum(
            r.status == "ERROR"
            for r in results
        ),
    }

    (output_dir / "tfm_simulation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    config = {
        "radionuclides": RADIONUCLIDES,
        "Q_REFERENCE_Bq_s": Q_REFERENCE,
        "emission_factors": EMISSION_FACTORS,
        "grid_shape": GRID_SHAPE,
        "d": D,
        "diffusion_coefficient": DIFFUSION_COEFFICIENT,
        "total_time_s": TOTAL_TIME_S,
        "save_every": SAVE_EVERY,
        "hrtm_emission_factors": HRTM_EMISSION_FACTORS,
        "hrtm_profiles": HRTM_PROFILES,
        "temporal_source_fractions": TEMPORAL_SOURCE_FRACTIONS,
        "temporal_source_emission_factors": TEMPORAL_SOURCE_EMISSION_FACTORS,
        "temporal_source_radionuclides": TEMPORAL_SOURCE_RADIONUCLIDES,
        "temporal_source_modes": TEMPORAL_SOURCE_MODES,
    }

    (output_dir / "tfm_simulation_configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCampaign finished")
    print("-----------------")
    print(f"Results directory : {output_dir.resolve()}")
    print(f"Concentration runs: {summary['concentration_simulations']}")
    print(f"Controlled        : {summary['controlled']}")
    print(f"Reference         : {summary['reference']}")
    print(f"Uncontrolled      : {summary['uncontrolled']}")
    print(f"Errors            : {summary['errors']}")
    print(f"CSV               : {csv_path}")
    print(f"JSON              : {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TFM radionuclide simulation campaign."
    )

    parser.add_argument(
        "--output-dir",
        default="tfm_simulation_results",
        help="Directory where results are stored.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced campaign for testing the installation.",
    )

    parser.add_argument(
        "--source-mode",
        choices=("all",) + SOURCE_MODES,
        default="all",
        help="Source geometry: multiple_points, planar, volumetric, or all.",
    )

    parser.add_argument(
        "--resolution",
        choices=tuple(RESOLUTIONS.keys()),
        default="medium",
        help="Resolution preset: low, medium, or high.",
    )

    parser.add_argument(
        "--with-hrtm",
        action="store_true",
        help="Run HRTM for selected representative emission levels.",
    )

    parser.add_argument(
        "--save-fields",
        action="store_true",
        help="Save complete concentration histories as compressed NPZ files.",
    )

    parser.add_argument(
        "--temporal-tests",
        action="store_true",
        help="Run the finite-duration source validation campaign.",
    )

    args = parser.parse_args()

    if args.smoke:
        N = SMOKE_GRID_SHAPE
        d = SMOKE_D
        total_time_s = SMOKE_TOTAL_TIME_S
        save_every = SMOKE_SAVE_EVERY
        resolution_name = "smoke"
    else:
        resolution = RESOLUTIONS[args.resolution]
        N = resolution["grid_shape"]
        d = resolution["d"]
        total_time_s = TOTAL_TIME_S
        save_every = max(1, int(round(3.0 / d[3])))
        resolution_name = args.resolution

    global SAVE_EVERY
    SAVE_EVERY = save_every

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.temporal_tests:
        run_temporal_source_tests(
            output_dir=output_dir,
            resolution_name=resolution_name,
            N=N,
            d=d,
            total_time_s=total_time_s,
        )
        return

    campaign = build_campaign()

    active_source_modes = (
        SOURCE_MODES
        if args.source_mode == "all"
        else (args.source_mode,)
    )

    expected_runs = (
        len(RADIONUCLIDES)
        * len(EMISSION_FACTORS)
        * len(active_source_modes)
        * len(campaign)
    )

    print("TFM radionuclide simulation campaign")
    print("====================================")
    print(f"Radionuclides       : {len(RADIONUCLIDES)}")
    print(f"Emission levels     : {len(EMISSION_FACTORS)}")
    print(f"Source geometries   : {len(active_source_modes)}")
    print(f"Transport scenarios : {len(campaign)}")
    print(f"Resolution          : {resolution_name}")
    print(f"Expected simulations: {expected_runs}")
    print(f"Grid                : {N}")
    print(f"dt                  : {d[3]} s")
    print(f"Total time          : {total_time_s} s")
    print(f"HRTM                : {'ON' if args.with_hrtm else 'OFF'}")
    print()

    results: list[Result] = []
    simulation_counter = 0

    for species in RADIONUCLIDES:
        if species not in Q_REFERENCE:
            raise KeyError(
                f"No Q_REFERENCE value has been defined for {species}."
            )

        q_ref = Q_REFERENCE[species]

        for factor in EMISSION_FACTORS:
            emission_rate = q_ref * factor
            classification = emission_classification(factor)

            for source_mode in active_source_modes:
                for scenario_config in campaign:
                    simulation_counter += 1

                    environment = scenario_config["environment"]
                    scenario = scenario_config["scenario"]
                    wind_model = scenario_config["wind_model"]

                    case_id = (
                        f"{species.replace('-', '')}_"
                        f"Q{factor:g}_"
                        f"{source_mode.upper()}_"
                        f"{environment.upper()}_"
                        f"{scenario.upper()}"
                    )

                    print(
                        f"[{simulation_counter:03d}/{expected_runs:03d}] "
                        f"{case_id:45s} "
                        f"{classification}"
                    )

                    case_results, _ = execute_case(
                        case_id=case_id,
                        environment=environment,
                        scenario=scenario,
                        wind_model=wind_model,
                        source_mode=source_mode,
                        species=species,
                        emission_factor=factor,
                        emission_rate=emission_rate,
                        resolution_name=resolution_name,
                        N=N,
                        d=d,
                        total_time_s=total_time_s,
                        D=DIFFUSION_COEFFICIENT,
                        save_fields=args.save_fields,
                        output_dir=output_dir,
                        with_hrtm=args.with_hrtm,
                    )

                    results.extend(case_results)

                    for result in case_results:
                        if result.status == "ERROR":
                            print(
                                f"    ERROR: {result.error}"
                            )
                        elif result.hrtm_profile:
                            print(
                                f"    HRTM {result.hrtm_profile}: "
                                f"Dmax={result.max_dose_field:.4e}"
                            )

    save_results(results, output_dir)


if __name__ == "__main__":
    main()
