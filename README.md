# DifAdDec

Python library for the numerical simulation of radioactive contaminant transport through **diffusion, advection and radioactive decay**, with additional tools for indoor and outdoor environments, wind-field modelling, visualization, data export and inhalation dose estimation using the Human Respiratory Tract Model (HRTM).

---

## Table of Contents

* [Overview](#overview)
* [Main capabilities](#main-capabilities)
* [Library structure](#library-structure)
* [Installation](#installation)
* [Basic workflow](#basic-workflow)
* [1. Define the simulation domain](#1-define-the-simulation-domain)
* [2. Define radioactive sources](#2-define-radioactive-sources)
* [3. Run a simulation](#3-run-a-simulation)
* [4. Visualize concentration fields](#4-visualize-concentration-fields)
* [5. Export concentration data](#5-export-concentration-data)
* [Indoor simulations](#indoor-simulations)
* [Outdoor simulations](#outdoor-simulations)
* [Wind-field models](#wind-field-models)
* [Dose calculation with HRTM](#dose-calculation-with-hrtm)
* [Exporting dose results](#exporting-dose-results)
* [Animations](#animations)
* [Numerical stability](#numerical-stability)
* [Complete examples](#complete-examples)
* [Recommended workflow](#recommended-workflow)
* [Physical units](#physical-units)
* [Limitations and considerations](#limitations-and-considerations)
* [License](#license)

---

# Overview

`DifAdDec` is a Python library developed to simulate the transport and radioactive decay of airborne radioactive contaminants in three-dimensional environments.

The library solves a numerical formulation of the diffusion-advection-decay equation:

$$
\frac{\partial C}{\partial t}=D_x\frac{\partial^2 C}{\partial x^2} + D_y\frac{\partial^2 C}{\partial y^2} + D_z\frac{\partial^2 C}{\partial z^2} - \vec{v}\cdot\nabla C \lambda C + S
$$

where:

* $C$ is the radioactive concentration.
* $D_x$, $D_y$ and $D_z$ are the diffusion coefficients.
* $\vec{v}$ is the velocity field.
* $\lambda$ is the radioactive decay constant.
* $S$ represents radioactive sources.

The computational domain is discretized as a three-dimensional Cartesian grid.

The library is designed to support simulations in different environments:

* **General diffusion-advection-decay simulations**
* **Indoor environments**
* **Outdoor environments**
* **Different wind-field configurations**
* **Multiple radioactive sources**
* **Concentration visualization**
* **Concentration data export**
* **Inhalation dose estimation**

---

# Main capabilities

The main functionalities of `DifAdDec` are:

### Transport simulation

The library can simulate:

* Diffusion
* Advection
* Radioactive decay
* Continuous radioactive emission
* Three-dimensional concentration distributions

### Indoor environments

Indoor simulations additionally support:

* Wall deposition
* Inlets
* Outlets
* Ventilation velocities
* Inlet concentrations
* Automatically generated velocity fields for the enclosed domain

### Outdoor environments

Outdoor simulations support externally defined wind models, allowing the use of different spatial and temporal velocity fields.

### Wind models

Several wind-field models are available:

* Uniform wind
* Vertical shear
* Gusting wind
* Vortex flow

### Visualization

Simulation results can be represented as:

* 2D concentration maps
* 3D concentration maps
* Time-dependent animations

### Data export

Concentration and dose fields can be exported to CSV files.

### Dosimetry

The `HRTM` class can use the saved concentration fields from a transport simulation to calculate inhalation dose distributions based on:

* Population type
* Age group
* Gender
* Physical activity
* Absorption type
* Exposure time

---

# Library structure

A typical `DifAdDec` project follows this structure:

```text
DifAdDec/
│
├── diffusion_advection_decay.py
├── indoors_diffusion_advection_decay.py
├── outdoors_diffusion_advection_decay.py
├── windfield.py
├── hrtm.py
│
├── utils/
│   ├── ...
│
└── visualization/
├── ...
```

The main public components are:

| Component                         | Purpose                      |
| --------------------------------- | ---------------------------- |
| `DiffusionAdvectionDecay`         | Base transport simulation    |
| `IndoorsDiffusionAdvectionDecay`  | Indoor transport simulation  |
| `OutdoorsDiffusionAdvectionDecay` | Outdoor transport simulation |
| `WindField`                       | Base class for wind models   |
| `UniformField`                    | Uniform wind velocity        |
| `ShearField`                      | Wind varying with height     |
| `GustField`                       | Time-dependent wind          |
| `VortexField`                     | Vortex wind field            |
| `HRTM`                            | Inhalation dose calculation  |

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or on Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The library relies on scientific Python packages including:

```text
numpy
scipy
matplotlib
```

The exact dependencies should be kept synchronized with the project's `requirements.txt`.

---

# Basic workflow

A typical `DifAdDec` simulation follows these steps:

```text
1. Define the computational grid
↓
2. Define spatial and temporal discretization
↓
3. Define radioactive species
↓
4. Define radioactive sources
↓
5. Select simulation environment
↓
6. Run the simulation
↓
7. Save concentration fields
↓
8. Visualize or export results
↓
9. Calculate dose using HRTM
```

The most important principle is:

> **The transport simulation must be executed before dose calculations can be performed.**

This is because `HRTM` uses the concentration fields saved by the transport simulation.

---

# 1. Define the simulation domain

The computational domain is defined using `grid_shape` and `d`.

## Grid size

`grid_shape` defines the number of nodes in each spatial direction:

```python
grid_shape = (50, 50, 50)
```

This corresponds to:

```text
Nx = 50
Ny = 50
Nz = 50
```

The grid therefore contains:

$$
N_xN_yN_z
$$

computational nodes.

## Spatial and temporal discretization

The tuple `d` contains:

```python
d = (dx, dy, dz, dt)
```

For example:

```python
d = (0.5, 0.5, 0.5, 0.1)
```

means:

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| `dx`      |   0.5 | m    |
| `dy`      |   0.5 | m    |
| `dz`      |   0.5 | m    |
| `dt`      |   0.1 | s    |

The physical dimensions of the domain are therefore determined by both the number of grid points and the spatial discretization.

For example:

```python
grid_shape = (50, 50, 50)
d = (0.5, 0.5, 0.5, 0.1)
```

corresponds to a domain with approximately:

```text
25 m × 25 m × 25 m
```

---

# 2. Define radioactive sources

Radioactive sources are defined using:

source_positions

and:

emission_rate

For example:

source_positions = [
    (25, 25, 25)
]

emission_rate = 3.0

defines a source at the grid position:

x = 25
y = 25
z = 25

The current implementation allows both continuous sources and temporally limited sources.

## Continuous radioactive sources

By default, the source remains active throughout the simulation. At every time step, the source contribution is injected into the corresponding grid cell.

The amount introduced during each time step is:

$$
\Delta C = Q\Delta t
$$

where:

$Q$ is the emission rate.

$\Delta t$ is the simulation time step.

For example:

simulation = DiffusionAdvectionDecay(
    ...
    source_positions=[(25, 25, 25)],
    emission_rate=3.0
)

If source_effective_iterations is not specified, the source remains active for the duration of the simulation.

## Temporally limited radioactive sources

A source can also be active only during a defined number of simulation iterations using:

source_effective_iterations

For example:

simulation = DiffusionAdvectionDecay(
    ...
    source_positions=[(25, 25, 25)],
    emission_rate=3.0,
    source_effective_iterations=100
)

In this case, the source is injected only during the initial 100 simulation iterations. After this period, no additional activity is introduced by the source, while the radioactive material already present in the domain continues to evolve according to diffusion, advection and radioactive decay.

The source duration in physical time is determined by:

$$
T_s = N_s\Delta t
$$

where:

$T_s$ is the source emission duration in seconds.

$N_s$ is source_effective_iterations.

$\Delta t$ is the simulation time step.

Therefore, for a simulation with:

dt = 0.1

a source intended to remain active for 60 seconds would require:

source_effective_iterations = 600

This functionality makes it possible to represent different source-release scenarios, including:

continuous emission throughout the simulation;

emission during a finite time interval;

an initial release followed by a period without further source injection.

Important: source_effective_iterations is expressed in number of simulation iterations, rather than directly in seconds.

## Multiple point sources

Several point sources can be defined simultaneously:

source_positions = [
    (10, 10, 10),
    (25, 25, 25),
    (40, 40, 40)
]

All specified positions are injected while the source is active. If several source points are placed next to each other, they can represent an extended two- or three-dimensional source region.

Important: source positions refer to grid indices, not directly to physical coordinates in metres.

For a grid spacing of:

d = (0.5, 0.5, 0.5, 0.1)

the grid point:

(10, 10, 10)

corresponds to approximately:

(5 m, 5 m, 5 m)

# 3. Run a simulation
## General simulation

Create a simulation:

```python
simulation = (Indoors/Outdoors)DiffusionAdvectionDecay(
grid_shape=(50, 50, 50),
d=(0.5, 0.5, 0.5, 0.1),
total_time=1000.0,
diffusion_coefficient=(1e-3, 1e-3, 1e-3),
species_name="U-234",
source_positions=[(25, 25, 25)],
emission_rate=3.0
...
)
```

Then execute:

```python
results = simulation.run(
save_every_X_iteration=100
)
```

The `run()` method returns the saved concentration fields.

The results are stored internally as a dictionary where:

```text
key   → simulation time
value → 3D concentration field
```

For example:

```python
results[0.0]
```

returns the initial concentration field.

If a field was saved at 10 seconds:

```python
results[10.0]
```

returns the corresponding three-dimensional concentration distribution.

---

# 4. Visualize concentration fields

After running a simulation, a saved concentration field can be visualized using:

```python
simulation.plot_instant(
plot_name="Radioactive concentration",
visualization_type="3d",
vertical_axis="z",
time_to_check=100.0
)
```

The available visualization types are:

```python
visualization_type="3d"
```

and:

```python
visualization_type="2d"
```

For example:

```python
simulation.plot_instant(
plot_name="Concentration at t = 100 s",
visualization_type="2d",
vertical_axis="z",
time_to_check=100.0
)
```

The `vertical_axis` parameter determines which coordinate is treated as the vertical/slicing direction.

The `levels` parameter can be used to specify the grid levels to display:

```python
simulation.plot_instant(
plot_name="Selected planes",
visualization_type="3d",
vertical_axis="z",
levels=[10, 25, 40],
time_to_check=100.0
)
```

The concentration is normalized for visualization using:

$$
\frac{C}{C_{\max}}
$$

while the maximum concentration is reported in the color bar.

---

# 5. Export concentration data

A saved concentration field can be exported to CSV:

```python
simulation.make_csv_for_instant(
time=100.0,
filename="concentration_100s.csv"
)
```

The generated file contains:

```text
x (m)
y (m)
z (m)
concentration (Bq/m³)
```

The coordinates are reconstructed from the spatial discretization:

```python
dx, dy, dz = d[:3]
```

and the concentration field corresponding to the selected simulation time is exported.

If no time is provided:

```python
simulation.make_csv_for_instant(
filename="concentration.csv"
)
```

the method attempts to use the total simulation time.

However, the requested time must have been saved during `run()`.

---

# Indoor simulations

Indoor environments are represented by:

```python
IndoorsDiffusionAdvectionDecay
```

Import it with:

```python
from DifAdDec import (
IndoorsDiffusionAdvectionDecay
)
```

The indoor model extends the general diffusion-advection-decay model and adds:

* Wall deposition
* Inlets
* Outlets
* Ventilation velocities
* Inlet concentration
* An internally calculated velocity field

---

## Basic indoor example

```python
simulation = IndoorsDiffusionAdvectionDecay(
grid_shape=(50, 50, 50),
d=(0.5, 0.5, 0.5, 0.1),
total_time=1000.0,
diffusion_coefficient=(1e-3, 1e-3, 1e-3),
species_name="U-234",
source_positions=[(25, 25, 25)],
emission_rate=3.0,
wall_deposition=1e-4
)

results = simulation.run(
save_every_X_iteration=100
)
```

---

# Indoor ventilation

Ventilation openings are defined through:

```python
inlet_regions
```

and:

```python
outlet_regions
```

Each region specifies:

* The wall containing the opening
* The range of indices defining the opening

For example:

```python
inlet_regions = [
{
"wall": "xmin",
"y": (10, 20),
"z": (10, 20)
}
]
```

This defines an inlet located on the `xmin` wall.

The available wall names are:

```text
xmin
xmax
ymin
ymax
zmin
zmax
```

The same structure is used for outlet regions.

Example:

```python
outlet_regions = [
{
"wall": "xmax",
"y": (30, 40),
"z": (10, 20)
}
]
```

Ventilation velocities are specified with:

```python
inlet_wind_velocity=1.0,
outlet_wind_velocity=1.0
```

and the inlet concentration can be specified using:

```python
inlet_concentration=0.0
```

---

# Outdoor simulations

Outdoor environments are represented by:

```python
OutdoorsDiffusionAdvectionDecay
```

Import:

```python
from DifAdDec import (
OutdoorsDiffusionAdvectionDecay
)
```

Unlike the indoor model, the outdoor model receives a wind model explicitly.

The basic structure is:

```python
simulation = OutdoorsDiffusionAdvectionDecay(
wind_model=wind_model,
grid_shape=(50, 50, 50),
d=(0.5, 0.5, 0.5, 0.1),
total_time=1000.0,
diffusion_coefficient=(1e-3, 1e-3, 1e-3),
species_name="U-234",
source_positions=[(25, 25, 25)],
emission_rate=3.0
)

results = simulation.run(
save_every_X_iteration=100
)
```

---

# Wind-field models

`DifAdDec` provides several wind-field implementations.

Import them from the wind-field module:

```python
from DifAdDec import (
UniformField,
ShearField,
GustField,
VortexField
)
```

---

## Uniform wind

A uniform velocity field is defined with:

```python
wind = UniformField(
grid_shape=(50, 50, 50),
initial_velocity=(5.0, 0.0, 0.0)
)
```

This creates a velocity field where:

```text
u = 5 m/s
v = 0 m/s
w = 0 m/s
```

The velocity is constant throughout the computational domain.

---

## Vertical shear

The `ShearField` model creates a wind speed that varies with height:

```python
wind = ShearField(
grid_shape=(50, 50, 50),
Uref=5.0,
zref=10,
alpha=0.20
)
```

The horizontal velocity follows a power-law relationship with height.

This model is useful for representing wind profiles whose velocity changes vertically.

---

## Gusting wind

The `GustField` model introduces temporal variation:

```python
wind = GustField(
grid_shape=(50, 50, 50),
Umean=5.0,
amplitude=2.0,
period=120
)
```

The wind velocity varies sinusoidally with time:

$$
U(t)=U_{\mathrm{mean}}
+
A\sin\left(
\frac{2\pi t}{T}
\right)
$$

where:

* `Umean` is the mean velocity.
* `amplitude` is the oscillation amplitude.
* `period` is the oscillation period.

The velocity is recalculated during the simulation.

---

## Vortex field

A rotational velocity field can be generated using:

```python
wind = VortexField(
grid_shape=(50, 50, 50),
omega=0.02
)
```

The resulting velocity field rotates around the central vertical axis of the computational domain.

---

# Complete outdoor example

```python
from DifAdDec import (
OutdoorsDiffusionAdvectionDecay
)

from DifAdDec import UniformField

grid_shape = (50, 50, 50)

d = (
0.5,   # dx [m]
0.5,   # dy [m]
0.5,   # dz [m]
0.1    # dt [s]
)

wind = UniformField(
grid_shape=grid_shape,
initial_velocity=(5.0, 0.0, 0.0)
)

simulation = OutdoorsDiffusionAdvectionDecay(
wind_model=wind,
grid_shape=grid_shape,
d=d,
total_time=1000.0,
diffusion_coefficient=(1e-3, 1e-3, 1e-3),
species_name="U-234",
source_positions=[
(25, 25, 25)
],
emission_rate=3.0
)

results = simulation.run(
save_every_X_iteration=100
)

simulation.plot_instant(
plot_name="Outdoor radioactive dispersion",
visualization_type="3d",
vertical_axis="z",
time_to_check=100.0
)
```

---

# Dose calculation with HRTM

Once a transport simulation has been completed, the concentration fields can be passed to the `HRTM` class.

Import:

```python
from DifAdDec import HRTM
```

Create an HRTM object using the completed simulation:

```python
hrtm = HRTM(
simulation,
population_type="public",
age_group="adult",
gender="male",
physical_activity="sitting",
absorption="F",
exposition_time=1000.0
)
```

The HRTM object obtains the concentration fields, radionuclide species and simulation information directly from the transport simulation.

---

## Calculate effective dose commitment

Run:

```python
dose = hrtm.effective_dose_commitment()
```

The method uses:

1. The simulated concentration fields.
2. The selected population characteristics.
3. The breathing rate.
4. The inhalation dose coefficients.
5. The exposure time.

The resulting dose fields are stored internally and returned by the method.

Conceptually, the workflow is:

```text
Concentration field
↓
Breathing rate
↓
Inhaled activity
↓
Inhalation dose coefficient
↓
Dose field
```

---

# HRTM parameters

The main parameters are:

| Parameter           | Purpose                                                |
| ------------------- | ------------------------------------------------------ |
| `population_type`   | Defines the population category                        |
| `age_group`         | Defines the age group                                  |
| `gender`            | Defines the gender used for the respiratory parameters |
| `physical_activity` | Defines the breathing-rate condition                   |
| `absorption`        | Defines the selected absorption type                   |
| `exposition_time`   | Exposure duration                                      |

For example:

```python
hrtm = HRTM(
simulation,
population_type="public",
age_group="adult",
gender="male",
physical_activity="sitting",
absorption="F",
exposition_time=600.0
)
```

If `exposition_time` is not provided, the simulation total time is used.

---

# Exporting dose results

After:

```python
hrtm.effective_dose_commitment()
```

the calculated dose field can be exported:

```python
hrtm.make_csv_for_instant(
time=100.0,
filename="dose_100s.csv"
)
```

The CSV contains:

```text
x (m)
y (m)
z (m)
dose (Bq)
```

The requested time must correspond to a dose field stored in the HRTM results.

---

# Dose visualization

The dose field can be visualized using:

```python
hrtm.plot_instant(
plot_name="Inhalation dose",
visualization_type="3d",
vertical_axis="z",
time_to_check=100.0
)
```

As with the concentration visualization, both 2D and 3D representations are supported.

---

# Animations

Both transport simulations and HRTM dose calculations provide an `animate()` method.

For example:

```python
simulation.animate(
plot_name="Concentration evolution"
)
```

The animation displays several horizontal slices of the three-dimensional field while the simulation time changes.

Specific `z` levels can be provided:

```python
simulation.animate(
plot_name="Concentration evolution",
z_values=[5, 10, 20, 30, 40, 45]
)
```

The same approach can be used for HRTM:

```python
hrtm.animate(
plot_name="Dose evolution",
z_values=[5, 10, 20, 30, 40, 45]
)
```

---

# Numerical stability

The numerical methods impose stability constraints on the selected discretization.

`DifAdDec` performs stability checks for:

* Diffusion
* Advection/CFL conditions

These checks are performed automatically when the corresponding simulation objects are initialized.

Therefore, the following parameters should not be selected independently:

```python
dx
dy
dz
dt
diffusion_coefficient
wind_velocity
```

For example, increasing the wind velocity may require a smaller temporal step `dt`.

Similarly, increasing the diffusion coefficient may impose stricter constraints on the spatial and temporal discretization.

If the stability conditions are not satisfied, the library raises a `ValueError`.

---

# Complete workflow example

The following example illustrates the recommended workflow from transport simulation to dose calculation.

```python
from DifAdDec import (
OutdoorsDiffusionAdvectionDecay
UniformField
HRTM
)

# --------------------------------------------------
# 1. Simulation configuration
# --------------------------------------------------

grid_shape = (50, 50, 50)

d = (
0.5,   # dx [m]
0.5,   # dy [m]
0.5,   # dz [m]
0.1    # dt [s]
)

total_time = 1000.0

# --------------------------------------------------
# 2. Wind field
# --------------------------------------------------

wind = UniformField(
grid_shape=grid_shape,
initial_velocity=(5.0, 0.0, 0.0)
)

# --------------------------------------------------
# 3. Transport model
# --------------------------------------------------

simulation = OutdoorsDiffusionAdvectionDecay(
wind_model=wind,
grid_shape=grid_shape,
d=d,
total_time=total_time,
diffusion_coefficient=(
1e-3,
1e-3,
1e-3
),
species_name="U-234",
source_positions=[
(25, 25, 25)
],
emission_rate=3.0
)

# --------------------------------------------------
# 4. Run simulation
# --------------------------------------------------

results = simulation.run(
save_every_X_iteration=100
)

# --------------------------------------------------
# 5. Visualize concentration
# --------------------------------------------------

simulation.plot_instant(
plot_name="Radioactive dispersion",
visualization_type="3d",
vertical_axis="z",
time_to_check=100.0
)

# --------------------------------------------------
# 6. Export concentration
# --------------------------------------------------

simulation.make_csv_for_instant(
time=100.0,
filename="concentration_100s.csv"
)

# --------------------------------------------------
# 7. Create HRTM model
# --------------------------------------------------

hrtm = HRTM(
simulation,
population_type="public",
age_group="adult",
gender="male",
physical_activity="sitting",
absorption="F",
exposition_time=100.0
)

# --------------------------------------------------
# 8. Calculate dose
# --------------------------------------------------

dose = hrtm.effective_dose_commitment()

# --------------------------------------------------
# 9. Visualize dose
# --------------------------------------------------

hrtm.plot_instant(
plot_name="Inhalation dose",
visualization_type="3d",
vertical_axis="z",
time_to_check=100.0
)

# --------------------------------------------------
# 10. Export dose
# --------------------------------------------------

hrtm.make_csv_for_instant(
time=100.0,
filename="dose_100s.csv"
)
```

---

# Recommended workflow

For most applications, the recommended workflow is:

### Step 1 — Select the environment

Choose between:

```text
DiffusionAdvectionDecay
IndoorsDiffusionAdvectionDecay
OutdoorsDiffusionAdvectionDecay
```

### Step 2 — Define the grid

Choose:

```python
grid_shape
d
```

according to the physical dimensions and desired spatial resolution.

### Step 3 — Select the radionuclide

Specify:

```python
species_name
```

The radioactive decay constant is obtained internally from the library.

### Step 4 — Define the sources

Specify:

```python
source_positions
emission_rate
```

### Step 5 — Configure transport

Set:

```python
diffusion_coefficient
```

and, when appropriate, a wind model or ventilation configuration.

### Step 6 — Check numerical stability

The library performs the stability checks automatically.

### Step 7 — Run the simulation

```python
results = simulation.run(
save_every_X_iteration=100
)
```

### Step 8 — Inspect the concentration field

Use:

```python
plot_instant()
```

and/or:

```python
animate()
```

### Step 9 — Export results

Use:

```python
make_csv_for_instant()
```

### Step 10 — Calculate dose

Create an `HRTM` object and execute:

```python
hrtm.effective_dose_commitment()
```

---

# Physical units

The current implementation uses SI units for the principal physical quantities.

| Quantity                   | Unit  |
| -------------------------- | ----- |
| Spatial coordinates        | m     |
| Time                       | s     |
| Diffusion coefficient      | m²/s  |
| Wind velocity              | m/s   |
| Wall deposition velocity   | m/s   |
| Concentration              | Bq/m³ |
| Radioactive decay constant | s⁻¹   |

The spatial discretization:

```python
d = (dx, dy, dz, dt)
```

must therefore be defined consistently with these units.

---

# Important implementation considerations

## Grid indices vs physical coordinates

Source positions are specified using grid indices:

```python
source_positions=[
(i, j, k)
]
```

rather than physical coordinates.

The physical position is:

$$
x=i\Delta x
$$

$$
y=j\Delta y
$$

$$
z=k\Delta z
$$

---

## Saved simulation times

Only concentration fields saved by:

```python
run(save_every_X_iteration=...)
```

are available for visualization, export and subsequent HRTM processing.

For example:

```python
simulation.run(
save_every_X_iteration=100
)
```

does not necessarily save every time step.

Consequently, requesting an unsaved time:

```python
simulation.plot_instant(
time_to_check=37.0
)
```

may result in an error if that time is not present in the saved fields.

---

## Memory requirements

Three-dimensional concentration fields can require significant memory.

Saving many fields with:

```python
save_every_X_iteration=1
```

can result in a large memory footprint.

For long simulations or high-resolution grids, it is therefore advisable to select an appropriate saving interval.

For example:

```python
save_every_X_iteration=100
```

stores considerably fewer fields than:

```python
save_every_X_iteration=1
```

---

# Limitations and considerations

This README documents the current implementation of `DifAdDec`. The numerical model and its physical interpretation should be considered carefully before applying the results to real radiological protection assessments.

In particular:

* Numerical stability depends on the selected discretization and transport parameters.
* Source positions are defined by grid indices.
* Saved fields determine which times can subsequently be visualized or exported.
* Increasing grid resolution significantly increases computational and memory requirements.
* The physical parameters supplied by the user should be consistent in units and representative of the scenario being simulated.
* Dose calculations depend on the concentration fields generated by the transport simulation and on the HRTM input parameters.

For research applications, model assumptions, numerical convergence and sensitivity to discretization should be assessed before interpreting simulation results.

---

# Contact

For questions, suggestions or issues related to `DifAdDec`, please use the GitHub issue tracker associated with this repository.