import numpy as np

from DifAdDec import (
    IndoorsDiffusionAdvectionDecay
)

if __name__ == "__main__":

    sim = IndoorsDiffusionAdvectionDecay(grid_shape=(100, 100, 20), total_time=250, diffusion_coefficient=(6e-3,6e-3,6e-3), species_name="Ra-226", wall_deposition=5e-5, source_positions=[(50, 50, 10)], emission_rate=30, inlet_wind_velocity=0.5, outlet_wind_velocity=0.5,
    inlet_regions=[
        {
            "wall":"xmin",
            "y":(45,55),
            "z":(5,15)
        }
    ],
    outlet_regions=[
        {
            "wall":"zmax",
            "x":(45,55),
            "y":(45,55)
        }
    ])

    results = sim.run(save_every_X_iteration=100)

    sim.animate(z_values=[8,9,10,11,12,13])
