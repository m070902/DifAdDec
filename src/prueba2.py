import numpy as np

from radioprotection import (
    IndoorsDiffusionAdvectionDecay
)

if __name__ == "__main__":

    source = []

    for i in range(45,56,1):
        for j in range(45,56,1):
            source.append((i,j,0))



    sim = IndoorsDiffusionAdvectionDecay(grid_shape=(100, 100, 20), total_time=5000, diffusion_coefficient=(6e-3,6e-3,6e-3), species_name="Ra-226", wall_deposition=5e-5, source_positions=source, emission_rate=30, inlet_wind_velocity=0.5, outlet_wind_velocity=0.5,
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

    results = sim.run(save_every=100)

    sim.animate()
