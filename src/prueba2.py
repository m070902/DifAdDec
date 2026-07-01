import numpy as np

from radioprotection import (
    IndoorsDiffusionAdvectionDecay
)

if __name__ == "__main__":

    source = []

    for i in range(45,56,1):
        for j in range(45,56,1):
            source.append((i,j,0))



    sim = IndoorsDiffusionAdvectionDecay(grid_shape=(100, 100, 30), total_time=3000, diffusion_coefficient=(8e-3,8e-3,8e-3), species_name="Ra-226", initial_velocity=(0.00,0.00,0.00), wall_deposition=5e-5, source_positions=source, emission_rate=70, inlet_velocity=0.5, outlet_velocity=0.5,
    inlet_regions=[
        {
            "wall":"xmin",
            "y":(30,70),
            "z":(10,20)
        }
    ],
    outlet_regions=[
        {
            "wall":"zmax",
            "x":(40,60),
            "y":(40,60)
        }
    ])

    results = sim.run(save_every=100)

    sim.animate()
