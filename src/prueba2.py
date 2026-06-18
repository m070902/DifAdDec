import numpy as np

from radioprotection import (
    IndoorsDiffusionAdvectionDecay
)

if __name__ == "__main__":
    sim = IndoorsDiffusionAdvectionDecay(grid_shape=(30,30,20), total_time=500, diffusion_coefficient=(5e-4,5e-4,5e-4), species_name="Th-234", initial_velocity=(0.15,0.02,0.0), wall_deposition=5e-5, source_positions=[(15,15,10)], emission_rate=50 )

    results = sim.run(save_every=100)

    sim.animate(z=10)