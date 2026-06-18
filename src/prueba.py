from radioprotection import (
    IndoorsDiffusionAdvectionDecay,
    HRTM
)

n = (50, 50, 50)

d = (0.5, 0.5, 0.5, 0.1)
base_D = [0.2, 0.2, 0.2]
base_v = (0.5, 0.0, 0.0)

base_species = "U-234"
base_source = [[25,25,25]]


base_emission = 20

base_time = 300

simulation = IndoorsDiffusionAdvectionDecay(
    n=n,
    d=d,
    D=base_D,
    v=base_v,
    species_name=base_species,
    source=base_source,
    emission_rate=base_emission,
    time = base_time
)
simulation.forward_euler_method()

dosage = HRTM(simulation)
dose = dosage.effective_dose_commitment()
dosage.spatial_visualization(visualization_type="3d", time = 300, vertical_axis = "x", levels = [0, 10, 20, 30, 40, 50])