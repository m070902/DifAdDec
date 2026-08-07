from radioprotection import (
    OutdoorsDiffusionAdvectionDecay,
    UniformField,
    ShearField,
    GustField,
    VortexField,
    HRTM
)

grid_shape = (100, 100, 100)

d = (0.5, 0.5, 0.5, 0.1)
base_D = [0.2, 0.2, 0.2]
base_v = (0.5, 0.0, 0.0)

base_species = "U-234"
base_source = [[25,25,25]]


base_emission = 20

base_time = 20

wind_model = GustField(grid_shape = grid_shape)

simulation = OutdoorsDiffusionAdvectionDecay(d = (0.5,0.5,0.5,0.02) , diffusion_coefficient = (0.05, 0.05, 0.05), total_time= base_time , wind_model=wind_model, emission_rate=100, grid_shape=grid_shape)
simulation.run(save_every=10)
simulation.make_csv_for_instant(10)

#dosage = HRTM(simulation)
#dose = dosage.effective_dose_commitment()
#dosage.spatial_visualization(visualization_type="3d", time = 300, vertical_axis = "x", levels = [0, 10, 20, 30, 40, 50])