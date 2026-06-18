import numpy as np

from radioprotection.utils import (
    diffusion_comprobation,
    CFL_comprobation,
    lambda_for_species
)

from radioprotection.visualization import (
    check_provided_time,
    check_number_of_Z_to_check,
    define_X_Y_values,
    stablish_maximum_concentration,
    define_initial_plotting_parameters,
    define_Z_values,
    plot_3d,
    plot_2d,
    define_color_bar,
    plot_title,
    show_plot
)

class OutdoorsDiffusionAdvectionDecay:
    def __init__(
        self,
        grid_shape=(50, 50, 30),
        d=(0.5, 0.5, 0.5, 0.01),
        total_time=100.0,
        diffusion_coefficient=(1e-3, 1e-3, 1e-3),   # m²/s
        initial_velocity=(0.05, 0.0, 0.0),      # m/s
        species_name = "U-234",
        source_positions=[(25, 25, 15)],
        emission_rate=1.0,
        ):

        self.__N = grid_shape
        self.__d = d
        self.__diffusion_coefficient = diffusion_coefficient
        self.__initial_velocity = initial_velocity
        self.__species_name = species_name
        self.__lamda = lambda_for_species(species_name)
        self.__source_positions = source_positions
        self.__emission_rate = emission_rate
        self.__total_time = total_time
        self.__concentration = np.zeros(self.__N)
        self.__saved_fields = {}

        #Pensar si realmente merece la pena utilizar un campo de velocidades usando la función vista en el otro lado
        #self.__velocity = self._build_velocity_field()

        if (diffusion_comprobation(diffusion_coefficient, d) == False) or (CFL_comprobation(initial_velocity, d) == False):
            raise ValueError("The provided values for the function 'diffusion_advection_decay' do not follow the stability conditions of the equation.")

    def forward_euler_method(self):
            Dx, Dy, Dz = self.__diffusion_coefficient
            lam = self.__half_life
            dt = self.__d[3]
            dx, dy, dz = self.__d[0], self.__d[1], self.__d[2]

            C = np.zeros((self.__n[0], self.__n[1], self.__n[2]))

            for idx in self.__source:
                C[idx[0], idx[1], idx[2]] = self.__emission_rate

            self.__concentration.update({'0': C.copy()})

            total_steps = int(self.__time / dt)

            save_interval = int(300 / dt) if dt <= 300 else 1

            for t in range(1, total_steps + 1):
                Cn = C.copy()

                diffusion = dt * (
                    Dx * (Cn[2:, 1:-1, 1:-1] - 2 * Cn[1:-1, 1:-1, 1:-1] + Cn[:-2, 1:-1, 1:-1]) / dx**2 +
                    Dy * (Cn[1:-1, 2:, 1:-1] - 2 * Cn[1:-1, 1:-1, 1:-1] + Cn[1:-1, :-2, 1:-1]) / dy**2 +
                    Dz * (Cn[1:-1, 1:-1, 2:] - 2 * Cn[1:-1, 1:-1, 1:-1] + Cn[1:-1, 1:-1, :-2]) / dz**2
                )

                adv_x = self.__v[0] * dt * (Cn[1:-1, 1:-1, 1:-1] - Cn[:-2, 1:-1, 1:-1]) / dx

                adv_y = self.__v[1] * dt * (Cn[1:-1, 1:-1, 1:-1] - Cn[1:-1, :-2, 1:-1]) / dy

                adv_z = self.__v[2] * dt * (Cn[1:-1, 1:-1, 2:] - Cn[1:-1, 1:-1, 1:-1]) / dz

                C[1:-1, 1:-1, 1:-1] = Cn[1:-1, 1:-1, 1:-1] + diffusion - adv_x - adv_y - adv_z - (lam * dt * Cn[1:-1, 1:-1, 1:-1])

                C[0, :, :]  = Cn[1, :, :]
                C[-1, :, :] = Cn[-2, :, :]

                C[:, 0, :]  = Cn[:, 1, :]
                C[:, -1, :] = Cn[:, -2, :]

                C[:, :, 0]  = Cn[:, :, 1]
                C[:, :, -1] = Cn[:, :, -2]

                # Re-inyección constante de la fuente si es emisión continua (opcional, según modelo)
                # for idx in self.__source:
                #     C[idx[0], idx[1], idx[2]] += self.__emission_rate * dt

                if t % save_interval == 0:
                    current_time_seconds = int(t * dt)
                    self.__concentration.update({f'{current_time_seconds}': C.copy()})

    def spatial_visualization(self, visualization_type = "3d", vertical_axis = "z", levels = None, time = None):

        time = check_provided_time(time, self.__time, self.__concentration)

        check_number_of_Z_to_check(vertical_axis, levels)

        X, Y, aux_axis = define_X_Y_values(vertical_axis, self.__n)

        concentration_max = stablish_maximum_concentration(time, self.__concentration)

        fig, norm = define_initial_plotting_parameters()

        for i, level in enumerate(levels):

            Z = define_Z_values(self.__concentration, vertical_axis, concentration_max, time, level)

            if (visualization_type=="3d"):
                plot_3d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, concentration_max, vertical_axis_label=fr"Concentration ($\times$ ({concentration_max:.2e})$^{{-1}}$ Bq/m$^3$)", iteration = i)

            elif (visualization_type == "2d"):
                plot_2d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, iteration = i)

            else:
                raise ValueError("The provided string for visualization type is not valid.")

        define_color_bar(fig, norm, concentration_max, vertical_axis_label = fr"Concentration ($\times$ ({concentration_max:.2e})$^{{-1}}$ Bq/m$^3$)")

        plot_title(fig, f"Radioisotope = {self.__species_name} | Visualization type = {visualization_type} | Instant = {time} s | Wind speed = {self.__v} | Diffusion coefficient = {self.__diffusion_coefficient}")

        show_plot()


    def provide_variables_hrtm(self):
        return self.__concentration, self.__n, self.__v, self.__species_name, self.__diffusion_coefficient, self.__time