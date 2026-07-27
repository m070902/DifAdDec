import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .diffusion_advection_decay import DiffusionAdvectionDecay

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

class OutdoorsDiffusionAdvectionDecay(DiffusionAdvectionDecay):
    def __init__(self,
        wind_model,
        grid_shape=(50, 50, 50),
        d=(0.5, 0.5, 0.5, 0.1),
        total_time=1000.0,
        diffusion_coefficient=(1e-3, 1e-3, 1e-3),   # m²/s
        species_name = "U-234",
        source_positions=[(25, 25, 25)],
        emission_rate=3.0
    ):
        super().__init__(grid_shape, d, total_time, diffusion_coefficient, species_name, source_positions, emission_rate)

        self.__wind_model = wind_model

        #if (diffusion_comprobation(self._diffusion_coefficient, self._d) == False) or (CFL_comprobation(self.__wind_velocity, self._d) == False):
        #    raise ValueError("The provided values for the function 'diffusion_advection_decay' do not follow the stability conditions of the equation.")

    def _compute_advection(self, concentration_aux: dict[str, list[float]], time: int):

        adv = np.zeros_like(concentration_aux[1:-1,1:-1,1:-1])

        vx, vy, vz = self.__wind_model.get_velocity(time = time)

        vx = vx[1:-1,1:-1,1:-1]
        vy = vy[1:-1,1:-1,1:-1]
        vz = vz[1:-1,1:-1,1:-1]

        dCdx_backward = (
            concentration_aux[1:-1,1:-1,1:-1]
            - concentration_aux[:-2,1:-1,1:-1]
        ) / self._d[0]

        dCdx_forward = (
            concentration_aux[2:,1:-1,1:-1]
            - concentration_aux[1:-1,1:-1,1:-1]
        ) / self._d[0]

        adv += np.where(
            vx >= 0,
            vx * dCdx_backward,
            vx * dCdx_forward
        )

        dCdy_backward = (
            concentration_aux[1:-1,1:-1,1:-1]
            - concentration_aux[1:-1,:-2,1:-1]
        ) / self._d[1]

        dCdy_forward = (
            concentration_aux[1:-1,2:,1:-1]
            - concentration_aux[1:-1,1:-1,1:-1]
        ) / self._d[1]

        adv += np.where(
            vy >= 0,
            vy * dCdy_backward,
            vy * dCdy_forward
        )

        dCdz_backward = (
            concentration_aux[1:-1,1:-1,1:-1]
            - concentration_aux[1:-1,1:-1,:-2]
        ) / self._d[2]

        dCdz_forward = (
            concentration_aux[1:-1,1:-1,2:]
            - concentration_aux[1:-1,1:-1,1:-1]
        ) / self._d[2]

        adv += np.where(
            vz >= 0,
            vz * dCdz_backward,
            vz * dCdz_forward
        )

        return adv


    def _apply_boundary_conditions_concentration(self, concentration_aux):
        self._concentration[0, :, :]  = concentration_aux[1, :, :]
        self._concentration[-1, :, :] = concentration_aux[-2, :, :]

        self._concentration[:, 0, :]  = concentration_aux[:, 1, :]
        self._concentration[:, -1, :] = concentration_aux[:, -2, :]

        self._concentration[:, :, 0]  = concentration_aux[:, :, 1]
        self._concentration[:, :, -1] = concentration_aux[:, :, -2]

    def _step_concentration(self, time):

        concentration_aux = self._concentration.copy()

        diffusion = self._compute_diffusion(concentration_aux)

        advection = self._compute_advection(concentration_aux, time)

        decay = self._lamda * concentration_aux[1:-1,1:-1,1:-1]

        self._concentration[1:-1,1:-1,1:-1] = (
            concentration_aux[1:-1,1:-1,1:-1]
            + self._d[3] * diffusion
            - self._d[3] * advection
            - self._d[3] * decay
        )

        self._concentration = np.maximum(self._concentration, 0)

        self._apply_boundary_conditions_concentration(concentration_aux)

        self._inject_sources()

    def run(self, save_every=100):

        total_steps = int(self._total_time / self._d[3])

        self._saved_fields[0.0] = self._concentration.copy()

        for n in range(1, total_steps + 1):

            current_time = n * self._d[3]

            self._step_concentration(current_time)

            if n % save_every == 0:

                self._saved_fields[current_time] = self._concentration.copy()

                print(
                    f"t = {current_time:.2f} s | "
                    f"max(C) = {np.max(self._concentration):.5e}"
                )

        return self._saved_fields

    def spatial_visualization(self, visualization_type = "3d", vertical_axis = "z", levels = [0, 10, 20, 30, 40, 50], time = None):

        time = check_provided_time(time, self._total_time, self._concentration)

        check_number_of_Z_to_check(vertical_axis, levels)

        X, Y, aux_axis = define_X_Y_values(vertical_axis, self._N)

        concentration_max = stablish_maximum_concentration(time, self._concentration)

        fig, norm = define_initial_plotting_parameters()

        for i, level in enumerate(levels):

            Z = define_Z_values(self._concentration, vertical_axis, concentration_max, time, level)

            if (visualization_type=="3d"):
                plot_3d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, concentration_max, vertical_axis_label=fr"Concentration ($\times$ ({concentration_max:.2e})$^{{-1}}$ Bq/m$^3$)", iteration = i)

            elif (visualization_type == "2d"):
                plot_2d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, iteration = i)

            else:
                raise ValueError("The provided string for visualization type is not valid.")

        define_color_bar(fig, norm, concentration_max, vertical_axis_label = fr"Concentration ($\times$ ({concentration_max:.2e})$^{{-1}}$ Bq/m$^3$)")

        plot_title(fig, f"Radioisotope = {self._species_name} | Visualization type = {visualization_type} | Instant = {time} s | Wind speed = {self.__wind_velocity} | Diffusion coefficient = {self._diffusion_coefficient}")

        show_plot()


    def provide_variables_hrtm(self):
        return self._concentration, self._n, self.__wind_velocity, self._species_name, self._diffusion_coefficient, self._time

    def animate(self, z_values=None):
        times = sorted(self._saved_fields.keys())

        # Seleccionar 6 valores de z si no se especifican
        if z_values is None:
            z_values = np.linspace(
                0,
                self._N[2] - 1,
                6,
                dtype=int
            )

        fig, axes = plt.subplots(4, 3, figsize=(12, 8))
        axes = axes.ravel()

        first = self._saved_fields[times[0]]

        ims = []
        for ax, z in zip(axes, z_values):
            im = ax.imshow(
                first[:, :, z].T,
                origin='lower',
                extent=[0, self._N[0], 0, self._N[1]],
                animated=True
            )
            ax.set_title(f"z = {z}")
            fig.colorbar(im, ax=ax)
            ims.append(im)

        suptitle = fig.suptitle(f"t = {times[0]:.2f} s")

        def update(frame):
            t = times[frame]

            for im, z in zip(ims, z_values):
                im.set_array(
                    self._saved_fields[t][:, :, z].T
                )

            suptitle.set_text(f"t = {t:.2f} s")

            return ims

        animation = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=100,
            blit=False
        )

        plt.tight_layout()
        plt.show()

        return animation