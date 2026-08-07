from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

from radioprotection.utils import (
    lambda_for_species
)

from radioprotection.visualization import (
    check_provided_time,
    check_or_stablish_Z_levels,
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

class DiffusionAdvectionDecay:
    def __init__(
        self,
        grid_shape=(50, 50, 50),
        d=(0.5, 0.5, 0.5, 0.1),
        total_time=1000.0,
        diffusion_coefficient=(1e-3, 1e-3, 1e-3),   # m²/s
        species_name = "U-234",
        source_positions=[(25, 25, 25)],
        emission_rate=3.0
        ):

        self._N = grid_shape
        self._d = d
        self._diffusion_coefficient = diffusion_coefficient
        self._species_name = species_name
        self._lamda = lambda_for_species(species_name)
        self._source_positions = source_positions
        self._emission_rate = emission_rate
        self._total_time = total_time
        self._concentration = np.zeros(self._N)
        self._saved_fields = {}


    def _compute_diffusion(self, concentration_aux: list[float]):

        return (
            self._diffusion_coefficient[0] * (
                concentration_aux[2:,1:-1,1:-1]
                - 2*concentration_aux[1:-1,1:-1,1:-1]
                + concentration_aux[:-2,1:-1,1:-1]
            ) / self._d[0]**2

            +

            self._diffusion_coefficient[1] * (
                concentration_aux[1:-1,2:,1:-1]
                - 2*concentration_aux[1:-1,1:-1,1:-1]
                + concentration_aux[1:-1,:-2,1:-1]
            ) / self._d[1]**2

            +

            self._diffusion_coefficient[2] * (
                concentration_aux[1:-1,1:-1,2:]
                - 2*concentration_aux[1:-1,1:-1,1:-1]
                + concentration_aux[1:-1,1:-1,:-2]
            ) / self._d[2]**2
        )

    def _inject_sources(self):
        for idx in self._source_positions:
            self._concentration[idx] += self._emission_rate * self._d[3]

    import csv

    def make_csv_for_instant(self, time, filename="concentration.csv"):

        if time not in self._saved_fields:
            raise ValueError(f"No concentration field stored at t = {time}")

        concentration = self._saved_fields[time]

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "x",
                "y",
                "z",
                "concentration"
            ])

            nx, ny, nz = concentration.shape

            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        writer.writerow([
                            x,
                            y,
                            z,
                            concentration[x, y, z]
                        ])

        print(f"CSV saved as '{filename}'")

    def plot_instant(self, plot_name = "Default Name", visualization_type = "3d", vertical_axis = "z", levels = None, time_to_check = None):

        if time_to_check == None: time_to_check = self._total_time

        time_to_check = check_provided_time(time_to_check, self._total_time, self._concentration)

        levels = check_or_stablish_Z_levels(self._N, vertical_axis, levels)

        X, Y, aux_axis = define_X_Y_values(vertical_axis, self._N)

        concentration_max = stablish_maximum_concentration(time_to_check, self._saved_fields)

        fig, norm = define_initial_plotting_parameters()

        for i, level in enumerate(levels):

            Z = define_Z_values(self._saved_fields, vertical_axis, concentration_max, time_to_check, level)

            if (visualization_type=="3d"):
                plot_3d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, concentration_max, vertical_axis_label=fr"Concentration ($\times$ ({concentration_max:.2e})$^{{-1}}$ Bq/m$^3$)", iteration = i)

            elif (visualization_type == "2d"):
                plot_2d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, iteration = i)

            else:
                raise ValueError("The provided string for visualization type is not valid.")

        define_color_bar(fig, norm, concentration_max, vertical_axis_label = fr"Concentration ($\times$ ({concentration_max:.2e})$^{{-1}}$ Bq/m$^3$)")

        plot_title(fig, plot_name)

        show_plot()


    def _provide_variables_hrtm(self):
        return self._concentration, self._N, self.__wind_velocity, self._species_name, self._diffusion_coefficient, self.__time

    def animate(self, plot_name="Default Name", z_values=None):
        times = sorted(self._saved_fields.keys())

        if z_values is None:
            z_values = np.linspace(
                0,
                self._N[2] - 1,
                6,
                dtype=int
            )

        fig, axes = plt.subplots(
            2, 3,
            figsize=(12, 8),
            num=plot_name
        )
        axes = axes.ravel()

        first = self._saved_fields[times[0]]

        ims = []
        for ax, z in zip(axes, z_values):
            im = ax.imshow(
                first[:, :, z].T,
                origin="lower",
                extent=[0, self._N[0], 0, self._N[1]],
                animated=True
            )
            ax.set_title(f"z = {z}")
            fig.colorbar(im, ax=ax)
            ims.append(im)

        # Título principal
        fig.suptitle(plot_name, fontsize=16, fontweight="bold")

        # Subtítulo con el tiempo
        subtitle = fig.text(
            0.5,
            0.94,
            f"t = {times[0]:.2f} s",
            ha="center",
            fontsize=12
        )

        def update(frame):
            t = times[frame]

            for im, z in zip(ims, z_values):
                im.set_array(
                    self._saved_fields[t][:, :, z].T
                )

            subtitle.set_text(f"t = {t:.2f} s")

            return ims + [subtitle]

        animation = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=100,
            blit=False
        )

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

        return animation