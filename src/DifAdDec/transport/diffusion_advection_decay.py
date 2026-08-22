from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

from DifAdDec.utils import (
    lambda_for_species
)

from DifAdDec.visualization import (
    check_provided_time,
    check_or_establish_Z_levels,
    define_X_Y_values,
    establish_maximum_concentration,
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
        source_effective_iterations = None,
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
        if source_effective_iterations == None:
            self._source_effective_iterations = total_time*d[3]
        else:
            self._source_effective_iterations = source_effective_iterations


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

    def make_csv_for_instant(self, time = None, filename="concentration.csv"):

        if time == None:
            time = self._total_time

        if time not in self._saved_fields:
            raise ValueError(f"No concentration field stored at t = {time}")

        concentration = self._saved_fields[time]

        dx, dy, dz = self._d[:3]

        x, y, z = np.indices(concentration.shape)

        data = np.column_stack((
            x.ravel() * dx,
            y.ravel() * dy,
            z.ravel() * dz,
            concentration.ravel()
        ))

        np.savetxt(
            filename,
            data,
            delimiter=",",
            header="x (m),y (m),z (m),concentration (Bq/m³)",
            comments=""
        )

        print(f"CSV saved as '{filename}'")

    def plot_instant(
        self,
        plot_name="Default Name",
        visualization_type="3d",
        vertical_axis="z",
        levels=None,
        time_to_check=None
    ):

        time_to_check = check_provided_time(
            time_to_check,
            self._saved_fields
        )

        levels = check_or_establish_Z_levels(
            self._N,
            vertical_axis,
            levels
        )

        X, Y, aux_axis = define_X_Y_values(
            vertical_axis,
            self._N,
            self._d
        )

        concentration_max = establish_maximum_concentration(
            self._saved_fields
        )

        fig, norm = define_initial_plotting_parameters()

        gs = fig.add_gridspec(
            2, 4,
            width_ratios=[1, 1, 1, 0.08],
            wspace=0.35,
            hspace=0.35)

        for i, level in enumerate(levels):

            Z = define_Z_values(
                self._saved_fields,
                vertical_axis,
                concentration_max,
                time_to_check,
                level
            )

            if visualization_type == "3d":

                plot_3d(
                    X,
                    Y,
                    Z,
                    fig,
                    norm,
                    vertical_axis,
                    level,
                    aux_axis,
                    concentration_max,
                    vertical_axis_label=(
                        r"Normalized concentration "
                        r"$C/C_{\max}$"
                    ),
                    d=self._d,
                    gs = gs,
                    iteration=i
                )

            elif visualization_type == "2d":

                plot_2d(
                    X,
                    Y,
                    Z,
                    fig,
                    norm,
                    vertical_axis,
                    level,
                    aux_axis,
                    d=self._d,
                    gs=gs,
                    iteration=i
                )

            else:

                raise ValueError(
                    "The provided string for visualization type is not valid."
                )

        fig.subplots_adjust(
            left=0.18,
            right=0.76,
            bottom=0.10,
            top=0.92,
            wspace=0.30,
            hspace=0.30
        )

        define_color_bar(
            fig,
            norm,
            colorbar_label = (
                rf"Normalized concentration $C/C_{{\max}}$"
                "\n"
                rf"$C_{{\max}} = {concentration_max:.2e}\ "
                rf"\mathrm{{Bq\,m^{{-3}}}}$"
            ),
            gs = gs
        )

        plot_title(
            fig,
            plot_name
        )

        show_plot()

    def animate(self, plot_name="Default Name", z_values=None):

        times = sorted(self._saved_fields.keys())

        if len(times) == 0:
            raise RuntimeError(
                "No saved concentration fields available."
            )

        if z_values is None:
            z_values = np.linspace(
                0,
                self._N[2] - 1,
                6,
                dtype=int
            )

        if len(z_values) != 6:
            raise ValueError(
                "animate() requires exactly 6 z-values."
            )

        concentration_max = max(
            np.max(field)
            for field in self._saved_fields.values()
        )

        if concentration_max == 0:
            concentration_max = 1.0

        x = np.arange(self._N[0]) * self._d[0]
        y = np.arange(self._N[1]) * self._d[1]

        extent = [
            x[0],
            x[-1],
            y[0],
            y[-1]
        ]


        fig = plt.figure(
            figsize=(14, 10),
            num=plot_name,
            clear=True
        )

        axes = [
            fig.add_axes([0.08, 0.56, 0.20, 0.28]),
            fig.add_axes([0.35, 0.56, 0.20, 0.28]),
            fig.add_axes([0.62, 0.56, 0.20, 0.28]),

            fig.add_axes([0.08, 0.12, 0.20, 0.28]),
            fig.add_axes([0.35, 0.12, 0.20, 0.28]),
            fig.add_axes([0.62, 0.12, 0.20, 0.28]),
        ]

        cax = fig.add_axes(
            [0.87, 0.16, 0.025, 0.68]
        )


        first = (
            self._saved_fields[times[0]]
            / concentration_max
        )

        ims = []


        for ax, z in zip(axes, z_values):

            im = ax.imshow(
                first[:, :, z].T,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="equal",
                animated=True
            )

            ax.set_title(
                f"z = {z * self._d[2]:.2f} m",
                fontsize=12,
                pad=8
            )

            ax.set_xlabel(
                "x (m)",
                labelpad=6
            )

            ax.set_ylabel(
                "y (m)",
                labelpad=6
            )

            ims.append(im)

        cbar = fig.colorbar(
            ims[0],
            cax=cax
        )

        cbar.set_label(
            rf"Normalized concentration $C/C_{{\max}}$"
            "\n"
            rf"$C_{{\max}} = "
            rf"{concentration_max:.2e}\ "
            rf"\mathrm{{Bq\,m^{{-3}}}}$",
            labelpad=12
        )


        fig.suptitle(
            plot_name,
            fontsize=16,
            fontweight="bold",
            x=0.5,
            y=0.985
        )

        subtitle = fig.text(
            0.5,
            0.935,
            f"t = {times[0]:.2f} s",
            ha="center",
            va="center",
            fontsize=12
        )

        def update(frame):

            t = times[frame]

            field = (
                self._saved_fields[t]
                / concentration_max
            )

            for im, z in zip(ims, z_values):

                im.set_array(
                    field[:, :, z].T
                )

            subtitle.set_text(
                f"t = {t:.2f} s"
            )

            return ims + [subtitle]

        animation = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=100,
            blit=False
        )

        plt.show()

        return animation