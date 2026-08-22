from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

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

from DifAdDec.utils import (
    read_hrtm_data,
    assign_gender_children,
    determine_breathing_rate,
    determine_inhalation_dose_coefficients,
    determine_dose
)

class HRTM:
    def __init__(self, any_diffusion_advection_decay, population_type = "public", age_group = "adult", gender = "male", physical_activity = "sitting", absorption = "F", exposition_time = None):
        self.__concentration_saved_fields = any_diffusion_advection_decay._saved_fields
        self.__species_name = any_diffusion_advection_decay._species_name
        self.__total_time = any_diffusion_advection_decay._total_time
        self.__N = any_diffusion_advection_decay._N
        self.__d = any_diffusion_advection_decay._d
        self.__population_type = population_type
        self.__age_group = age_group
        self.__gender = gender
        self.__physical_activity = physical_activity
        self.__absorption = absorption
        if exposition_time == None:
            self.__exposition_time = any_diffusion_advection_decay._total_time
        else: self.__exposition_time = exposition_time

        self.__data = read_hrtm_data()

    def effective_dose_commitment(self):
        self.__gender = assign_gender_children(self.__gender, self.__age_group)

        breathing_rate = determine_breathing_rate(self.__data, self.__age_group, self.__gender, self.__physical_activity)

        inhalation_dose_coefficients = determine_inhalation_dose_coefficients(self.__data, self.__population_type, self.__species_name, self.__absorption, self.__age_group)

        self.__saved_fields = determine_dose(self.__concentration_saved_fields, breathing_rate, self.__exposition_time, inhalation_dose_coefficients)

        return self.__saved_fields

    def make_csv_for_instant(self, time = None, filename="dose.csv"):

        if time == None:
            time = self.__total_time

        if time not in self.__saved_fields:
            raise ValueError(f"No dose field stored at t = {time}")

        dose = self.__saved_fields[time]

        dx, dy, dz = self.__d[:3]

        x, y, z = np.indices(dose.shape)

        data = np.column_stack((
            x.ravel() * dx,
            y.ravel() * dy,
            z.ravel() * dz,
            dose.ravel()
        ))

        np.savetxt(
            filename,
            data,
            delimiter=",",
            header="x (m),y (m),z (m),dose (Bq)",
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
            self.__saved_fields
        )

        levels = check_or_establish_Z_levels(
            self.__N,
            vertical_axis,
            levels
        )

        X, Y, aux_axis = define_X_Y_values(
            vertical_axis,
            self.__N,
            self.__d
        )

        dosage_max = establish_maximum_concentration(
            self.__saved_fields
        )

        fig, norm = define_initial_plotting_parameters()

        gs = fig.add_gridspec(
            2, 4,
            width_ratios=[0.9, 0.9, 0.9, 0.08],
            wspace=0.35,
            hspace=0.35)

        for i, level in enumerate(levels):

            Z = define_Z_values(
                self.__saved_fields,
                vertical_axis,
                dosage_max,
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
                    dosage_max,
                    vertical_axis_label=(
                        r"Normalized dosage "
                        r"$D/D_{\max}$"
                    ),
                    d=self.__d,
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
                    d=self.__d,
                    gs = gs,
                    iteration=i
                )

            else:

                raise ValueError(
                    "The provided string for visualization type is not valid."
                )

        define_color_bar(
            fig,
            norm,
            colorbar_label = (
                rf"Normalized dose $D/D_{{\max}}$"
                "\n"
                rf"$D_{{\max}} = {dosage_max:.2e}\ "
                rf"\mathrm{{Bq}}$"
            ),
            gs = gs
        )

        plot_title(
            fig,
            plot_name
        )

        show_plot()

    def animate(self, plot_name="Default Name", z_values=None):

        times = sorted(self.__saved_fields.keys())

        if len(times) == 0:
            raise RuntimeError(
                "No saved dosage fields available."
            )

        if z_values is None:
            z_values = np.linspace(
                0,
                self.__N[2] - 1,
                6,
                dtype=int
            )

        if len(z_values) != 6:
            raise ValueError(
                "animate() requires exactly 6 z-values."
            )

        dosage_max = max(
            np.max(field)
            for field in self.__saved_fields.values()
        )

        if dosage_max == 0:
            dosage_max = 1.0

        x = np.arange(self.__N[0]) * self.__d[0]
        y = np.arange(self.__N[1]) * self.__d[1]

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
            self.__saved_fields[times[0]]
            / dosage_max
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
                f"z = {z * self.__d[2]:.2f} m",
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
            rf"Normalized Dosage $D/D_{{\max}}$"
            "\n"
            rf"$D_{{\max}} = "
            rf"{dosage_max:.2e}\ "
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
                self.__saved_fields[t]
                / dosage_max
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

