import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

def check_provided_time(time_to_check: float | None, saved_fields: dict) -> float:

    valid_times = list(saved_fields.keys())

    if not valid_times:
        raise ValueError("No concentration fields have been saved.")

    if time_to_check is None:
        return valid_times[-1]

    if time_to_check not in saved_fields:
        raise ValueError(
            "The provided time is not registered.\n"
            f"Available times:\n"
            f"{', '.join(map(str, valid_times))}"
        )

    return time_to_check

def check_or_establish_Z_levels(
    N: tuple[int, int, int],
    vertical_axis: str,
    levels=None
):
    if levels is None:

        if vertical_axis == "x":
            levels = np.linspace(
                0,
                N[0] - 1,
                6,
                dtype=int
            )

        elif vertical_axis == "y":
            levels = np.linspace(
                0,
                N[1] - 1,
                6,
                dtype=int
            )

        elif vertical_axis == "z":
            levels = np.linspace(
                0,
                N[2] - 1,
                6,
                dtype=int
            )

        else:
            raise ValueError(
                "The provided string for vertical axis is not valid."
            )

    levels = np.asarray(levels, dtype=int)

    if len(levels) > 6:
        raise ValueError(
            f"6 or less values of {vertical_axis} must be provided."
        )

    return levels

def define_X_Y_values(
    vertical_axis: str,
    N: tuple[int, int, int],
    d: tuple[float, float, float, float]
):
    dx, dy, dz, _ = d

    if vertical_axis == "x":

        y = np.arange(N[1]) * dy
        z = np.arange(N[2]) * dz

        X, Y = np.meshgrid(y, z)

        aux_axis = ["y", "z"]

    elif vertical_axis == "y":

        x = np.arange(N[0]) * dx
        z = np.arange(N[2]) * dz

        X, Y = np.meshgrid(x, z)

        aux_axis = ["x", "z"]

    elif vertical_axis == "z":

        x = np.arange(N[0]) * dx
        y = np.arange(N[1]) * dy

        X, Y = np.meshgrid(x, y)

        aux_axis = ["x", "y"]

    else:
        raise ValueError(
            "The provided string for vertical axis is not valid."
        )

    return X, Y, aux_axis

def establish_maximum_concentration(
    saved_fields: dict
) -> float:

    concentration_max = max(
        np.max(field)
        for field in saved_fields.values()
    )

    if concentration_max <= 0:
        raise ValueError(
            "The maximum concentration must be greater than zero."
        )

    return concentration_max

def define_initial_plotting_parameters():
    norm = mcolors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(20, 10))
    return fig, norm

def define_Z_values(saved_values: dict, vertical_axis: str, concentration_max: float, time_to_check: float, level: int):

    field = saved_values[time_to_check]

    if vertical_axis == "x":
        Z = field[level, :, :].T

    elif vertical_axis == "y":
        Z = field[:, level, :].T

    elif vertical_axis == "z":
        Z = field[:, :, level].T

    else:
        raise ValueError(
            "The provided string for vertical axis is not valid."
        )

    return Z / concentration_max

def define_physical_level(
    level: int,
    vertical_axis: str,
    d: tuple[float, float, float, float]
):
    dx, dy, dz, _ = d

    if vertical_axis == "x":
        return level * dx

    elif vertical_axis == "y":
        return level * dy

    elif vertical_axis == "z":
        return level * dz

    else:
        raise ValueError(
            "The provided string for vertical axis is not valid."
        )

def plot_3d(
    X,
    Y,
    Z,
    fig,
    norm,
    vertical_axis,
    level,
    aux_axis,
    concentration_max,
    vertical_axis_label,
    d,
    iteration=0
):

    physical_level = define_physical_level(
        level,
        vertical_axis,
        d
    )

    ax = fig.add_subplot(
        2,
        3,
        iteration + 1,
        projection="3d"
    )

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="none",
        norm=norm
    )

    ax.set_zlim(0, 1)

    ax.set_title(
        rf"{vertical_axis} = {physical_level:.2f} m"
    )

    ax.set_xlabel(aux_axis[0] + " (m)")
    ax.set_ylabel(aux_axis[1] + " (m)")

    ax.set_zlabel(
        vertical_axis_label,
        fontsize=8,
        labelpad=10
    )

    ax.zaxis.get_offset_text().set_fontsize(7)


def plot_2d(
    X,
    Y,
    Z,
    fig,
    norm,
    vertical_axis,
    level,
    aux_axis,
    d,
    iteration=0
):

    physical_level = define_physical_level(
        level,
        vertical_axis,
        d
    )

    ax = fig.add_subplot(
        2,
        3,
        iteration + 1
    )

    surf = ax.contourf(
        X,
        Y,
        Z,
        vmin=0,
        vmax=1,
        cmap="viridis",
        norm=norm
    )

    ax.set_title(
        rf"{vertical_axis} = {physical_level:.2f} m"
    )

    ax.set_xlabel(
        aux_axis[0] + " (m)"
    )

    ax.set_ylabel(
        aux_axis[1] + " (m)"
    )

def define_color_bar(
    fig,
    norm,
    colorbar_label
):
    sm = cm.ScalarMappable(
        norm=norm,
        cmap="viridis"
    )
    sm.set_array([])


    fig.colorbar(
        sm,
        ax=fig.axes,
        shrink=0.5,
        aspect=20,
        label=colorbar_label,
        location="bottom"
    )

def plot_title(fig, title):
    fig.suptitle(title)

def show_plot():
        plt.show()