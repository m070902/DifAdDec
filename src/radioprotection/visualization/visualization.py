import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

def check_provided_time(time: float, duration: float, concentration: dict[str, list[float]]) -> float:
    if time is None:
        time = len(concentration) - 1
        return  time
    elif (time < 0 or time > duration):
        valid_times = list(range(len(concentration)))

        raise ValueError(
            "Provided time value is either not registered or not correctly provided.\n"
            f"Please introduce one of the following:\n"
            f"{', '.join(map(str, valid_times))}"
        )
    return time

def check_number_of_Z_to_check(vertical_axis: str, levels: int):
    if len(levels) > 6:
        raise ValueError(f"6 or less values of {vertical_axis} must be provided.")

def define_X_Y_values(vertical_axis: str, n: tuple[float, float, float]) -> tuple[list[float], list[float], list[str, str]]:
    if vertical_axis == "x":
        y = np.arange(n[1])
        z = np.arange(n[2])
        X, Y = np.meshgrid(y, z)
        aux_axis = ["y","z"]
    elif vertical_axis == "y":
        x = np.arange(n[0])
        z = np.arange(n[2])
        X, Y = np.meshgrid(x, z)
        aux_axis = ["x","z"]
    elif vertical_axis == "z":
        x = np.arange(n[0])
        y = np.arange(n[1])
        X, Y = np.meshgrid(x, y)
        aux_axis = ["x","y"]
    else:
        raise ValueError("The provided string for vertical axis is not valid.")
    return X, Y, aux_axis

def stablish_maximum_concentration(time: float, concentration: dict[str, list[float]]):
    return np.max(concentration[f"{time}"])

def define_initial_plotting_parameters():
    norm = mcolors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(20, 10))
    return fig, norm

def define_Z_values(concentration: dict[str, list[float]], vertical_axis: str, concentration_max:  float, time: float, level: int):
    if vertical_axis == "x":
        Z = concentration[f'{time}'][level-1, :, :].T / concentration_max
    elif vertical_axis == "y":
        Z = concentration[f'{time}'][:, level-1, :].T / concentration_max
    elif vertical_axis == "z":
        Z = concentration[f'{time}'][:, :, level-1].T / concentration_max
    return Z

def plot_3d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, concentration_max, vertical_axis_label, iteration = 0):
    ax = fig.add_subplot(2, 3, iteration + 1, projection='3d')
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap='viridis',
        vmin=0,
        vmax=1,
        edgecolor='none',
        norm = norm
    )

    ax.set_zlim(0, 1)
    ax.set_title(rf'{vertical_axis} = {level}')
    ax.set_xlabel(aux_axis[0])
    ax.set_ylabel(aux_axis[1])
    ax.set_zlabel(vertical_axis_label, fontsize=8, labelpad=10)
    ax.zaxis.get_offset_text().set_fontsize(7)


def plot_2d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, iteration = 0):
    ax = fig.add_subplot(2, 3, iteration + 1)
    surf = ax.contourf(
        X,
        Y,
        Z,
        vmin=0,
        vmax=1,
        cmap='viridis',
        norm = norm
    )

    ax.set_title(rf'{vertical_axis} = {level}')
    ax.set_xlabel(aux_axis[0])
    ax.set_ylabel(aux_axis[1])

def define_color_bar(fig, norm, concentration_max, vertical_axis_label):
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])

    fig.colorbar(
        sm,
        ax=fig.axes,
        shrink=0.5,
        aspect=20,
        label=vertical_axis_label,
        location = "bottom"
    )

def plot_title(fig, title):
    fig.suptitle(title)

def show_plot():
        plt.show()