import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from radioprotection.utils import (
        diffusion_comprobation,
        CFL_comprobation,
        lambda_for_species
)

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


    def _compute_diffusion(self, concentration_aux: dict[str, list[float]]):

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
