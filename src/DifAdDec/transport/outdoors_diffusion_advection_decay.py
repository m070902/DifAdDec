import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .diffusion_advection_decay import DiffusionAdvectionDecay

from DifAdDec.utils import (
    diffusion_comprobation,
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
        source_effective_iterations = None,
        emission_rate=3.0
    ):
        super().__init__(grid_shape, d, total_time, diffusion_coefficient, species_name, source_positions, source_effective_iterations,emission_rate)

        self.__wind_model = wind_model

        if (diffusion_comprobation(self._diffusion_coefficient, self._d) == False):
            raise ValueError("The provided values for the function 'diffusion_advection_decay' do not follow the stability conditions of the equation.")

    def _compute_advection(self, concentration_aux: list[float], time: int):

        adv = np.zeros_like(concentration_aux[1:-1,1:-1,1:-1])

        vx, vy, vz = self.__wind_model.get_velocity(self._d, time)

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

    def run(self, save_every_X_iteration=100):

        total_steps = int(self._total_time / self._d[3])

        self._saved_fields[0.0] = self._concentration.copy()

        for n in range(1, total_steps + 1):

            current_time = n * self._d[3]

            self._step_concentration(current_time)

            if n < self._source_effective_iterations:
                self._inject_sources()

            if n % save_every_X_iteration == 0:

                self._saved_fields[current_time] = self._concentration.copy()

                print(
                    f"t = {current_time:.2f} s | "
                    f"max(C) = {np.max(self._concentration):.5e}"
                )

        return self._saved_fields