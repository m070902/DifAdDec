import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from DifAdDec.utils import (
        diffusion_comprobation,
        CFL_comprobation,
)

from .diffusion_advection_decay import DiffusionAdvectionDecay

class IndoorsDiffusionAdvectionDecay(DiffusionAdvectionDecay):
    def __init__(
        self,
        grid_shape=(50, 50, 50),
        d=(0.5, 0.5, 0.5, 0.1),
        total_time=1000.0,
        diffusion_coefficient=(1e-3, 1e-3, 1e-3),   # m²/s
        species_name = "U-234",
        source_positions=[(25, 25, 25)],
        source_effective_iterations = None,
        emission_rate=3.0,
        wall_deposition=1e-4,           # m/s
        inlet_regions=None,
        outlet_regions=None,
        inlet_wind_velocity=0.0,
        outlet_wind_velocity=0.0,
        inlet_concentration=0.0
    ):
        super().__init__(grid_shape, d, total_time, diffusion_coefficient, species_name, source_positions, source_effective_iterations, emission_rate)

        self.__wall_deposition = wall_deposition

        self.__inlet_regions = inlet_regions or []
        self.__outlet_regions = outlet_regions or []

        self.__inlet_wind_velocity = inlet_wind_velocity
        self.__outlet_wind_velocity = outlet_wind_velocity
        self.__inlet_concentration = inlet_concentration

        self.__ventilation_masks = (
        self._build_ventilation_masks()
        )

        self.__wind_velocity = self._build_wind_velocity_field_for_closed_room()

        if (diffusion_comprobation(self._diffusion_coefficient, self._d) == False) or (CFL_comprobation(self.__wind_velocity, self._d) == False):
            raise ValueError("The provided values for the function 'diffusion_advection_decay' do not follow the stability conditions of the equation.")


    def _build_ventilation_masks(self):

        masks = {

            "xmin_in": np.zeros(self._N,dtype=bool),
            "xmin_out": np.zeros(self._N,dtype=bool),

            "xmax_in": np.zeros(self._N,dtype=bool),
            "xmax_out": np.zeros(self._N,dtype=bool),

            "ymin_in": np.zeros(self._N,dtype=bool),
            "ymin_out": np.zeros(self._N,dtype=bool),

            "ymax_in": np.zeros(self._N,dtype=bool),
            "ymax_out": np.zeros(self._N,dtype=bool),

            "zmin_in": np.zeros(self._N,dtype=bool),
            "zmin_out": np.zeros(self._N,dtype=bool),

            "zmax_in": np.zeros(self._N,dtype=bool),
            "zmax_out": np.zeros(self._N,dtype=bool)
        }

        for reg in self.__inlet_regions:

            wall = reg["wall"]

            if wall == "xmin":

                y0,y1 = reg["y"]
                z0,z1 = reg["z"]

                masks["xmin_in"][
                    0,
                    y0:y1,
                    z0:z1
                ] = True

            if wall == "xmax":

                y0,y1 = reg["y"]
                z0,z1 = reg["z"]

                masks["xmax_in"][
                    -1,
                    y0:y1,
                    z0:z1
                ] = True

            if wall == "ymin":

                x0,x1 = reg["x"]
                z0,z1 = reg["z"]

                masks["ymin_in"][
                    x0:x1,
                    0,
                    z0:z1
                ] = True

            if wall == "ymax":

                x0,x1 = reg["x"]
                z0,z1 = reg["z"]

                masks["ymax_in"][
                    x0:x1,
                    -1,
                    z0:z1
                ] = True

            if wall == "zmin":

                x0,x1 = reg["x"]
                y0,y1 = reg["y"]

                masks["zmin_in"][
                    x0:x1,
                    y0:y1,
                    0
                ] = True

            if wall == "zmax":

                x0,x1 = reg["x"]
                y0,y1 = reg["y"]

                masks["zmax_in"][
                    x0:x1,
                    y0:y1,
                    -1
                ] = True

        for reg in self.__outlet_regions:

            wall = reg["wall"]

            if wall == "xmin":

                y0,y1 = reg["y"]
                z0,z1 = reg["z"]

                masks["xmin_out"][
                    0,
                    y0:y1,
                    z0:z1
                ] = True

            if wall == "xmax":

                y0,y1 = reg["y"]
                z0,z1 = reg["z"]

                masks["xmax_out"][
                    -1,
                    y0:y1,
                    z0:z1
                ] = True

            if wall == "ymin":

                x0,x1 = reg["x"]
                z0,z1 = reg["z"]

                masks["ymin_out"][
                    x0:x1,
                    0,
                    z0:z1
                ] = True

            if wall == "ymax":

                x0,x1 = reg["x"]
                z0,z1 = reg["z"]

                masks["ymax_out"][
                    x0:x1,
                    -1,
                    z0:z1
                ] = True

            if wall == "zmin":

                x0,x1 = reg["x"]
                y0,y1 = reg["y"]

                masks["zmin_out"][
                    x0:x1,
                    y0:y1,
                    0
                ] = True

            if wall == "zmax":

                x0,x1 = reg["x"]
                y0,y1 = reg["y"]

                masks["zmax_out"][
                    x0:x1,
                    y0:y1,
                    -1
                ] = True

        return masks

    def _build_wind_velocity_field_for_closed_room(self):
        Nx, Ny, Nz = self._N
        dx, dy, dz = self._d[0], self._d[1], self._d[2]
        total_nodes = Nx * Ny * Nz

        # Matriz del sistema en formato LIL para construcción eficiente
        A = lil_matrix((total_nodes, total_nodes), dtype=float)
        b = np.zeros(total_nodes)

        # Precomputación de coeficientes espaciales
        idx2 = 1.0 / (dx**2)
        idy2 = 1.0 / (dy**2)
        idz2 = 1.0 / (dz**2)

        # Mapeo de índices 3D a 1D
        def get_index(i, j, k):
            return i + j * Nx + k * Nx * Ny

        # Recuperar máscaras completas combinadas para simplificar la lógica de frontera
        m = self.__ventilation_masks

        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    idx = get_index(i, j, k)

                    # Determinar si el nodo está en alguna frontera externa
                    is_xmin = (i == 0)
                    is_xmax = (i == Nx - 1)
                    is_ymin = (j == 0)
                    is_ymax = (j == Ny - 1)
                    is_zmin = (k == 0)
                    is_zmax = (k == Nz - 1)

                    if is_xmin or is_xmax or is_ymin or is_ymax or is_zmin or is_zmax:
                        # Ecuación de frontera (Neumann: dphi/dn = v_normal)
                        # Usamos aproximaciones de diferencias finitas de primer orden para la frontera
                        if is_xmin:
                            A[idx, idx] = -1.0 / dx
                            A[idx, get_index(i + 1, j, k)] = 1.0 / dx
                            if m["xmin_in"][i, j, k]:
                                b[idx] = self.__inlet_wind_velocity
                            elif m["xmin_out"][i, j, k]:
                                b[idx] = -self.__outlet_wind_velocity
                            else:
                                b[idx] = 0.0

                        elif is_xmax:
                            A[idx, idx] = 1.0 / dx
                            A[idx, get_index(i - 1, j, k)] = -1.0 / dx
                            if m["xmax_in"][i, j, k]:
                                b[idx] = -self.__inlet_wind_velocity
                            elif m["xmax_out"][i, j, k]:
                                b[idx] = self.__outlet_wind_velocity
                            else:
                                b[idx] = 0.0

                        elif is_ymin:
                            A[idx, idx] = -1.0 / dy
                            A[idx, get_index(i, j + 1, k)] = 1.0 / dy
                            if m["ymin_in"][i, j, k]:
                                b[idx] = self.__inlet_wind_velocity
                            elif m["ymin_out"][i, j, k]:
                                b[idx] = -self.__outlet_wind_velocity
                            else:
                                b[idx] = 0.0

                        elif is_ymax:
                            A[idx, idx] = 1.0 / dy
                            A[idx, get_index(i, j - 1, k)] = -1.0 / dy
                            if m["ymax_in"][i, j, k]:
                                b[idx] = -self.__inlet_wind_velocity
                            elif m["ymax_out"][i, j, k]:
                                b[idx] = self.__outlet_wind_velocity
                            else:
                                b[idx] = 0.0

                        elif is_zmin:
                            A[idx, idx] = -1.0 / dz
                            A[idx, get_index(i, j, k + 1)] = 1.0 / dz
                            if m["zmin_in"][i, j, k]:
                                b[idx] = self.__inlet_wind_velocity
                            elif m["zmin_out"][i, j, k]:
                                b[idx] = -self.__outlet_wind_velocity
                            else:
                                b[idx] = 0.0

                        elif is_zmax:
                            A[idx, idx] = 1.0 / dz
                            A[idx, get_index(i, j, k - 1)] = -1.0 / dz
                            if m["zmax_in"][i, j, k]:
                                b[idx] = -self.__inlet_wind_velocity
                            elif m["zmax_out"][i, j, k]:
                                b[idx] = self.__outlet_wind_velocity
                            else:
                                b[idx] = 0.0
                    else:
                        # Nodos internos: Ecuación de Laplace (Stencil de 7 puntos)
                        A[idx, idx] = -2.0 * (idx2 + idy2 + idz2)
                        A[idx, get_index(i - 1, j, k)] = idx2
                        A[idx, get_index(i + 1, j, k)] = idx2
                        A[idx, get_index(i, j - 1, k)] = idy2
                        A[idx, get_index(i, j + 1, k)] = idy2
                        A[idx, get_index(i, j, k - 1)] = idz2
                        A[idx, get_index(i, j, k + 1)] = idz2
                        b[idx] = 0.0

        # Al ser un problema puramente de Neumann, el potencial está definido salvo una constante.
        # Fijamos el potencial del primer nodo para asegurar unicidad en la solución del sistema.
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = 0.0

        # Conversión a formato CSR para resolución eficiente
        A = A.tocsr()

        # Resolver el sistema lineal
        phi_flat = spsolve(A, b)
        phi = phi_flat.reshape((Nx, Ny, Nz))

        # Calcular el campo de velocidades: V = grad(phi) utilizando diferencias centrales
        vx = np.zeros((Nx, Ny, Nz))
        vy = np.zeros((Nx, Ny, Nz))
        vz = np.zeros((Nx, Ny, Nz))

        vx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
        vy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
        vz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)

        # Condiciones de contorno para los componentes de velocidad en los extremos externos
        vx[0, :, :] = vx[1, :, :]
        vx[-1, :, :] = vx[-2, :, :]
        vy[:, 0, :] = vy[:, 1, :]
        vy[:, -1, :] = vy[:, -2, :]
        vz[:, :, 0] = vz[:, :, 1]
        vz[:, :, -1] = vz[:, :, -2]

        return [vx, vy, vz]

    def _compute_advection(self, concentration_aux: list[float]):

        adv = np.zeros_like(concentration_aux[1:-1,1:-1,1:-1])

        vx = self.__wind_velocity[0][1:-1,1:-1,1:-1]
        vy = self.__wind_velocity[1][1:-1,1:-1,1:-1]
        vz = self.__wind_velocity[2][1:-1,1:-1,1:-1]

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

    def _apply_boundary_conditions_concentration(self):

        alpha_x_wall = (
            self.__wall_deposition
            * self._d[0]
            / self._diffusion_coefficient[0]
        )

        alpha_y_wall = (
            self.__wall_deposition
            * self._d[1]
            / self._diffusion_coefficient[1]
        )

        alpha_z_wall = (
            self.__wall_deposition
            * self._d[2]
            / self._diffusion_coefficient[2]
        )

        alpha_x_out = (
            self.__outlet_wind_velocity
            * self._d[0]
            / self._diffusion_coefficient[0]
        )

        alpha_y_out = (
            self.__outlet_wind_velocity
            * self._d[1]
            / self._diffusion_coefficient[1]
        )

        alpha_z_out = (
            self.__outlet_wind_velocity
            * self._d[2]
            / self._diffusion_coefficient[2]
        )

        inlet_mask = self.__ventilation_masks["xmin_in"][0,:,:]
        outlet_mask = self.__ventilation_masks["xmin_out"][0,:,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self._concentration[0,:,:][wall_mask] = (
            self._concentration[1,:,:][wall_mask]
            /(1 + alpha_x_wall)
        )

        self._concentration[0,:,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self._concentration[0,:,:][outlet_mask] = (
            self._concentration[1,:,:][outlet_mask]
            /(1 + alpha_x_out)
        )

        inlet_mask = self.__ventilation_masks["xmax_in"][-1,:,:]
        outlet_mask = self.__ventilation_masks["xmax_out"][-1,:,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self._concentration[-1,:,:][wall_mask] = (
            self._concentration[-2,:,:][wall_mask]
            /(1 + alpha_x_wall)
        )

        self._concentration[-1,:,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self._concentration[-1,:,:][outlet_mask] = (
            self._concentration[-2,:,:][outlet_mask]
            /(1 + alpha_x_out)
        )

        inlet_mask = self.__ventilation_masks["ymin_in"][:,0,:]
        outlet_mask = self.__ventilation_masks["ymin_out"][:,0,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self._concentration[:,0,:][wall_mask] = (
            self._concentration[:,1,:][wall_mask]
            /(1 + alpha_y_wall)
        )

        self._concentration[:,0,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self._concentration[:,0,:][outlet_mask] = (
            self._concentration[:,1,:][outlet_mask]
            /(1 + alpha_y_out)
        )

        inlet_mask = self.__ventilation_masks["ymax_in"][:,-1,:]
        outlet_mask = self.__ventilation_masks["ymax_out"][:,-1,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self._concentration[:,-1,:][wall_mask] = (
            self._concentration[:,-2,:][wall_mask]
            /(1 + alpha_y_wall)
        )

        self._concentration[:,-1,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self._concentration[:,-1,:][outlet_mask] = (
            self._concentration[:,-2,:][outlet_mask]
            /(1 + alpha_y_out)
        )

        inlet_mask = self.__ventilation_masks["zmin_in"][:,:,0]
        outlet_mask = self.__ventilation_masks["zmin_out"][:,:,0]

        wall_mask = ~(inlet_mask | outlet_mask)

        self._concentration[:,:,0][wall_mask] = (
            self._concentration[:,:,1][wall_mask]
            /(1 + alpha_z_wall)
        )

        self._concentration[:,:,0][inlet_mask] = (
            self.__inlet_concentration
        )

        self._concentration[:,:,0][outlet_mask] = (
            self._concentration[:,:,1][outlet_mask]
            /(1 + alpha_z_out)
        )

        inlet_mask = self.__ventilation_masks["zmax_in"][:,:,-1]
        outlet_mask = self.__ventilation_masks["zmax_out"][:,:,-1]

        wall_mask = ~(inlet_mask | outlet_mask)

        self._concentration[:,:,-1][wall_mask] = (
            self._concentration[:,:,-2][wall_mask]
            /(1 + alpha_z_wall)
        )

        self._concentration[:,:,-1][inlet_mask] = (
            self.__inlet_concentration
        )

        self._concentration[:,:,-1][outlet_mask] = (
            self._concentration[:,:,-2][outlet_mask]
            /(1 + alpha_z_out)
        )

    def _step_concentration(self):

        concentration_aux = self._concentration.copy()

        diffusion = self._compute_diffusion(concentration_aux)

        advection = self._compute_advection(concentration_aux)

        decay = self._lamda * concentration_aux[1:-1,1:-1,1:-1]

        self._concentration[1:-1,1:-1,1:-1] = (
            concentration_aux[1:-1,1:-1,1:-1]
            + self._d[3] * diffusion
            - self._d[3] * advection
            - self._d[3] * decay
        )

        self._concentration = np.maximum(self._concentration, 0)

        self._apply_boundary_conditions_concentration()


    def run(self, save_every_X_iteration=100):

        total_steps = int(self._total_time / self._d[3])

        self._saved_fields[0.0] = self._concentration.copy()

        for n in range(1, total_steps + 1):

            current_time = n * self._d[3]

            self._step_concentration()

            if n < self._source_effective_iterations:
                self._inject_sources()

            if n % save_every_X_iteration == 0:

                self._saved_fields[current_time] = self._concentration.copy()

                print(
                    f"t = {current_time:.2f} s | "
                    f"max(C) = {np.max(self._concentration):.5e}"
                )

        return self._saved_fields
