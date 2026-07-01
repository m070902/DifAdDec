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

class IndoorsDiffusionAdvectionDecay:

    def __init__(
        self,
        grid_shape=(50, 50, 30),
        d=(0.5, 0.5, 0.5, 0.1),
        total_time=100.0,
        diffusion_coefficient=(1e-3, 1e-3, 1e-3),   # m²/s
        initial_velocity=(0.05, 0.0, 0.0),      # m/s
        species_name = "U-234",
        source_positions=[(25, 25, 15)],
        emission_rate=1.0,
        wall_deposition=1e-4,           # m/s
        inlet_regions=None,
        outlet_regions=None,
        inlet_velocity=0.0,
        outlet_velocity=0.0,
        inlet_concentration=0.0
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
        self.__wall_deposition = wall_deposition
        self.__concentration = np.zeros(self.__N)
        self.__saved_fields = {}

        self.__inlet_regions = inlet_regions or []
        self.__outlet_regions = outlet_regions or []

        self.__inlet_velocity = inlet_velocity
        self.__outlet_velocity = outlet_velocity
        self.__inlet_concentration = inlet_concentration

        self.__ventilation_masks = (
        self._build_ventilation_masks()
        )

        self.__velocity = self._build_velocity_field_for_closed_room()

        if (diffusion_comprobation(self.__diffusion_coefficient, d) == False) or (CFL_comprobation(self.__velocity, d) == False):
            raise ValueError("The provided values for the function 'diffusion_advection_decay' do not follow the stability conditions of the equation.")


    def _build_ventilation_masks(self):

        masks = {

            "xmin_in": np.zeros(self.__N,dtype=bool),
            "xmin_out": np.zeros(self.__N,dtype=bool),

            "xmax_in": np.zeros(self.__N,dtype=bool),
            "xmax_out": np.zeros(self.__N,dtype=bool),

            "ymin_in": np.zeros(self.__N,dtype=bool),
            "ymin_out": np.zeros(self.__N,dtype=bool),

            "ymax_in": np.zeros(self.__N,dtype=bool),
            "ymax_out": np.zeros(self.__N,dtype=bool),

            "zmin_in": np.zeros(self.__N,dtype=bool),
            "zmin_out": np.zeros(self.__N,dtype=bool),

            "zmax_in": np.zeros(self.__N,dtype=bool),
            "zmax_out": np.zeros(self.__N,dtype=bool)
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

    def _build_velocity_field_for_closed_room(self):
        vx = np.full((self.__N), self.__initial_velocity[0])
        vy = np.full((self.__N), self.__initial_velocity[1])
        vz = np.full((self.__N), self.__initial_velocity[2])

        # Distancia normalizada a paredes
        x = np.linspace(0, 1, self.__N[0])
        y = np.linspace(0, 1, self.__N[1])
        z = np.linspace(0, 1, self.__N[2])

        fx = np.sin(np.pi * x)
        fy = np.sin(np.pi * y)
        fz = np.sin(np.pi * z)

        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')

        # Perfil suave tipo no-slip
        vx *= FX
        vy *= FY
        vz *= FZ

        for inlet in self.__inlet_regions:

            if inlet["wall"] == "xmin":

                y0,y1 = inlet["y"]
                z0,z1 = inlet["z"]

                vx[
                    0:30,
                    y0:y1,
                    z0:z1
                ] += self.__inlet_velocity

            if inlet["wall"] == "xmax":

                y0,y1 = inlet["y"]
                z0,z1 = inlet["z"]

                vx[
                    0:30,
                    y0:y1,
                    z0:z1
                ] -= self.__inlet_velocity

            if inlet["wall"] == "ymin":

                x0,x1 = inlet["x"]
                z0,z1 = inlet["z"]

                vy[
                    x0:x1,
                    0:30,
                    z0:z1
                ] += self.__inlet_velocity

            if inlet["wall"] == "ymax":

                x0,x1 = inlet["x"]
                z0,z1 = inlet["z"]

                vy[
                    x0:x1,
                    0:30,
                    z0:z1
                ] -= self.__inlet_velocity

            if inlet["wall"] == "zmin":

                x0,x1 = inlet["x"]
                y0,y1 = inlet["y"]

                vz[
                    x0:x1,
                    y0:y1,
                    0:20
                ] += self.__inlet_velocity

            if inlet["wall"] == "zmax":

                x0,x1 = inlet["x"]
                y0,y1 = inlet["y"]

                vz[
                    x0:x1,
                    y0:y1,
                    0:20
                ] -= self.__inlet_velocity

        for outlet in self.__outlet_regions:

            if outlet["wall"] == "xmin":

                y0,y1 = outlet["y"]
                z0,z1 = outlet["z"]

                vx[
                    -30:,
                    y0:y1,
                    z0:z1
                ] += self.__outlet_velocity

            if outlet["wall"] == "xmax":

                y0,y1 = outlet["y"]
                z0,z1 = outlet["z"]

                vx[
                    -30:,
                    y0:y1,
                    z0:z1
                ] -= self.__outlet_velocity

            if outlet["wall"] == "ymin":

                x0,x1 = outlet["x"]
                z0,z1 = outlet["z"]

                vy[
                    x0:x1,
                    -30:,
                    z0:z1
                ] += self.__outlet_velocity

            if outlet["wall"] == "ymax":

                x0,x1 = outlet["x"]
                z0,z1 = outlet["z"]

                vy[
                    x0:x1,
                    -30:,
                    z0:z1
                ] -= self.__outlet_velocity

            if outlet["wall"] == "zmin":

                x0,x1 = outlet["x"]
                y0,y1 = outlet["y"]

                vz[
                    x0:x1,
                    y0:y1,
                    -20:
                ] += self.__outlet_velocity

            if outlet["wall"] == "zmax":

                x0,x1 = outlet["x"]
                y0,y1 = outlet["y"]

                vz[
                    x0:x1,
                    y0:y1,
                    -20:
                ] -= self.__outlet_velocity

        return [vx, vy, vz]

    def _compute_diffusion(self, concentration_aux: dict[str, list[float]]):

        return (
            self.__diffusion_coefficient[0] * (
                concentration_aux[2:,1:-1,1:-1]
                - 2*concentration_aux[1:-1,1:-1,1:-1]
                + concentration_aux[:-2,1:-1,1:-1]
            ) / self.__d[0]**2

            +

            self.__diffusion_coefficient[1] * (
                concentration_aux[1:-1,2:,1:-1]
                - 2*concentration_aux[1:-1,1:-1,1:-1]
                + concentration_aux[1:-1,:-2,1:-1]
            ) / self.__d[1]**2

            +

            self.__diffusion_coefficient[2] * (
                concentration_aux[1:-1,1:-1,2:]
                - 2*concentration_aux[1:-1,1:-1,1:-1]
                + concentration_aux[1:-1,1:-1,:-2]
            ) / self.__d[2]**2
        )

    # ==========================================================
    # Advección UPWIND
    # ==========================================================
    def _compute_advection(self, concentration_aux: dict[str, list[float]]):

        adv = np.zeros_like(concentration_aux[1:-1,1:-1,1:-1])

        vx = self.__velocity[0][1:-1,1:-1,1:-1]
        vy = self.__velocity[1][1:-1,1:-1,1:-1]
        vz = self.__velocity[2][1:-1,1:-1,1:-1]

        # ======================================
        # X
        # ======================================

        dCdx_backward = (
            concentration_aux[1:-1,1:-1,1:-1]
            - concentration_aux[:-2,1:-1,1:-1]
        ) / self.__d[0]

        dCdx_forward = (
            concentration_aux[2:,1:-1,1:-1]
            - concentration_aux[1:-1,1:-1,1:-1]
        ) / self.__d[0]

        adv += np.where(
            vx >= 0,
            vx * dCdx_backward,
            vx * dCdx_forward
        )

        # ======================================
        # Y
        # ======================================

        dCdy_backward = (
            concentration_aux[1:-1,1:-1,1:-1]
            - concentration_aux[1:-1,:-2,1:-1]
        ) / self.__d[1]

        dCdy_forward = (
            concentration_aux[1:-1,2:,1:-1]
            - concentration_aux[1:-1,1:-1,1:-1]
        ) / self.__d[1]
        adv += np.where(
            vy >= 0,
            vy * dCdy_backward,
            vy * dCdy_forward
        )

        # ======================================
        # Z
        # ======================================

        dCdz_backward = (
            concentration_aux[1:-1,1:-1,1:-1]
            - concentration_aux[1:-1,1:-1,:-2]
        ) / self.__d[2]

        dCdz_forward = (
            concentration_aux[1:-1,1:-1,2:]
            - concentration_aux[1:-1,1:-1,1:-1]
        ) / self.__d[2]

        adv += np.where(
            vz >= 0,
            vz * dCdz_backward,
            vz * dCdz_forward
        )

        return adv

    # ==========================================================
    # Condiciones de contorno
    # ==========================================================
    def _apply_boundary_conditions_concentration(self):

        # =====================================================
        # Parámetros Robin
        # =====================================================

        alpha_x_wall = (
            self.__wall_deposition
            * self.__d[0]
            / self.__diffusion_coefficient[0]
        )

        alpha_y_wall = (
            self.__wall_deposition
            * self.__d[1]
            / self.__diffusion_coefficient[1]
        )

        alpha_z_wall = (
            self.__wall_deposition
            * self.__d[2]
            / self.__diffusion_coefficient[2]
        )

        alpha_x_out = (
            self.__outlet_velocity
            * self.__d[0]
            / self.__diffusion_coefficient[0]
        )

        alpha_y_out = (
            self.__outlet_velocity
            * self.__d[1]
            / self.__diffusion_coefficient[1]
        )

        alpha_z_out = (
            self.__outlet_velocity
            * self.__d[2]
            / self.__diffusion_coefficient[2]
        )

        # =====================================================
        # XMIN
        # =====================================================

        inlet_mask = self.__ventilation_masks["xmin_in"][0,:,:]
        outlet_mask = self.__ventilation_masks["xmin_out"][0,:,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        # pared sólida
        self.__concentration[0,:,:][wall_mask] = (
            self.__concentration[1,:,:][wall_mask]
            /(1 + alpha_x_wall)
        )

        # impulsión
        self.__concentration[0,:,:][inlet_mask] = (
            self.__inlet_concentration
        )

        # extracción
        self.__concentration[0,:,:][outlet_mask] = (
            self.__concentration[1,:,:][outlet_mask]
            /(1 + alpha_x_out)
        )

        # =====================================================
        # XMAX
        # =====================================================

        inlet_mask = self.__ventilation_masks["xmax_in"][-1,:,:]
        outlet_mask = self.__ventilation_masks["xmax_out"][-1,:,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self.__concentration[-1,:,:][wall_mask] = (
            self.__concentration[-2,:,:][wall_mask]
            /(1 + alpha_x_wall)
        )

        self.__concentration[-1,:,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self.__concentration[-1,:,:][outlet_mask] = (
            self.__concentration[-2,:,:][outlet_mask]
            /(1 + alpha_x_out)
        )

        # =====================================================
        # YMIN
        # =====================================================

        inlet_mask = self.__ventilation_masks["ymin_in"][:,0,:]
        outlet_mask = self.__ventilation_masks["ymin_out"][:,0,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self.__concentration[:,0,:][wall_mask] = (
            self.__concentration[:,1,:][wall_mask]
            /(1 + alpha_y_wall)
        )

        self.__concentration[:,0,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self.__concentration[:,0,:][outlet_mask] = (
            self.__concentration[:,1,:][outlet_mask]
            /(1 + alpha_y_out)
        )

        # =====================================================
        # YMAX
        # =====================================================

        inlet_mask = self.__ventilation_masks["ymax_in"][:,-1,:]
        outlet_mask = self.__ventilation_masks["ymax_out"][:,-1,:]

        wall_mask = ~(inlet_mask | outlet_mask)

        self.__concentration[:,-1,:][wall_mask] = (
            self.__concentration[:,-2,:][wall_mask]
            /(1 + alpha_y_wall)
        )

        self.__concentration[:,-1,:][inlet_mask] = (
            self.__inlet_concentration
        )

        self.__concentration[:,-1,:][outlet_mask] = (
            self.__concentration[:,-2,:][outlet_mask]
            /(1 + alpha_y_out)
        )

        # =====================================================
        # ZMIN
        # =====================================================

        inlet_mask = self.__ventilation_masks["zmin_in"][:,:,0]
        outlet_mask = self.__ventilation_masks["zmin_out"][:,:,0]

        wall_mask = ~(inlet_mask | outlet_mask)

        self.__concentration[:,:,0][wall_mask] = (
            self.__concentration[:,:,1][wall_mask]
            /(1 + alpha_z_wall)
        )

        self.__concentration[:,:,0][inlet_mask] = (
            self.__inlet_concentration
        )

        self.__concentration[:,:,0][outlet_mask] = (
            self.__concentration[:,:,1][outlet_mask]
            /(1 + alpha_z_out)
        )

        # =====================================================
        # ZMAX
        # =====================================================

        inlet_mask = self.__ventilation_masks["zmax_in"][:,:,-1]
        outlet_mask = self.__ventilation_masks["zmax_out"][:,:,-1]

        wall_mask = ~(inlet_mask | outlet_mask)

        self.__concentration[:,:,-1][wall_mask] = (
            self.__concentration[:,:,-2][wall_mask]
            /(1 + alpha_z_wall)
        )

        self.__concentration[:,:,-1][inlet_mask] = (
            self.__inlet_concentration
        )

        self.__concentration[:,:,-1][outlet_mask] = (
            self.__concentration[:,:,-2][outlet_mask]
            /(1 + alpha_z_out)
        )

    # ==========================================================
    # Fuentes
    # ==========================================================
    def _inject_sources(self):

        for idx in self.__source_positions:
            self.__concentration[idx] += self.__emission_rate * self.__d[3]

    # ==========================================================
    # STEP
    # ==========================================================
    def _step_concentration(self):

        concentration_aux = self.__concentration.copy()

        diffusion = self._compute_diffusion(concentration_aux)

        advection = self._compute_advection(concentration_aux)

        decay = self.__lamda * concentration_aux[1:-1,1:-1,1:-1]

        self.__concentration[1:-1,1:-1,1:-1] = (
            concentration_aux[1:-1,1:-1,1:-1]
            + self.__d[3] * diffusion
            - self.__d[3] * advection
            - self.__d[3] * decay
        )

        # evitar negativos numéricos
        self.__concentration = np.maximum(self.__concentration, 0)

        # BC
        self._apply_boundary_conditions_concentration()

        # fuente
        self._inject_sources()

    # ==========================================================
    # SIMULACIÓN
    # ==========================================================
    def run(self, save_every=100):

        total_steps = int(self.__total_time / self.__d[3])

        self.__saved_fields[0.0] = self.__concentration.copy()

        for n in range(1, total_steps + 1):

            #self._step_velocity()
            self._step_concentration()

            if n % save_every == 0:

                current_time = n * self.__d[3]

                self.__saved_fields[current_time] = self.__concentration.copy()

                print(
                    f"t = {current_time:.2f} s | "
                    f"max(C) = {np.max(self.__concentration):.5e}"
                )

        return self.__saved_fields

    def animate(self, z_values=None):
        times = sorted(self.__saved_fields.keys())

        # Seleccionar 6 valores de z si no se especifican
        if z_values is None:
            z_values = np.linspace(
                0,
                self.__N[2] - 1,
                6,
                dtype=int
            )

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()

        first = self.__saved_fields[times[0]]

        ims = []
        for ax, z in zip(axes, z_values):
            im = ax.imshow(
                first[:, :, z].T,
                origin='lower',
                extent=[0, self.__N[0], 0, self.__N[1]],
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
                    self.__saved_fields[t][:, :, z].T
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