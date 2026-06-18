import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

from radioprotection.utils import (
        diffusion_comprobation,
        CFL_comprobation,
        lambda_for_species
)

VISCOSITY_OF_AIR = 1.5e10-5 # micro pascales x segundo

class IndoorsDiffusionAdvectionDecay:

    def __init__(
        self,
        grid_shape=(50, 50, 30),
        d=(0.5, 0.5, 0.5, 0.01),
        total_time=100.0,
        diffusion_coefficient=(1e-3, 1e-3, 1e-3),   # m²/s
        initial_velocity=(0.05, 0.0, 0.0),      # m/s
        species_name = "U-234",
        source_positions=[(25, 25, 15)],
        emission_rate=1.0,
        wall_deposition=1e-4           # m/s
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

        self.__velocity = self._build_velocity_field_for_closed_room()

        self._init_poisson_matrix()

        # ==========================
        # Verificación estabilidad
        # ==========================
        if (diffusion_comprobation(self.__diffusion_coefficient, d) == False) or (CFL_comprobation(self.__velocity, d) == False):
            raise ValueError("The provided values for the function 'diffusion_advection_decay' do not follow the stability conditions of the equation.")

    def _init_poisson_matrix(self):
        """Construye la matriz A asegurando que sea invertible y bien condicionada"""
        Nx, Ny, Nz = self.__N
        total_cells = Nx * Ny * Nz
        A = lil_matrix((total_cells, total_cells))

        def get_idx(i, j, k):
            return i * (Ny * Nz) + j * Nz + k

        for i in range(total_cells):
            # Por defecto, ponemos un 1 en la diagonal para las celdas de frontera
            A[i, i] = 1.0

        # Llenamos las ecuaciones de Poisson SOLO para los puntos internos
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    idx = get_idx(i, j, k)

                    # Forzamos un punto de referencia (presión de medición = 0)
                    # Elegimos la primera celda interna [1, 1, 1] para romper la singularidad
                    if i == 1 and j == 1 and k == 1:
                        A[idx, idx] = 1.0
                        # No añadimos vecinos para que p[1,1,1] dependa únicamente de div
                        continue

                    # Ecuación estándar del laplaciano de 7 puntos
                    A[idx, idx] = -6.0
                    A[idx, get_idx(i+1, j, k)] = 1.0
                    A[idx, get_idx(i-1, j, k)] = 1.0
                    A[idx, get_idx(i, j+1, k)] = 1.0
                    A[idx, get_idx(i, j-1, k)] = 1.0
                    A[idx, get_idx(i, j, k+1)] = 1.0
                    A[idx, get_idx(i, j, k-1)] = 1.0

        self.__A_csr = A.tocsr()

    # ==========================================================
    # Campo de velocidades con no-slip aproximado (utilizamos las condiciones de conservación de masa y navier stokes, así como el hecho de que existe no-slip, es decir, que no hay velocidad en la superficie de las paredes)
    # ==========================================================
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

        return [vx, vy, vz]

    def _laplacian(self, f):
        """Calcula el Laplaciano discreto de 7 puntos en 3D"""
        dx2 = self.__d[0]**2
        lap = np.zeros_like(f)
        # Diferencias finitas (asumiendo dx = dy = dz = 1 por simplicidad)
        lap[1:-1, 1:-1, 1:-1] = (
            (f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1] - 2*f[1:-1, 1:-1, 1:-1]) / dx2 +
            (f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1] - 2*f[1:-1, 1:-1, 1:-1]) / dx2 +
            (f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2] - 2*f[1:-1, 1:-1, 1:-1]) / dx2
        )
        return lap

    def _step_velocity(self):
        # 1. ADVECCIÓN (Simplificada)
        vx_prev, vy_prev, vz_prev = self.__velocity[0].copy(), self.__velocity[1].copy(), self.__velocity[2].copy()

        # 2. DIFUSIÓN (Viscosidad Cinemática Real del Aire en SI: ~1.5e-5 m²/s)
        # Nota: Multiplicar por un factor de seguridad bajo para evitar rebasar el límite explícito
        nu = 1.5e-5
        dt = self.__d[3]

        self.__velocity[0] += dt * nu * self._laplacian(vx_prev)
        self.__velocity[1] += dt * nu * self._laplacian(vy_prev)
        self.__velocity[2] += dt * nu * self._laplacian(vz_prev)

        # Aplicar condiciones de contorno de habitación cerrada (No-slip)
        self._apply_boundaries_velocity()

        # 3. PROYECCIÓN (Incompresibilidad)
        # 3.1 Calcular Divergencia
        dx, dy, dz = self.__d[0], self.__d[1], self.__d[2]
        div = np.zeros_like(self.__velocity[0])

        div[1:-1, 1:-1, 1:-1] = (
            (self.__velocity[0][2:, 1:-1, 1:-1] - self.__velocity[0][:-2, 1:-1, 1:-1]) / (2.0 * dx) +
            (self.__velocity[1][1:-1, 2:, 1:-1] - self.__velocity[1][1:-1, :-2, 1:-1]) / (2.0 * dy) +
            (self.__velocity[2][1:-1, 1:-1, 2:] - self.__velocity[2][1:-1, 1:-1, :-2]) / (2.0 * dz)
        )

        # 3.2 Resolver Poisson para la Presión (∇²p = div)
        # Nota: En producción, 'p' se resuelve aplanando la matriz y usando un solver lineal (CG/Krylov)
        p = self._solve_poisson(div)

        # 3.3 Restar el Gradiente de Presión
        # 3.3 Restar el Gradiente de Presión corregido espacialmente
        dx, dy, dz = self.__d[0], self.__d[1], self.__d[2]

        self.__velocity[0][1:-1, 1:-1, 1:-1] -= (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * dx)
        self.__velocity[1][1:-1, 1:-1, 1:-1] -= (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * dy)
        self.__velocity[2][1:-1, 1:-1, 1:-1] -= (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * dz)

        self._apply_boundaries_velocity()

    def _apply_boundaries_velocity(self):
        """Fuerza que la velocidad en las paredes de la habitación sea cero"""
        for v in self.__velocity:
            v[0, :, :] = v[-1, :, :] = 0
            v[:, 0, :] = v[:, -1, :] = 0
            v[:, :, 0] = v[:, :, -1] = 0

    def _solve_poisson(self, div):
        """Resuelve el sistema lineal de presión usando un solver directo robusto"""
        # 1. Asegurar que las fronteras de la divergencia no inyecten ruido al sistema
        b = div.copy()

        # Forzar que los bordes del término independiente sean 0
        # ya que en la matriz A pusimos la identidad (1.0) en las fronteras
        b[0, :, :] = b[-1, :, :] = 0
        b[:, 0, :] = b[:, -1, :] = 0
        b[:, :, 0] = b[:, :, -1] = 0

        # Forzar la presión de referencia en el nodo [1, 1, 1]
        b[1, 1, 1] = 0.0

        # 2. Aplanar el término independiente corregido
        b_flat = b.flatten()

        # 3. Resolver usando solver directo (adiós problemas de overflow en productos punto)
        try:
            p_flat = spsolve(self.__A_csr, b_flat)
        except:
            # En caso de emergencia si la matriz se desborda
            p_flat = np.zeros_like(b_flat)

        # 4. Reestructurar a la matriz 3D original
        p = p_flat.reshape(self.__N)

        # 5. Aplicar condiciones de contorno Neumann para la presión
        p[0, :, :] = p[1, :, :]
        p[-1, :, :] = p[-2, :, :]
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]

        return p


    # ==========================================================
    # Difusión
    # ==========================================================
    def _compute_diffusion(self, concentration_aux):

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
    def _compute_advection(self, concentration_aux):

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

        alpha_x = self.__wall_deposition * self.__d[0] / self.__diffusion_coefficient[0]
        alpha_y = self.__wall_deposition * self.__d[1] / self.__diffusion_coefficient[1]
        alpha_z = self.__wall_deposition * self.__d[2] / self.__diffusion_coefficient[2]

        # ======================================
        # Robin BC:
        #
        # -D dC/dn = vd C
        # ======================================

        self.__concentration[0,:,:] = self.__concentration[1,:,:] / (1 + alpha_x)
        self.__concentration[-1,:,:] = self.__concentration[-2,:,:] / (1 + alpha_x)

        self.__concentration[:,0,:] = self.__concentration[:,1,:] / (1 + alpha_y)
        self.__concentration[:,-1,:] = self.__concentration[:,-2,:] / (1 + alpha_y)

        self.__concentration[:,:,0] = self.__concentration[:,:,1] / (1 + alpha_z)
        self.__concentration[:,:,-1] = self.__concentration[:,:,-2] / (1 + alpha_z)

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

    def animate(self, z=None):

        if z is None:
            z = self.N[2] // 2

        times = sorted(self.__saved_fields.keys())

        fig, ax = plt.subplots(figsize=(8,6))

        first = self.__saved_fields[times[0]]

        im = ax.imshow(
            first[:,:,z].T,
            origin='lower',
            extent=[0,self.__N[0],0,self.__N[1]],
            animated=True
        )

        plt.colorbar(im)

        def update(frame):

            t = times[frame]

            im.set_array(
                self.__saved_fields[t][:,:,z].T
            )

            ax.set_title(f"t = {t:.2f} s")

            return [im]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=100
        )

        plt.show()

        return ani

