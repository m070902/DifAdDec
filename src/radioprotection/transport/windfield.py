import numpy as np

class WindField:

    def __init__(self, grid_shape):
        self.Nx, self.Ny, self.Nz = grid_shape

        x = np.arange(self.Nx)
        y = np.arange(self.Ny)
        z = np.arange(self.Nz)

        self.X, self.Y, self.Z = np.meshgrid(
            x, y, z, indexing="ij"
        )

class UniformField(WindField):
    def __init__(self, grid_shape, initial_velocity=(5.0, 0.0, 0.0)):

        super().__init__(grid_shape)

        u = np.full((self.Nx, self.Ny, self.Nz), initial_velocity[0])
        v = np.full((self.Nx, self.Ny, self.Nz), initial_velocity[1])
        w = np.full((self.Nx, self.Ny, self.Nz), initial_velocity[2])

        self._u, self._v, self._w =  u, v, w

    def get_velocity(self, **kwargs):
        return self._u, self._v, self._w

class ShearField(WindField):
    def __init__(self,
                   grid_shape,
                   Uref=5.0,
                   zref=10,
                   alpha=0.20):

        super().__init__(grid_shape)

        u = np.zeros((self.Nx, self.Ny, self.Nz))

        for k in range(self.Nz):

            height = max(k + 1, 1)

            velocity = Uref * (height / zref) ** alpha

            u[:, :, k] = velocity

        v = np.zeros_like(u)
        w = np.zeros_like(u)

        self._u, self._v, self._w =  u, v, w

    def get_velocity(self, **kwargs):
        return self._u, self._v, self._w

class GustField(WindField):
    def __init__(self,
             grid_shape,
             Umean=5.0,
             amplitude=2.0,
             period=120):

        super().__init__(grid_shape)
        self.__Umean = Umean
        self.__amplitude = amplitude
        self.__period = period

    def get_velocity(self, **kwargs):
        velocity = self.__Umean + self.__amplitude * np.sin(
            2 * np.pi * next(iter(kwargs.values())) / self.__period
        )

        u = np.full((self.Nx, self.Ny, self.Nz), velocity)
        v = np.zeros_like(u)
        w = np.zeros_like(u)

        return u, v, w

class VortexField(WindField):
    def __init__(self,
               grid_shape,
               omega=0.02):

        super().__init__(grid_shape)

        xc = self.Nx / 2
        yc = self.Ny / 2

        u2d = -omega * (self.Y[:, :, 0] - yc)
        v2d =  omega * (self.X[:, :, 0] - xc)

        u = np.repeat(u2d[:, :, np.newaxis], self.Nz, axis=2)
        v = np.repeat(v2d[:, :, np.newaxis], self.Nz, axis=2)
        w = np.zeros_like(u)

        self._u, self._v, self._w =  u, v, w

    def get_velocity(self, **kwargs):
        return self._u, self._v, self._w