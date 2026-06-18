import numpy as np

# Para controlar la estabilidad de la advección
def CFL_comprobation(v, d):
    courant_numbers = []
    for i in range(3):
        courant_numbers.append((np.max(v[i]) * d[3])/d[i])

    if (max(courant_numbers) <= 1) or ((courant_numbers[0]+courant_numbers[1]+courant_numbers[2]) <= 1):
        return True
    else:
        return False

# Para controlar la estabilidad de la difusión
def diffusion_comprobation(D, d):
    diffusion_numbers = []
    for i in range(3):
        diffusion_numbers.append((np.sqrt(D[0]**2+D[1]**2+D[2]**2) * d[3])/(d[i]**2))

    if (diffusion_numbers[0]+diffusion_numbers[1]+diffusion_numbers[2]) <= 1/2:
        return True
    else:
        return False