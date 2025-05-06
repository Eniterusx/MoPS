import numpy as np
from numba import jit

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def sdm_parallel(pairs, u01, mult, mass, coeff, dt, dv):
    p_scale = len(mult) * (len(mult) - 1) / 2 / len(pairs)
    eps = 1e-6
    for alpha in range(len(pairs)):
        j, k = pairs[alpha]
        if mult[j] < mult[k]:
            j, k = k, j
        p_alpha = mult[j] * p_scale * coeff * (mass[j] + mass[k]) * dt / dv
        gamma = int(p_alpha // 1 + (p_alpha - p_alpha // 1) > u01[alpha])
        if gamma != 0:
            gamma = min(gamma, mult[j] // (mult[k] + eps))
            if mult[j] - gamma * mult[k] > 0:
                mult[j] -= gamma * mult[k]
                mass[k] += gamma * mass[j]
            else:
                mult[j] = mult[k] // 2
                mult[k] -= mult[j]
                mass[k] += gamma * mass[j]
                mass[j] = mass[k]