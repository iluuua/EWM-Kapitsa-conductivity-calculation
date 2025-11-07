import numpy as np
from .constants import hbar, kB, pi


def bose_kernel(omega, T):
    x = np.clip(hbar * omega / (kB * T), None, 80.0)
    ex = np.exp(x)
    den = np.maximum(ex - 1.0, 1e-30)
    return ((hbar*omega) ** 2 / (kB * T * T)) * (ex / (den * den))


def DOS_debye(omega, v):
    return (omega ** 2) / (2 * pi ** 2 * v ** 3)
