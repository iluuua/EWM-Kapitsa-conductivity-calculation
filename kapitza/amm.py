import numpy as np
from .physics import bose_kernel, DOS_debye


def alpha_AMM_branch(v1, v2, rho1, rho2, thetas):
    s1 = np.sin(thetas)
    c1 = np.cos(thetas)
    r = (v2 / v1)
    s2 = np.clip(r * s1, -0.999999, 0.999999)
    c2 = np.sqrt(1.0 - s2 ** 2)
    Z1 = rho1 * v1
    Z2 = rho2 * v2
    num = 4.0 * Z1 * Z2 * c1 * c2
    den = (Z1 * c1 + Z2 * c2) ** 2 + 1e-30

    return np.clip(num / den, 0.0, 1.0)


def angle_average_AMM(mat_i, mat_t, thetas):
    w = np.cos(thetas) * np.sin(thetas)
    A_P = np.trapz(alpha_AMM_branch(mat_i.vP, mat_t.vP, mat_i.rho, mat_t.rho, thetas) * w, thetas)
    A_S = np.trapz(alpha_AMM_branch(mat_i.vS, mat_t.vS, mat_i.rho, mat_t.rho, thetas) * w, thetas)

    return max(A_P, 0.0), max(A_S, 0.0)


def hk_landauer_amm(mat1, mat2, T, *, theta_min_deg=0.0, n_theta=241, n_omega=801):
    thetas = np.linspace(np.radians(theta_min_deg), np.pi / 2, n_theta)
    A_P_12, A_S_12 = angle_average_AMM(mat1, mat2, thetas)
    A_P_21, A_S_21 = angle_average_AMM(mat2, mat1, thetas)
    A_P = 0.5 * (A_P_12 + A_P_21)
    A_S = 0.5 * (A_S_12 + A_S_21)

    wmax_P = min(mat1.wD_P, mat2.wD_P)
    wmax_S = min(mat1.wD_S, mat2.wD_S)
    wmax = max(wmax_P, wmax_S)
    omegas = np.linspace(1e-12, wmax, n_omega)
    KER = bose_kernel(omegas, T)

    def fint(mat, A_P, A_S):
        Dp = DOS_debye(omegas, mat.vP)
        Ds = DOS_debye(omegas, mat.vS)
        IP = np.trapz((mat.vP * Dp * KER * A_P) * (omegas <= wmax_P), omegas)
        IS = np.trapz((mat.vS * Ds * KER * A_S) * (omegas <= wmax_S), omegas)

        return float(np.nan_to_num(IP + IS, nan=0.0, posinf=0.0, neginf=0.0))

    return max(0.5 * (fint(mat1, A_P, A_S) + fint(mat2, A_P, A_S)), 0.0)
