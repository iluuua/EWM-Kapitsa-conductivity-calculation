import numpy as np
from .physics import bose_kernel, DOS_debye
from .elastic import alpha_PSV_batch, alpha_SH_batch
from .roughness import angle_average_A
from .amm import hk_landauer_amm
from .constants import pi


def hk_landauer_exact(mat1, mat2, T, *, sigma=0.0, Lcorr=10e-9,
                      theta_min_deg=0.0, n_theta=241, nphi=121, n_omega=801,
                      symmetric=True):
    thetas = np.linspace(np.radians(theta_min_deg), pi / 2, n_theta)

    def avg_A_12(m1, m2):
        A_P = angle_average_A(thetas, lambda th: alpha_PSV_batch(m1, m2, th, 'P'),
                              sigma=sigma, Lcorr=Lcorr, nphi=nphi)
        A_SV = angle_average_A(thetas, lambda th: alpha_PSV_batch(m1, m2, th, 'SV'),
                               sigma=sigma, Lcorr=Lcorr, nphi=nphi)
        A_SH = angle_average_A(thetas, lambda th: alpha_SH_batch(m1, m2, th),
                               sigma=sigma, Lcorr=Lcorr, nphi=nphi)

        def fb(val, maker): return maker() if (not np.isfinite(val) or val <= 0) else val

        A_P = fb(A_P, lambda: angle_average_A(
                 thetas, lambda th: alpha_PSV_batch(m1, m2, th, 'P'),
                 sigma=0.0, Lcorr=Lcorr, nphi=nphi))

        A_SV = fb(A_SV, lambda: angle_average_A(
                  thetas, lambda th: alpha_PSV_batch(m1, m2, th, 'SV'),
                  sigma=0.0, Lcorr=Lcorr, nphi=nphi))

        A_SH = fb(A_SH, lambda: angle_average_A(
                  thetas, lambda th: alpha_SH_batch(m1, m2, th),
                  sigma=0.0, Lcorr=Lcorr, nphi=nphi))

        return A_P, A_SV, A_SH

    A_P_1, A_SV_1, A_SH_1 = avg_A_12(mat1, mat2)
    if symmetric:
        A_P_2, A_SV_2, A_SH_2 = avg_A_12(mat2, mat1)

    wmax_P = min(mat1.wD_P, mat2.wD_P)
    wmax_S = min(mat1.wD_S, mat2.wD_S)
    wmax = max(wmax_P, wmax_S)
    omegas = np.linspace(1e-12, wmax, n_omega)
    KER = bose_kernel(omegas, T)

    def fint(mat, A_P, A_SV, A_SH):
        Dp = DOS_debye(omegas, mat.vP)
        Ds = DOS_debye(omegas, mat.vS)
        IP = np.trapz((mat.vP * Dp * KER * A_P) * (omegas <= wmax_P), omegas)
        ISV = np.trapz((mat.vS * Ds * KER * A_SV) * (omegas <= wmax_S), omegas)
        ISH = np.trapz((mat.vS * Ds * KER * A_SH) * (omegas <= wmax_S), omegas)
        return float(np.nan_to_num(IP + ISV + ISH, nan=0.0, posinf=0.0, neginf=0.0))

    I1 = fint(mat1, A_P_1, A_SV_1, A_SH_1)
    h_exact = 0.5 * (I1 + fint(mat2, A_P_2, A_SV_2, A_SH_2)) if symmetric else I1
    if (not np.isfinite(h_exact)) or (h_exact <= 0.0) or (h_exact < 1e-60):
        return hk_landauer_amm(mat1, mat2, T,
                               theta_min_deg=theta_min_deg,
                               n_theta=n_theta, n_omega=n_omega)

    return h_exact
