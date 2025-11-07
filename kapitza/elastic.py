import numpy as np
from .materials import Material


def lame_params(mat: Material):
    mu = mat.rho * mat.vS ** 2
    lam = mat.rho * mat.vP ** 2 - 2 * mu

    return lam, mu


def kz_from_kx(k_tot, kx):
    return np.lib.scimath.sqrt(k_tot ** 2 - kx ** 2)


def fields_P(kx, kz, lam, mu, amp):
    kx = np.asarray(kx, dtype=np.complex128)
    kz = np.asarray(kz, dtype=np.complex128)
    amp = np.asarray(amp, dtype=np.complex128)
    lam = float(lam)
    mu = float(mu)
    ux = 1j * kx * amp
    uz = 1j * kz * amp
    k2 = (kx ** 2 + kz ** 2)
    sig_xz = (mu * (-2 * kx * kz)) * amp
    sig_zz = (-(lam * k2 + 2 * mu * kz ** 2)) * amp

    return ux, uz, sig_xz, sig_zz


def fields_SV(kx, kz, mu, amp):
    kx = np.asarray(kx, dtype=np.complex128)
    kz = np.asarray(kz, dtype=np.complex128)
    amp = np.asarray(amp, dtype=np.complex128)
    mu = float(mu)
    ux = 1j * kz * amp
    uz = -1j * kx * amp
    sig_xz = (mu * (kx ** 2 - kz ** 2)) * amp
    sig_zz = (2 * mu * kx * kz) * amp
    return ux, uz, sig_xz, sig_zz


def energy_flux_z(ux, uz, sxz, szz, omega=1.0):
    vx = -1j * omega * ux
    vz = -1j * omega * uz
    S = sxz * np.conj(vx) + szz * np.conj(vz)
    Pz = 0.5 * np.real(S)

    return np.maximum(Pz, 0.0) + 1e-30


def _forward_kz(kz):
    kz = np.asarray(kz, dtype=np.complex128)
    return np.where(np.real(kz) >= 0, kz, -kz)


def solve_stable(A, b, ridge=1e-12):
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    if not np.isfinite(x).all():
        AH = A.conj().transpose(0, 2, 1) if A.ndim == 3 else A.conj().T
        I = np.eye(A.shape[-1], dtype=np.complex128)
        try:
            x = np.linalg.solve(AH @ A + ridge * I, AH @ b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(AH @ A + ridge * I, AH @ b, rcond=None)

    return x


def alpha_PSV_batch(mat_i: Material, mat_t: Material, thetas, incident='P'):
    th = np.asarray(thetas, dtype=np.float64)
    lam1, mu1 = lame_params(mat_i)
    lam2, mu2 = lame_params(mat_t)
    w = 1.0
    kP1 = w / mat_i.vP
    kS1 = w / mat_i.vS
    kP2 = w / mat_t.vP
    kS2 = w / mat_t.vS
    s = np.sin(th)
    ones = lambda x: np.ones_like(x, dtype=np.complex128)

    if incident.upper() == 'P':
        kx = kP1 * s
        kzPi = _forward_kz(kz_from_kx(kP1, kx))
        kzPr = -kzPi
        kzSr = -_forward_kz(kz_from_kx(kS1, kx))
        ux_i, uz_i, sxz_i, szz_i = fields_P(kx, kzPi, lam1, mu1, ones(kx))
    else:
        kx = kS1 * s
        kzSi = _forward_kz(kz_from_kx(kS1, kx))
        kzSr = -kzSi
        kzPr = -_forward_kz(kz_from_kx(kP1, kx))
        ux_i, uz_i, sxz_i, szz_i = fields_SV(kx, kzSi, mu1, ones(kx))

    kzPt = _forward_kz(kz_from_kx(kP2, kx))
    kzSt = _forward_kz(kz_from_kx(kS2, kx))

    ux_rP, uz_rP, sxz_rP, szz_rP = fields_P(kx, kzPr, lam1, mu1, ones(kx))
    ux_rS, uz_rS, sxz_rS, szz_rS = fields_SV(kx, kzSr, mu1, ones(kx))
    ux_tP, uz_tP, sxz_tP, szz_tP = fields_P(kx, kzPt, lam2, mu2, ones(kx))
    ux_tS, uz_tS, sxz_tS, szz_tS = fields_SV(kx, kzSt, mu2, ones(kx))

    N = th.size
    A = np.empty((N, 4, 4), dtype=np.complex128)
    b = np.empty((N, 4), dtype=np.complex128)
    A[:, 0, :] = np.stack([ux_rP, ux_rS, -ux_tP, -ux_tS], axis=-1)
    A[:, 1, :] = np.stack([uz_rP, uz_rS, -uz_tP, -uz_tS], axis=-1)
    A[:, 2, :] = np.stack([sxz_rP, sxz_rS, -sxz_tP, -sxz_tS], axis=-1)
    A[:, 3, :] = np.stack([szz_rP, szz_rS, -szz_tP, -szz_tS], axis=-1)
    b[:] = -np.stack([ux_i, uz_i, sxz_i, szz_i], axis=-1)
    sol = solve_stable(A, b)
    tP, tSV = sol[:, 2], sol[:, 3]

    if incident.upper() == 'P':
        P_inc = energy_flux_z(*fields_P(kx, kzPi, lam1, mu1, ones(kx)))
    else:
        P_inc = energy_flux_z(*fields_SV(kx, kzSi, mu1, ones(kx)))
    P_tP = energy_flux_z(*fields_P(kx, kzPt, lam2, mu2, tP))
    P_tSV = energy_flux_z(*fields_SV(kx, kzSt, mu2, tSV))

    alpha = (P_tP + P_tSV) / np.maximum(P_inc, 1e-30)
    alpha = np.clip(np.nan_to_num(np.real(alpha), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return alpha.astype(np.float64)


def alpha_SH_batch(mat_i: Material, mat_t: Material, thetas):
    th = np.asarray(thetas, dtype=np.float64)
    mu1 = mat_i.rho * mat_i.vS ** 2
    mu2 = mat_t.rho * mat_t.vS ** 2
    w = 1.0
    kS1 = w / mat_i.vS
    kS2 = w / mat_t.vS
    kx = kS1 * np.sin(th)
    kz1 = _forward_kz(kz_from_kx(kS1, kx))
    kz2 = _forward_kz(kz_from_kx(kS2, kx))

    N = th.size
    A = np.empty((N, 2, 2), dtype=np.complex128)
    b = np.empty((N, 2), dtype=np.complex128)
    A[:, 0, 0] = 1.0
    A[:, 0, 1] = -1.0
    A[:, 1, 0] = 1j * mu1 * kz1
    A[:, 1, 1] = 1j * mu2 * kz2
    b[:, 0] = -1.0
    b[:, 1] = 1j * mu1 * kz1

    rt = solve_stable(A, b)
    t = rt[:, 1]

    def Psh(mu, kz, amp): return np.maximum(0.5 * mu * 1.0 * np.real(kz) * np.abs(amp) ** 2, 0.0) + 1e-30

    P_inc = Psh(mu1, kz1, np.ones_like(kx, dtype=np.complex128))
    P_tr = Psh(mu2, kz2, t)
    alpha = P_tr / np.maximum(P_inc, 1e-30)
    return np.clip(np.nan_to_num(np.real(alpha), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float64)
