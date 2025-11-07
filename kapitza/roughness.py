import numpy as np
from .constants import pi


def W_phi(phis, gamma):
    c = np.cos(phis)
    t = np.tan(phis)
    return (1.0 / (gamma * np.sqrt(2 * np.pi) * c ** 2)) * np.exp(-(t ** 2) / (2 * gamma ** 2))


def theta_local(thetas, phis):
    theta = thetas[:, None]
    phi = phis[None, :]
    S = theta + phi
    left = (phi <= -theta)
    mid = (~left) & (phi < (pi / 2 - theta))
    right = ~(left | mid)
    theta_0 = np.empty_like(S, dtype=np.float64)
    theta_0[left] = -S[left]
    theta_0[mid] = S[mid]
    theta_0[right] = pi / 2

    return np.clip(theta_0, 0.0, pi / 2)


def angle_average_A(thetas, get_alpha, *, sigma=0.0, Lcorr=10e-9, nphi=121):
    w_theta = np.cos(thetas) * np.sin(thetas)
    if sigma <= 0 or Lcorr <= 0:
        a = get_alpha(thetas)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)

        return max(np.trapz(a * w_theta, thetas), 0.0)

    gamma = sigma / Lcorr
    eps = 1e-6
    phis = np.linspace(-pi / 2 + eps, pi / 2 - eps, nphi)
    wphi = W_phi(phis, gamma)

    theta_0 = theta_local(thetas, phis)
    shadow = theta_0 >= (pi / 2 - 1e-8)
    alpha = get_alpha(theta_0.ravel()).reshape(theta_0.shape)
    alpha = np.where(shadow, 0.0, alpha)
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)

    num = np.trapz(alpha * wphi, phis, axis=1)
    den = np.trapz(wphi, phis)
    alpha_eff = num / max(den, 1e-30)
    A_val = np.trapz(alpha_eff * w_theta, thetas)

    if not np.isfinite(A_val) or A_val <= 0.0:
        a_spec = np.nan_to_num(get_alpha(thetas), nan=0.0, posinf=1.0, neginf=0.0)
        A_val = np.trapz(a_spec * w_theta, thetas)
        A_val = max(A_val, 0.0)

    return A_val
