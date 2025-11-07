import numpy as np
from .constants import pi


def W_phi(phis, gamma):
    c = np.cos(phis)
    t = np.tan(phis)
    return (1.0 / (gamma * np.sqrt(2 * np.pi) * c ** 2)) * np.exp(-(t ** 2) / (2 * gamma ** 2))


def theta_local(thetas, phis):
    θ = thetas[:, None]
    φ = phis[None, :]
    S = θ + φ
    left = (φ <= -θ)
    mid = (~left) & (φ < (pi / 2 - θ))
    right = ~(left | mid)
    θ0 = np.empty_like(S, dtype=np.float64)
    θ0[left] = -S[left]
    θ0[mid] = S[mid]
    θ0[right] = pi / 2

    return np.clip(θ0, 0.0, pi / 2)


def angle_average_A(thetas, get_alpha, *, sigma=0.0, Lcorr=10e-9, nphi=121):
    wθ = np.cos(thetas) * np.sin(thetas)
    if sigma <= 0 or Lcorr <= 0:
        a = get_alpha(thetas)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)

        return max(np.trapz(a * wθ, thetas), 0.0)

    gamma = sigma / Lcorr
    eps = 1e-6
    phis = np.linspace(-pi / 2 + eps, pi / 2 - eps, nphi)
    wphi = W_phi(phis, gamma)

    θ0 = theta_local(thetas, phis)
    shadow = θ0 >= (pi / 2 - 1e-8)
    α = get_alpha(θ0.ravel()).reshape(θ0.shape)
    α = np.where(shadow, 0.0, α)
    α = np.nan_to_num(α, nan=0.0, posinf=1.0, neginf=0.0)

    num = np.trapz(α * wphi, phis, axis=1)
    den = np.trapz(wphi, phis)
    αeff = num / max(den, 1e-30)
    A_val = np.trapz(αeff * wθ, thetas)

    if not np.isfinite(A_val) or A_val <= 0.0:
        a_spec = np.nan_to_num(get_alpha(thetas), nan=0.0, posinf=1.0, neginf=0.0)
        A_val = np.trapz(a_spec * wθ, thetas)
        A_val = max(A_val, 0.0)

    return A_val
