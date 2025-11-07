from .constants import NA, pi, RAD, THz


class Material:
    __slots__ = ("rho", "vP", "vS", "wD_P", "wD_S", "name")

    def __init__(self, rho, vP, vS, wD_P, wD_S, name=""):
        self.rho = float(rho)
        self.vP = float(vP)
        self.vS = float(vS)
        self.wD_P = float(wD_P)
        self.wD_S = float(wD_S)
        self.name = name


def debye_velocity(vP, vS):
    return ((1.0 / 3.0) * (vP ** -3 + 2 * vS ** -3)) ** (-1 / 3)


def omega_D_from_rho_M_atoms(rho, M_kg_per_mol, atoms_per_fu, vP, vS):
    n_atoms = rho * NA / M_kg_per_mol * atoms_per_fu
    kD = (6 * pi ** 2 * n_atoms) ** (1 / 3)
    return debye_velocity(vP, vS) * kD


# ---- defaults per ТЗ
Al = Material(2700.0, 6240.0, 3040.0, RAD * 10.0 * THz, RAD * 5.8 * THz, name="Al")
Si = Material(2330.0, 8970.0, 5332.0, RAD * 12.3 * THz, RAD * 4.7 * THz, name="Si")

M_SiO2 = 60.0843e-3
wD_SiO2 = omega_D_from_rho_M_atoms(2200.0, M_SiO2, 3, 5968.0, 3764.0)
SiO2 = Material(2200.0, 5968.0, 3764.0, wD_SiO2, wD_SiO2, name="SiO2")

M_Si3N4 = 140.283e-3
wD_Si3N4 = omega_D_from_rho_M_atoms(3100.0, M_Si3N4, 7, 10000.0, 6000.0)
Si3N4 = Material(3100.0, 10000.0, 6000.0, wD_Si3N4, wD_Si3N4, name="Si3N4")


def default_pairs():
    """
    Interface pairs and stacks
    """
    return [
        ("Al Si3N4", Al, Si3N4),
        ("Si3N4 SiO2", Si3N4, SiO2),
        ("SiO2 Si", SiO2, Si),
        ("Al SiO2", Al, SiO2),
        ("Si3N4 Si", Si3N4, Si),
    ]


def stack_variants():
    """
    Two stacks: name -> sum of stacks' elements
    """
    return {
        "Al-SiO2-Si": ("Al SiO2", "SiO2 Si"),
        "Al-Si3N4-Si": ("Al Si3N4", "Si3N4 Si"),
    }
