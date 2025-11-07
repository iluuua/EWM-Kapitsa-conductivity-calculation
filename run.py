# -*- coding: utf-8 -*-
# Runs the sweep. Produces ONLY: 2 grouped PNGs (with two stacks) + one Excel (sheet per σ).

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kapitza import *


def run_kapitza_sweep_make_plots_and_excel(
        hk_fn,
        pairs,
        stack_map,  # dict: stack_name -> tuple(pair_name1, pair_name2, ...)
        T_list=(50, 100, 150, 200, 250, 300),
        sigma_list=(0e-9, 1e-9, 5e-9, 10e-9, 16e-9),
        Lcorr=10e-9,
        theta_min_deg=0.0,
        n_theta=161, nphi=81, n_omega=401,
        outdir="plots",
        excel_filename=None
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # palette for σ
    sigmas_nm = [int(round(s * 1e9)) for s in sigma_list]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, len(sigmas_nm))]
    colors_map = {nm: col for nm, col in zip(sigmas_nm, colors)}

    # name -> (A,B) lookup
    pair_dict = {name: (A, B) for name, A, B in pairs}

    def _hk_safe(A, B, T, sigma):
        h = hk_fn(A, B, float(T),
                  sigma=float(sigma), Lcorr=float(Lcorr),
                  theta_min_deg=float(theta_min_deg),
                  n_theta=n_theta, nphi=nphi, n_omega=n_omega,
                  symmetric=True)
        if (not np.isfinite(h)) or (h <= 0.0):
            h = hk_landauer_amm(A, B, float(T),
                                theta_min_deg=float(theta_min_deg),
                                n_theta=n_theta, n_omega=n_omega)
        return max(float(h), 0.0)

    def _compute_pair_table(A, B, sigma):
        h_vals = [_hk_safe(A, B, T, sigma) for T in T_list]
        R_vals = [(1.0 / h if h > 0 else np.inf) for h in h_vals]
        return np.asarray(h_vals, float), np.asarray(R_vals, float)

    # ---- compute for all σ ----
    # by_sigma[sigma_key][name] = {"h": arr, "R": arr}  for both pairs and stacks
    by_sigma = {}
    for sigma in sigma_list:
        sigma_key = f"{int(round(sigma * 1e9))}nm"
        per_sigma = {}

        # 1) all pairs in `pairs`
        for name, A, B in pairs:
            h_arr, R_arr = _compute_pair_table(A, B, sigma)
            per_sigma[name] = {"h": h_arr, "R": R_arr}

        # 2) each stack in stack_map: sum Rs of its constituent pairs
        for stack_name, pair_names in stack_map.items():
            R_stack = None
            for pname in pair_names:
                if pname not in per_sigma:
                    # if a needed pair is not precomputed, compute ad-hoc
                    A, B = pair_dict[pname]
                    h_arr, R_arr = _compute_pair_table(A, B, sigma)
                    per_sigma[pname] = {"h": h_arr, "R": R_arr}
                arr = per_sigma[pname]["R"]
                R_stack = arr if R_stack is None else (R_stack + arr)
            h_stack = np.where(np.isfinite(R_stack) & (R_stack > 0.0), 1.0 / R_stack, 0.0)
            per_sigma[stack_name] = {"h": h_stack, "R": R_stack}

        by_sigma[sigma_key] = per_sigma

    # ---- Excel (sheet per σ) ----
    if excel_filename is None:
        excel_filename = f"hk_results_by_sigma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    xlsx_path = out / excel_filename
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        for sigma in sigma_list:
            key = f"{int(round(sigma * 1e9))}nm"
            rows = []
            for name, val in by_sigma[key].items():  # includes all pairs and both stacks
                for T, hK, R in zip(T_list, val["h"], val["R"]):
                    rows.append({"Interface": name, "T_K": int(T),
                                 "h_Wm2K": float(hK), "R_m2KW": float(R)})
            df = pd.DataFrame(rows, columns=["Interface", "T_K", "h_Wm2K", "R_m2KW"])
            sheet = f"sigma_{int(round(float(sigma) * 1e9))}nm"
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            wb = writer.book
            fmt_sci = wb.add_format({"num_format": "0.00E+00"})
            ws.freeze_panes(1, 0)
            ws.set_column("A:A", 26)
            ws.set_column("B:B", 8)
            ws.set_column("C:D", 16, fmt_sci)

    # ---- Two grouped figures (2×2) with two stacks on panel 4 ----
    T = np.array(T_list, float)
    # собрать только нужные интерфейсы и оба стека
    per_ifc_dict = {}
    # панели интерфейсов: Al|SiO2, Al|Si3N4, SiO2|Si
    panel_ifcs = ("Al SiO2", "Al Si3N4", "SiO2 Si")
    for name in panel_ifcs:
        per_ifc_dict[name] = {key: by_sigma[key][name] for key in by_sigma.keys()}
    # оба стека
    for st_name in stack_map.keys():
        per_ifc_dict[st_name] = {key: by_sigma[key][st_name] for key in by_sigma.keys()}

    plot_group_two_stacks(T, per_ifc_dict, sigma_list, colors_map, out)

    print(f"Saved images & Excel in: {out.resolve()}")
    print(f"Excel: {xlsx_path.resolve()}")
    return {"excel_path": str(xlsx_path.resolve())}


if __name__ == "__main__":
    pairs = default_pairs()
    stacks = stack_variants()

    # sanity
    h_test = hk_landauer_exact(Al, SiO2, 300, sigma=16e-9, Lcorr=10e-9,
                               theta_min_deg=0.0, n_theta=101, nphi=81, n_omega=401)
    print(f"[sanity] h_K(Al SiO2, 300K, σ=16nm) ≈ {h_test:.3e} W/m^2/K")

    run_kapitza_sweep_make_plots_and_excel(
        hk_fn=hk_landauer_exact,
        pairs=pairs,
        stack_map=stacks,
        T_list=(50, 100, 150, 200, 250, 300),
        sigma_list=(0e-9, 1e-9, 5e-9, 10e-9, 16e-9),
        Lcorr=10e-9,
        theta_min_deg=0.0,
        n_theta=900, nphi=300, n_omega=1800,
        outdir="plots",
        excel_filename=None
    )
