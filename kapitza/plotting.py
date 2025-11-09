import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import EngFormatter, AutoMinorLocator, LogLocator, LogFormatterSciNotation


def _setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 10, "axes.grid": True,
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
    })


def _style_axis(ax, ylabel, yscale="linear"):
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yscale == "log":
        major = LogLocator(base=10.0, numticks=12)
        minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
        ax.yaxis.set_major_locator(major)
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(EngFormatter(unit=""))
    ax.grid(which="major", alpha=0.7)
    ax.grid(which="minor", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _finite_xy(x, y):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    return x[m], y[m]


def plot_group_two_stacks(
        T_arr,
        per_ifc_dict,
        sigma_list,
        colors_map,
        outdir,
        stack_names=("Al-SiO2-Si", "Al-Si3N4-Si"),
        panel_interfaces=("Al SiO2", "Al Si3N4", "SiO2 Si")
):
    """
    Creates and saves two graph pictures:
      - Three of them are interfaces;
      - Fourth panel shows two stacks.
    """
    _setup_matplotlib()

    # ---------- h_K ----------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    # 3 interfaces
    for i, name in enumerate(panel_interfaces):
        ax = axes[i // 2, i % 2]
        for s in sigma_list:
            nm = int(round(s * 1e9))
            key = f"{nm}nm"
            clr = colors_map[nm]
            x, y = _finite_xy(T_arr, per_ifc_dict[name][key]["h"])
            ax.plot(x, y, marker="o", lw=2.0, ms=5.0, color=clr, label=f"{nm}")
        ax.set_title(name)
        ax.set_xlabel("T (K)")
        _style_axis(ax, "h_K (W·m$^{-2}$·K$^{-1}$)", "linear")

    # Two stacks
    ax = axes[1, 1]
    linestyles = {stack_names[0]: "-", stack_names[1]: "--"}
    for s in sigma_list:
        nm = int(round(s * 1e9))
        key = f"{nm}nm"
        clr = colors_map[nm]
        for st in stack_names:
            x, y = _finite_xy(T_arr, per_ifc_dict[st][key]["h"])
            ax.plot(x, y, lw=2.2, color=clr, linestyle=linestyles[st])
    ax.set_title(f"Stacks: {stack_names[0]} (—), {stack_names[1]} (– –)")
    ax.set_xlabel("T (K)")
    _style_axis(ax, "h_K (W·m$^{-2}$·K$^{-1}$)", "linear")

    # Legend by σ
    leg_labels = [f"{int(round(s * 1e9))}" for s in sigma_list]
    handles = [plt.Line2D([0], [0], color=colors_map[int(round(s * 1e9))], lw=2.5) for s in sigma_list]
    fig.legend(handles, leg_labels, title="σ (nm)", loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels), frameon=False)
    fig.suptitle("h_K(T)", y=1.02, fontsize=14)
    fig.savefig(str(Path(outdir) / "group_h_all.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------- R ----------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    # 3 interfaces
    for i, name in enumerate(panel_interfaces):
        ax = axes[i // 2, i % 2]
        for s in sigma_list:
            nm = int(round(s * 1e9))
            key = f"{nm}nm"
            clr = colors_map[nm]
            x, y = _finite_xy(T_arr, per_ifc_dict[name][key]["R"])
            ax.plot(x, y, marker="o", lw=2.0, ms=5.0, color=clr, label=f"{nm}")
        ax.set_title(name)
        ax.set_xlabel("T (K)")
        _style_axis(ax, "R (m$^2$·K·W$^{-1}$)", "log")

    # Two stacks
    ax = axes[1, 1]
    for s in sigma_list:
        nm = int(round(s * 1e9))
        key = f"{nm}nm"
        clr = colors_map[nm]
        for st in stack_names:
            x, y = _finite_xy(T_arr, per_ifc_dict[st][key]["R"])
            ax.plot(x, y, lw=2.2, color=clr, linestyle=linestyles[st])
    ax.set_title(f"Stacks: {stack_names[0]} (—), {stack_names[1]} (– –)")
    ax.set_xlabel("T (K)")
    _style_axis(ax, "R (m$^2$·K·W$^{-1}$)", "log")

    # Legend by σ
    handles = [plt.Line2D([0], [0], color=colors_map[int(round(s * 1e9))], lw=2.5) for s in sigma_list]
    fig.legend(handles, leg_labels, title="σ (nm)", loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels), frameon=False)
    fig.suptitle("R(T)", y=1.02, fontsize=14)
    fig.savefig(str(Path(outdir) / "group_R_all.png"), bbox_inches="tight")
    plt.close(fig)
