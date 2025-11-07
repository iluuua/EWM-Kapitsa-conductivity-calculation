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
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def _finite_xy(x, y):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    return x[m], y[m]


def plot_group_all_interfaces(T_arr, per_ifc_dict, sigma_list, colors_map, outdir):
    """Сохраняет ровно ДВЕ картинки: group_h_all.png и group_R_all.png (по 4 графика каждая)."""
    _setup_matplotlib()
    interfaces = list(per_ifc_dict.keys())  # 3 пары + 'stack'
    # ----- h_K -----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for i, name in enumerate(interfaces):
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

    leg_labels = [f"{int(round(s*1e9))}" for s in sigma_list]
    handles = [plt.Line2D([0],[0], color=colors_map[int(round(s * 1e9))], lw=2.5) for s in sigma_list]

    fig.legend(handles, leg_labels, title="σ (nm)", loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels), frameon=False)
    fig.suptitle("h_K(T)", y=1.02, fontsize=14)
    fig.savefig(str(Path(outdir) / "group_h_all.png"), bbox_inches="tight")
    plt.close(fig)

    # ----- R -----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for i, name in enumerate(interfaces):
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

    handles = [plt.Line2D([0],[0], color=colors_map[int(round(s * 1e9))], lw=2.5) for s in sigma_list]

    fig.legend(handles, leg_labels, title="σ (nm)", loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(leg_labels), frameon=False)
    fig.suptitle("R(T)", y=1.02, fontsize=14)
    fig.savefig(str(Path(outdir) / "group_R_all.png"), bbox_inches="tight")
    plt.close(fig)
