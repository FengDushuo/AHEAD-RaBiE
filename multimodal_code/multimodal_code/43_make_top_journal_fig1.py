#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a polished vector Fig. 1 for the AddH-out multi-view workflow."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("outputs_addh_top_journal_fig1")


COLORS = {
    "ink": "#1B1B1B",
    "muted": "#5B6470",
    "grid": "#D6DCE5",
    "blue": "#2F6FA3",
    "blue_light": "#EAF3FA",
    "green": "#009E73",
    "green_light": "#EAF7F1",
    "pink": "#CC79A7",
    "orange": "#D55E00",
    "orange_light": "#FAEDE4",
    "gray_light": "#F4F6F8",
    "yellow_light": "#FFF4D7",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def box(ax, x, y, w, h, fc, ec, radius=0.025, lw=0.9, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, size=7.2, weight="regular", color=None, ha="center", va="center", **kwargs):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        linespacing=1.15,
        **kwargs,
    )


def arrow(ax, p1, p2, color=None, lw=0.9, rad=0.0, dashed=False, z=4):
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color or COLORS["ink"],
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def chip(ax, x, y, w, label, fc="#FFFFFF", ec=None, color=None):
    box(ax, x, y, w, 0.055, fc, ec or COLORS["grid"], radius=0.012, lw=0.75)
    text(ax, x + w / 2, y + 0.0275, label, size=6.4, weight="bold", color=color or COLORS["ink"])


def mini_lattice(ax, x, y, scale=1.0):
    pts = [(0, 0), (0.045, 0.025), (0.09, 0), (0.045, -0.025), (0, -0.05), (0.09, -0.05)]
    for i, (px, py) in enumerate(pts):
        c = COLORS["green"] if i in (1, 4) else "#8A8F98"
        ax.add_patch(Circle((x + px * scale, y + py * scale), 0.0085 * scale, facecolor=c, edgecolor="white", lw=0.4))
    for a, b in [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]:
        ax.plot(
            [x + pts[a][0] * scale, x + pts[b][0] * scale],
            [y + pts[a][1] * scale, y + pts[b][1] * scale],
            color="#9AA3AF",
            lw=0.5,
            zorder=1,
        )


def draw_panel_a(ax):
    text(ax, 0.055, 0.925, "A", size=10, weight="bold")
    text(ax, 0.100, 0.925, "Source-domain fine-tuning", size=8.2, weight="bold", ha="left")
    text(ax, 0.100, 0.895, "Fine-tune the multi-view backbone", size=6.0, color=COLORS["muted"], ha="left")

    box(ax, 0.055, 0.745, 0.155, 0.102, "#FFFFFF", COLORS["grid"], radius=0.018)
    text(ax, 0.1325, 0.810, "source labels", size=6.7, weight="bold")
    text(ax, 0.1325, 0.781, r"$D_s=\{(x_i,y_i)\}$", size=8.2)

    # Network glyph.
    x0, y0 = 0.070, 0.612
    layers = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    nodes = []
    for li in range(3):
        for ni in layers[li]:
            xx = x0 + li * 0.055
            yy = y0 + ni * 0.045
            color = COLORS["blue_light"] if li == 0 else ("#FFFFFF" if li == 1 else COLORS["green_light"])
            circ = Circle((xx, yy), 0.0105, facecolor=color, edgecolor=COLORS["muted"], lw=0.55, zorder=3)
            ax.add_patch(circ)
            nodes.append((xx, yy))
    for li in range(2):
        for a in range(3):
            for b in range(3):
                ax.plot(
                    [nodes[li * 3 + a][0] + 0.010, nodes[(li + 1) * 3 + b][0] - 0.010],
                    [nodes[li * 3 + a][1], nodes[(li + 1) * 3 + b][1]],
                    color="#C5CBD3",
                    lw=0.35,
                    zorder=1,
                )
    text(ax, 0.125, 0.565, "multi-view backbone", size=6.2, color=COLORS["muted"])

    box(ax, 0.055, 0.430, 0.210, 0.076, "#FFFFFF", COLORS["grid"], radius=0.014)
    text(ax, 0.160, 0.468, r"$\theta^\ast=\arg\min_\theta \mathcal{L}_{s}(\theta)$", size=8.6)
    text(ax, 0.160, 0.443, r"$\mathcal{L}_{s}=\frac{1}{N_s}\sum_i \rho_\delta(y_i-f_\theta(x_i))+\lambda\|\theta-\theta_0\|_2^2$", size=6.0)

    arrow(ax, (0.210, 0.795), (0.255, 0.655), color=COLORS["muted"], rad=-0.25)
    arrow(ax, (0.170, 0.585), (0.160, 0.506), color=COLORS["muted"])


def draw_panel_b(ax):
    text(ax, 0.355, 0.925, "B", size=10, weight="bold")
    text(ax, 0.398, 0.925, "Multi-view latent fusion", size=8.2, weight="bold", ha="left")
    text(ax, 0.398, 0.895, "Fuse structure, chemistry, statistics, and priors", size=6.0, color=COLORS["muted"], ha="left")

    labels = [
        ("composition", 0.330, 0.795),
        ("local geometry", 0.455, 0.795),
        ("graph embedding", 0.580, 0.795),
        ("element prior", 0.392, 0.712),
        ("source statistics", 0.517, 0.712),
    ]
    for label, x, y in labels:
        chip(ax, x, y, 0.105, label)

    box(ax, 0.410, 0.565, 0.210, 0.095, COLORS["blue_light"], COLORS["blue"], radius=0.018, lw=0.9)
    text(ax, 0.515, 0.622, r"view encoders $\phi_v$", size=7.4, weight="bold")
    text(ax, 0.515, 0.590, r"$h_i^{(v)}=\phi_v(x_i^{(v)})$", size=8.1)

    for _, x, y in labels:
        arrow(ax, (x + 0.052, y), (0.515, 0.660), color="#77808A", lw=0.65)

    # Attention bars.
    for k, h in enumerate([0.075, 0.052, 0.088, 0.043, 0.062]):
        x = 0.357 + k * 0.025
        ax.add_patch(FancyBboxPatch((x, 0.438), 0.014, h, boxstyle="round,pad=0.001,rounding_size=0.004", facecolor=COLORS["blue"], edgecolor="none", alpha=0.75))
    text(ax, 0.410, 0.412, r"attention weights $\alpha_v$", size=6.2, color=COLORS["muted"])

    box(ax, 0.490, 0.420, 0.145, 0.100, "#FFFFFF", COLORS["grid"], radius=0.014)
    text(ax, 0.562, 0.480, r"$z_i=\sum_v \alpha_i^{(v)}h_i^{(v)}$", size=8.3)
    text(ax, 0.562, 0.448, r"$\hat{y}_i^{MV}=g_{\theta^\ast}(z_i)$", size=8.3)
    arrow(ax, (0.515, 0.565), (0.562, 0.520), color=COLORS["blue"])


def draw_panel_c(ax):
    text(ax, 0.685, 0.925, "C", size=10, weight="bold")
    text(ax, 0.730, 0.925, r"CeO$_2$/ZnO target prediction", size=8.2, weight="bold", ha="left")
    text(ax, 0.730, 0.895, "Anchor and calibrate AddH-out predictions", size=6.0, color=COLORS["muted"], ha="left")

    box(ax, 0.700, 0.742, 0.125, 0.098, "#FFFFFF", COLORS["grid"], radius=0.016)
    mini_lattice(ax, 0.722, 0.792, 1.25)
    ax.add_patch(Circle((0.865, 0.802), 0.013, facecolor=COLORS["green"], edgecolor="white", lw=0.5))
    ax.add_patch(Circle((0.865, 0.768), 0.013, facecolor=COLORS["pink"], edgecolor="white", lw=0.5))
    text(ax, 0.888, 0.802, r"CeO$_2$", size=6.6, ha="left")
    text(ax, 0.888, 0.768, "ZnO", size=6.6, ha="left")

    box(ax, 0.700, 0.605, 0.250, 0.072, "#FFFFFF", COLORS["grid"], radius=0.014)
    text(ax, 0.825, 0.641, r"$a_j=\hat{y}^{MV}_j+b(m_j,d_j,c_j)$", size=8.4)
    text(ax, 0.825, 0.617, "chemistry-guided anchor", size=6.2, color=COLORS["muted"])
    arrow(ax, (0.762, 0.742), (0.825, 0.677), color=COLORS["muted"])

    box(ax, 0.700, 0.472, 0.250, 0.082, COLORS["yellow_light"], "#B88C2D", radius=0.014)
    text(ax, 0.825, 0.519, r"$\tilde{y}_j=a_j+r_{\eta}(q_j)$", size=8.4)
    text(ax, 0.825, 0.494, r"$\eta^\ast=\arg\min_\eta \sum_{j\in C}(y_j-a_j-r_\eta(q_j))^2+\gamma\|\eta\|_2^2$", size=5.5)
    arrow(ax, (0.825, 0.605), (0.825, 0.554), color=COLORS["orange"])

    # Mini parity/ranking plot.
    box(ax, 0.700, 0.335, 0.250, 0.090, COLORS["green_light"], "#7AA985", radius=0.014)
    text(ax, 0.760, 0.394, "held-out validation", size=7.0, weight="bold", ha="left")
    text(ax, 0.760, 0.367, "MAE, RMSE, Pearson, Spearman", size=6.2, color=COLORS["muted"], ha="left")
    ax.plot([0.872, 0.930], [0.360, 0.405], color=COLORS["ink"], lw=0.75)
    for x, y, c in [(0.878, 0.362, COLORS["pink"]), (0.897, 0.382, COLORS["green"]), (0.925, 0.402, COLORS["green"])]:
        ax.add_patch(Circle((x, y), 0.006, facecolor=c, edgecolor="white", lw=0.25, zorder=5))
    arrow(ax, (0.825, 0.472), (0.825, 0.425), color=COLORS["green"])


def main() -> None:
    setup()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.3, 4.25), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.25, 1.0)
    ax.axis("off")

    # Soft panel backgrounds.
    box(ax, 0.035, 0.305, 0.285, 0.650, COLORS["gray_light"], "#E1E5EB", radius=0.018, lw=0.8)
    box(ax, 0.348, 0.305, 0.305, 0.650, COLORS["blue_light"], "#D6E5F2", radius=0.018, lw=0.8)
    box(ax, 0.678, 0.305, 0.287, 0.650, "#F5FAF7", "#D6E8DC", radius=0.018, lw=0.8)

    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)

    # Cross-panel arrows.
    arrow(ax, (0.270, 0.468), (0.410, 0.612), color=COLORS["blue"], lw=1.0, rad=-0.18)
    arrow(ax, (0.635, 0.470), (0.700, 0.640), color=COLORS["orange"], lw=1.0, dashed=True)

    text(
        ax,
        0.040,
        0.270,
        "Held-out performance is evaluated within repeated material-stratified AddH-out splits after frozen few-shot calibration.",
        size=5.8,
        color=COLORS["muted"],
        ha="left",
    )

    for ext in ["pdf", "svg", "png"]:
        path = OUT_DIR / f"fig1_top_journal_multiview_addhout.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
