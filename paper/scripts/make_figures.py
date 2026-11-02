#!/usr/bin/env python3
"""make_figures.py — Figuras del paper, todas desde artefactos ya generados.

No simula nada: lee data/*.json y data/spectral_field.npz. Salida vectorial.

    fig1_architecture   diagrama de bloques del pipeline
    fig2_coverage       barrido: libre, exclusiones (voraz) y exclusiones (exacto)
    fig3_montecarlo     RMSE por controlador y nivel de viento (360 episodios)
    fig4_delay          margen de retardo de actuación
    fig5_spectral       NDVI / NDWI / mapa de salud / zonas marcadas
    fig6_spectral_stats dispersión del benchmark espectral sobre 30 ensayos
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon as MplPoly  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
DATA = os.path.join(PAPER, "data")
FIGS = os.path.join(PAPER, "figures")
sys.path.insert(0, os.path.join(ROOT, "src", "mission_planner"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType, not Type 3 (PDF eXpress)
})
C_PLAN, C_FLOWN, C_OBS = "#c0392b", "#2e7d32", "#5d6d7e"
C_CTRL = {"pid": "#8e44ad", "lqr": "#e08a1e", "nmpc": "#c0392b"}
COL1, COL2 = 3.45, 7.16

FIELD = [(0.0, 0.0), (20.0, 0.0), (20.0, -14.0), (0.0, -14.0)]
OBSTACLES = [[(6, -4), (10, -4), (10, -9), (6, -9)],
             [(14, -1), (17, -1), (17, -4), (14, -4)]]


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close(fig)
    print("->", name + ".pdf")


def jload(n):
    return json.load(open(os.path.join(DATA, n), encoding="utf-8"))


# ------------------------------------------------------------------- fig 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    ax.set_xlim(0, 106); ax.set_ylim(0, 36); ax.axis("off"); ax.grid(False)
    W, H = 19.0, 10.0
    XS = [1.0, 29.0, 57.0, 85.0]
    TOP, BOT = 23.0, 3.0
    for x, y, label, col in [
            (XS[0], TOP, "Coverage planner\ncellular decomp.\n+ exact ordering", "#eaf2f8"),
            (XS[1], TOP, "NMPC\nCasADi $\\rightarrow$ acados C\nROS 2 wrapper", "#e8f6ef"),
            (XS[2], TOP, "18-state plant\nRK4, 100 Hz (MC)\n1 kHz (delay sweep)", "#fdf2e2"),
            (XS[3], TOP, "Ground station\ngeoreferencing\nCSV artefacts", "#f4ecf7"),
            (XS[1], BOT, "OU wind +\nactuation delay", "#f2f4f4"),
            (XS[2], BOT, "Synthetic\nmultispectral bench.\nNDVI / NDWI", "#fdf2e2"),
            (XS[3], BOT, "Streamlit GCS\nhealth map +\nflagged zones", "#f4ecf7")]:
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.35",
                                    facecolor=col, edgecolor="#34495e", lw=0.7))
        ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", fontsize=6.4)
    lbl = dict(ha="center", va="center", fontsize=5.6, color="#34495e",
               bbox=dict(fc="white", ec="none", pad=0.6))

    def arrow(x0, y0, x1, y1, label="", ls="-"):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, ls=ls, color="#34495e"))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, **lbl)
    yc = TOP + H / 2
    arrow(XS[0] + W, yc, XS[1], yc, "timed\nreference")
    arrow(XS[1] + W, yc, XS[2], yc, "rotor\ncommands")
    arrow(XS[2] + W, yc, XS[3], yc, "detections\n+ CWSI")
    arrow(XS[2] + W / 2, TOP, XS[2] + W / 2, BOT + H)
    arrow(XS[2] + W, BOT + H / 2, XS[3], BOT + H / 2, "health\nmap")
    arrow(XS[1] + W / 2, BOT + H, XS[1] + W / 2, TOP, "disturbance", ":")
    ax.annotate("", (XS[1] + W / 2, TOP), (XS[2] + W / 2, TOP),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C_PLAN,
                                connectionstyle="arc3,rad=-0.55"))
    ax.text((XS[1] + XS[2] + W) / 2, TOP - 6.6, "state feedback   100 Hz",
            ha="center", fontsize=5.8, color=C_PLAN,
            bbox=dict(fc="white", ec="none", pad=0.6))
    save(fig, "fig1_architecture")


# ------------------------------------------------------------------- fig 2
def fig_coverage():
    """(a) free field, (b) exclusions with the corrected planner, (c) the
    exclusion cost against a rigid shift of both obstacles, for the corrected
    planner and for the submitted one (degenerate decomposition at dy = 0)."""
    from mission_planner.coverage_planner import CoveragePlanner
    fig = plt.figure(figsize=(COL2, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    for k, (obs, title) in enumerate([(None, "(a) no exclusion zones"),
                                      (OBSTACLES, "(b) two exclusion zones")]):
        ax = fig.add_subplot(gs[0, k])
        planner = CoveragePlanner(FIELD, 7.0, 2.0, track_spacing_m=3.0,
                                  obstacles_xy=obs, obstacle_clearance_m=0.5,
                                  cell_ordering="exact")
        pl = planner.plan()
        w = np.array(pl.waypoints_xy)
        ax.add_patch(MplPoly(FIELD, closed=True, facecolor="#f7f9f9",
                             edgecolor="#34495e", lw=0.8, zorder=0))
        for o in planner.obstacles:      # inflated (no-fly + clearance)
            ax.add_patch(MplPoly(list(o.exterior.coords), closed=True, fill=False,
                                 edgecolor="#2c3e50", lw=0.5, ls="--", zorder=1))
        for o in (obs or []):
            ax.add_patch(MplPoly(o, closed=True, facecolor=C_OBS, alpha=0.55,
                                 edgecolor="#2c3e50", lw=0.7, hatch="///", zorder=1))
        ax.plot(w[:, 0], w[:, 1], color=C_PLAN, lw=0.9, zorder=3)
        ax.plot(w[0, 0], w[0, 1], "o", color="#1a5276", ms=4, zorder=5)
        m = pl.metrics
        ax.set_title(f"{title}\ncells {m['cells']:.0f} · L {m['path_length_m']:.1f} m"
                     f" · E {m['energy_wh']:.2f} Wh", fontsize=7)
        ax.set_xlabel("x [m]"); ax.set_aspect("equal")
        if k == 0:
            ax.set_ylabel("y [m]")
    ax = fig.add_subplot(gs[0, 2])
    fixed = jload("coverage_study.json")["shift"]["rows"]
    sub = sorted(jload("coverage_shift_submitted_planner.json"), key=lambda r: r["dy"])
    dy = [r["dy"] for r in fixed]
    free = jload("coverage_study.json")["published"]["free"]
    ax.plot(dy, [100 * (r["exact"]["energy_wh"] / free["energy_wh"] - 1) for r in fixed],
            color=C_PLAN, lw=1.0, label="energy")
    ax.plot(dy, [100 * (r["exact"]["path_m"] / free["path_m"] - 1) for r in fixed],
            color="#1a5276", lw=1.0, label="path length")
    ax.plot([r["dy"] for r in sub], [100 * r["dE"] for r in sub], color="#7f8c8d",
            lw=0.9, ls="--", label="energy, no span merging", zorder=1)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("obstacle shift $\\Delta y$ [m]")
    ax.set_ylabel("change vs. no exclusions [%]")
    ax.set_title("(c) sensitivity to obstacle position", fontsize=7)
    ax.legend(loc="upper left", fontsize=5.6, frameon=False)
    save(fig, "fig2_coverage")


# ------------------------------------------------------------------- fig 3
def fig_montecarlo():
    R = jload("mc_main.json")
    sigmas = sorted({r["sigma"] for r in R})
    ctrls = ["pid", "lqr", "nmpc"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.5),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    width = 0.24
    for i, c in enumerate(ctrls):
        data = [[r["rmse_m"] for r in R
                 if r["controller"] == c and abs(r["sigma"] - s) < 1e-9]
                for s in sigmas]
        pos = np.arange(len(sigmas)) + (i - 1) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=0.9))
        for box in bp["boxes"]:
            box.set(facecolor=C_CTRL[c], alpha=0.75, lw=0.6)
        ax.plot([], [], color=C_CTRL[c], lw=4, alpha=0.75, label=c.upper())
    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlabel(r"wind $\sigma$  [N$\cdot$s$^{-1/2}$]")
    ax.set_ylabel("tracking RMSE [m]")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title(f"{len(R)} episodes, 30 seeds per cell", fontsize=7.5)

    # panel derecho: mejora relativa del NMPC frente al mejor baseline
    imp = []
    for s in sigmas:
        best = min(np.mean([r["rmse_m"] for r in R if r["controller"] == c
                            and abs(r["sigma"] - s) < 1e-9]) for c in ("pid", "lqr"))
        nm = np.mean([r["rmse_m"] for r in R if r["controller"] == "nmpc"
                      and abs(r["sigma"] - s) < 1e-9])
        imp.append(100 * (1 - nm / best))
    ax2.bar(range(len(sigmas)), imp, color=C_CTRL["nmpc"], alpha=0.8, width=0.6)
    for i, v in enumerate(imp):
        ax2.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=6.5)
    ax2.set_xticks(range(len(sigmas)))
    ax2.set_xticklabels([f"{s:g}" for s in sigmas])
    ax2.set_xlabel(r"wind $\sigma$")
    ax2.set_ylabel("RMSE reduction vs\nbest linear baseline [%]")
    ax2.set_ylim(0, 105)
    save(fig, "fig3_montecarlo")


# ------------------------------------------------------------------- fig 4
def fig_delay():
    R = jload("mc_delay.json")
    ds = sorted({r["delay_ms"] for r in R})
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    mean, std, div = [], [], []
    for d in ds:
        g = [r for r in R if abs(r["delay_ms"] - d) < 1e-9]
        ok = [r["rmse_m"] for r in g if not r["diverged"]]
        mean.append(np.mean(ok) if ok else np.nan)
        std.append(np.std(ok, ddof=1) if len(ok) > 1 else 0.0)
        div.append(100 * sum(r["diverged"] for r in g) / len(g))
    ax.errorbar(ds, mean, yerr=std, marker="o", ms=4, lw=1.2,
                color=C_CTRL["nmpc"], capsize=2, label="RMSE (stable runs)")
    ax.set_xlabel("actuation delay [ms]")
    ax.set_ylabel("tracking RMSE [m]", color=C_CTRL["nmpc"])
    ax.tick_params(axis="y", labelcolor=C_CTRL["nmpc"])
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.bar(ds, div, width=0.7, color="#2c3e50", alpha=0.25, label="divergence")
    ax2.set_ylabel("diverged episodes [%]", color="#2c3e50")
    ax2.set_ylim(0, 105)
    ax.set_title(r"$\sigma=0.55$, plant integrated at 1 kHz, 10 seeds per delay", fontsize=7.5)
    save(fig, "fig4_delay")


# ------------------------------------------------------------------- fig 5
def fig_spectral():
    z = np.load(os.path.join(DATA, "spectral_field.npz"))
    sm = jload("spectral_metrics.json")
    ext = [0, 20, -14, 0]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.85))
    panels = [(z["ndvi"], "NDVI", "RdYlGn", (-0.1, 0.95)),
              (z["ndwi"], "NDWI", "BrBG", (-0.35, 0.45)),
              (np.where(z["canopy"], z["health"], np.nan), "health index",
               "RdYlGn", (0, 1)),
              (None, "flagged zones", None, None)]
    for ax, (img, title, cmap, lim) in zip(axes, panels):
        if img is not None:
            im = ax.imshow(img, extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1],
                           interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=5.5)
        else:
            ax.imshow(np.where(z["canopy"], 0.85, 0.35), extent=ext,
                      cmap="Greys_r", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~z["flagged"], z["flagged"]), extent=ext,
                      cmap=matplotlib.colors.ListedColormap(["#c0392b"]),
                      alpha=0.75, interpolation="nearest")
            for cx, cy, r in sm["meta"]["pest_clusters"]:
                ax.add_patch(Circle((cx, cy), r, fill=False, ec="#1a5276",
                                    lw=0.9, ls="--"))
        ax.set_title(title); ax.set_xlabel("x [m]"); ax.grid(False)
        ax.tick_params(labelsize=5.5)
    axes[0].set_ylabel("y [m]")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    save(fig, "fig5_spectral")


# ------------------------------------------------------------------- fig 6
def fig_spectral_stats():
    t = jload("spectral_trials.json")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.2))
    vals = [[r["precision"] for r in t["trials"]],
            [r["recall"] for r in t["trials"]],
            [r["f1"] for r in t["trials"]]]
    bp = ax.boxplot(vals, labels=["precision", "recall", "$F_1$"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=0.9))
    for b, c in zip(bp["boxes"], ["#2980b9", "#e08a1e", C_FLOWN]):
        b.set(facecolor=c, alpha=0.7, lw=0.6)
    for i, v in enumerate(vals):
        ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(3).normal(0, 0.05, len(v)),
                   v, s=5, color="#2c3e50", alpha=0.45, zorder=3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title(f"{t['n_trials']} randomized test trials", fontsize=7.5)

    c = np.array(t["calibration_curve"])
    ax2.plot(c[:, 0], c[:, 1], "-", color=C_FLOWN, lw=1.2)
    ax2.axvline(t["threshold"], color=C_PLAN, ls="--", lw=1.0,
                label=f"selected $\\tau$ = {t['threshold']:.2f}")
    ax2.set_xlabel(r"health-index threshold $\tau$")
    ax2.set_ylabel("mean $F_1$ (calibration set)")
    ax2.legend(frameon=False, loc="lower center")
    ax2.set_title(f"threshold calibrated on {t['n_calib']} disjoint trials",
                  fontsize=7.5)
    save(fig, "fig6_spectral_stats")


# ------------------------------------------------------------------- fig 7
def fig_mission():
    """El gemelo digital y la percepción, con imagen real de la misión SIL.

    Los paneles se RECORTAN del cuadro compuesto que grabó mission_recorder_node
    durante la corrida en Gazebo, cuyo trazado es fijo y conocido (1920x1080,
    cabecera de 64 px, columna izquierda de 1280 px, vista de persecución de
    720 px de alto). No se re-renderiza nada: es la misma imagen que produjo la
    misión.
    """
    import cv2
    import pandas as pd

    src = os.path.join(ROOT, "assets", "mission", "mission_sweep.png")
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    HEAD, COL, CHASE_H = 64, 1280, 720
    MID = HEAD + (1080 - HEAD) // 2
    # se saltan 58 px: es el rótulo que el grabador superpone en vivo
    chase = img[HEAD + 58:HEAD + CHASE_H - 1, 1:COL - 1]
    pest = img[HEAD + 29:MID - 1, COL + 1:1920 - 1]
    cwsi = img[MID + 29:1080 - 1, COL + 1:1920 - 1]

    fig = plt.figure(figsize=(COL2, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.45, wspace=0.30)

    axA = fig.add_subplot(gs[0, :2])
    axA.imshow(chase); axA.axis("off")
    axA.set_title("(a) digital twin, chase view over the row crop", fontsize=7.5)

    axD = fig.add_subplot(gs[0, 2])
    det = pd.read_csv(os.path.join(ROOT, "mission_artifacts", "pest_detections.csv"))
    grid = pd.read_csv(os.path.join(ROOT, "mission_artifacts", "cwsi_grid.csv"))
    met = jload("metrics.json")
    axD.add_patch(MplPoly(FIELD, closed=True, facecolor="#f7f9f9",
                          edgecolor="#34495e", lw=0.7, zorder=0))
    sc = axD.scatter(grid["x_m"], grid["y_m"], c=grid["cwsi_mean"], s=26,
                     cmap="coolwarm", marker="s", alpha=0.85, zorder=1,
                     edgecolors="none")
    axD.plot(det["x_m"], det["y_m"], "-", color=C_FLOWN, lw=0.35, alpha=0.5, zorder=2)
    pe = det[det["class"].str.contains("pest")]
    axD.plot(pe["x_m"], pe["y_m"], ".", color=C_PLAN, ms=1.6, zorder=3)
    for cx, cy, r in met["ground_truth"]["pest_clusters"]:
        axD.add_patch(Circle((cx, cy), r, fill=False, ec="#1a5276", lw=0.8, ls="--",
                             zorder=4))
    cb = fig.colorbar(sc, ax=axD, fraction=0.046, pad=0.03)
    cb.set_label("CWSI", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    axD.set_aspect("equal"); axD.tick_params(labelsize=5.5)
    axD.set_xlabel("x [m]", fontsize=6.5, labelpad=1)
    axD.set_ylabel("y [m]", fontsize=6.5, labelpad=1)
    axD.set_title("(d) georeferenced products", fontsize=7.5)

    axB = fig.add_subplot(gs[1, 0]); axB.imshow(pest); axB.axis("off")
    axB.set_title("(b) nadir RGB + pest detections", fontsize=7.5)
    axC = fig.add_subplot(gs[1, 1]); axC.imshow(cwsi); axC.axis("off")
    axC.set_title("(c) CWSI from the thermal channel", fontsize=7.5)

    axE = fig.add_subplot(gs[1, 2])
    axE.hist(grid["cwsi_mean"], bins=16, color="#2980b9", alpha=0.8,
             edgecolor="white", lw=0.4)
    axE.axvline(met["water_stress"]["best_threshold_cwsi"], color=C_PLAN,
                ls="--", lw=1.0)
    axE.set_xlabel("cell mean CWSI", fontsize=6.5, labelpad=1)
    axE.set_ylabel("cells", fontsize=6.5, labelpad=1)
    axE.tick_params(labelsize=5.5)
    axE.set_title("(e) CWSI distribution", fontsize=7.5)
    save(fig, "fig7_mission")


def main():
    os.makedirs(FIGS, exist_ok=True)
    only = set(sys.argv[1:])           # e.g. make_figures.py fig_coverage
    if only:
        for name in only:
            globals()[name]()
        return
    fig_architecture()
    fig_mission()
    fig_coverage()
    fig_montecarlo()
    fig_spectral()
    fig_spectral_stats()
    if os.path.exists(os.path.join(DATA, "mc_delay.json")):
        fig_delay()


if __name__ == "__main__":
    main()
