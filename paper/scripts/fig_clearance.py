"""Planned versus executed clearance on the exclusion plan.

Reads only deposited data:
  data/clearance_tracks.npz -- executed tracks and references of the episodes
      shown (sigma = 0, seed 1000; sigma = 0.8, seed 1005), whose RMSE and
      minimum clearance are identical to data/mc_exclusions.json;
  data/mc_exclusions.json   -- minimum clearance of every closed-loop episode.
(a) the exclusion plan flown without disturbance; (b) detail where PID enters a
no-fly polygon at sigma = 0.8; (c) minimum distance from the executed track to
the raw polygons for every episode, against the planned 0.5 m clearance.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as Poly

HERE = Path(__file__).resolve().parents[1]
mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
OBST = [[(6, -4), (10, -4), (10, -9), (6, -9)], [(14, -1), (17, -1), (17, -4), (14, -4)]]
CLEAR = 0.5
STY = {"pid": ("PID", "#E69F00", (0, (4, 1.5))), "lqr": ("LQR", "#0072B2", (0, (1.5, 1))),
       "nmpc": ("NMPC", "#009E73", "-")}
T = np.load(HERE / "data/clearance_tracks.npz")
E = json.loads((HERE / "data/mc_exclusions.json").read_text())
E = E["episodes"] if isinstance(E, dict) and "episodes" in E else E


def obstacles(ax):
    for o in OBST:
        (x0, y0), (x1, y1) = o[0], o[2]
        ax.add_patch(Poly(o, closed=True, fc="#E6E6E6", ec="k", lw=0.6, hatch="////", zorder=1))
        ax.add_patch(plt.Rectangle((min(x0, x1) - CLEAR, min(y0, y1) - CLEAR),
                                   abs(x1 - x0) + 2 * CLEAR, abs(y1 - y0) + 2 * CLEAR,
                                   fill=False, ls="--", lw=0.6, ec="k"))


fig = plt.figure(figsize=(7.16, 2.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.25], wspace=0.32)
a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

# (a) whole plan, no disturbance
obstacles(a)
a.plot(*T["nmpc_s0_ref"].T, color="0.55", lw=1.6, alpha=0.6, label="reference")
for k, (lab, col, ls) in STY.items():
    a.plot(*T[f"{k}_s0_xy"].T, color=col, ls=ls, label=lab)
a.set_xlim(-0.8, 20.8); a.set_ylim(-14.6, 0.6); a.set_aspect("equal")
a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
a.set_title(r"(a) exclusion plan, $\sigma=0$", fontsize=7)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, handlelength=2.2, columnspacing=0.9)

# (b) detail around the deepest PID intrusion at sigma = 0.8
xy = T["pid_s08_1005_xy"]
def depth(x, y):  # penetration into a raw polygon (0 outside)
    return max(max(0.0, min(x - o[0][0], o[2][0] - x, y - o[2][1], o[0][1] - y)) for o in OBST)
d = np.array([depth(x, y) for x, y in xy]); i_max = int(d.argmax())
cx, cy = xy[i_max]
obstacles(b)
b.plot(*T["pid_s08_1005_ref"].T, color="0.55", lw=1.6, alpha=0.6)
for k, (lab, col, ls) in STY.items():
    b.plot(*T[f"{k}_s08_1005_xy"].T, color=col, lw=1.1, zorder=4,
           ls={"pid": (0, (2, 0.6)), "lqr": (0, (1, 0.6)), "nmpc": "-"}[k])
b.plot(cx, cy, "x", color="k", ms=4, mew=0.8, zorder=5)
b.set_xlim(cx - 0.75, cx + 0.75); b.set_ylim(cy - 0.75, cy + 0.75); b.set_aspect("equal")
b.set_xlabel("x [m]"); b.set_ylabel("y [m]")
b.set_title(rf"(b) $\sigma=0.8$, seed 1005: PID {1e2 * d.max():.0f} cm inside", fontsize=7)

# (c) minimum clearance of every episode (exact ordering)
sig = [0.0, 0.55, 0.8]
rng = np.random.default_rng(0)
for j, k in enumerate(STY):
    lab, col, _ = STY[k]
    for i, s in enumerate(sig):
        v = [e["min_clearance_m"] for e in E
             if e["controller"] == k and e["sigma"] == s and e.get("ordering") == "exact"]
        x = i + (j - 1) * 0.25 + rng.uniform(-0.05, 0.05, len(v))
        c.scatter(x, v, s=7, color=col, marker="os^"[j], lw=0.3, ec="k",
                  label=lab if i == 0 else None, zorder=3)
c.axhline(CLEAR, color="k", ls="--", lw=0.6)
c.axhline(0.0, color="k", lw=0.6)
c.text(-0.45, CLEAR + 0.01, "planned clearance", ha="left", va="bottom", fontsize=6)
c.text(-0.45, 0.01, "polygon boundary", ha="left", va="bottom", fontsize=6)
c.set_xticks(range(3), [r"$\sigma=0$", r"$\sigma=0.55$", r"$\sigma=0.8$"])
c.set_ylabel("min. distance to polygon [m]")
c.set_ylim(-0.03, 0.56); c.set_xlim(-0.5, 2.5)
c.set_title("(c) executed clearance, 10 seeds", fontsize=7)
fig.savefig(HERE / "figures/fig_clearance.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote figures/fig_clearance.pdf")
