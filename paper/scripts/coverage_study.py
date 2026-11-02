#!/usr/bin/env python3
"""coverage_study.py — Table I and the geometric sensitivity of the camera-ready.

Three parts, all with the corrected planner (touching spans merged, row links
routed, disconnected free areas refused):

  published  the paper's 20 x 14 m field: free, exclusions greedy, exclusions
             exact (Table I);
  shift      the same field with both obstacles shifted in y over +-1.5 m in
             1 cm steps: how much the exclusion cost depends on where the scan
             rows fall relative to the obstacle edges;
  random     200 random rectangular fields x 3 sweep directions with 1-3
             exclusions (some rotated): how the result generalizes.

Energy is the planning model of eval_metrics.py (power_model over the timed
reference, per-segment trapezoid, dt = 0.05 s). Every relative change is
computed per scenario against the exclusion-free field of the same scenario;
the summary reports medians and quantiles of those per-scenario changes and,
separately, the ratio of totals, which is a different quantity.

Usage: python coverage_study.py [--jobs 4] [--out ../data/coverage_study.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "mission_planner"))
sys.path.insert(0, HERE)

FIELD = [(0.0, 0.0), (20.0, 0.0), (20.0, -14.0), (0.0, -14.0)]
OBST = [[(6, -4), (10, -4), (10, -9), (6, -9)],
        [(14, -1), (17, -1), (17, -4), (14, -4)]]
DT = 0.05


def evaluate(field, obstacles, ordering="exact", angle=None):
    from mission_planner.coverage_planner import CoveragePlanner
    from shapely.geometry import LineString
    import power_model as pw

    class _Planner(CoveragePlanner):
        def _optimal_sweep_angle(self):
            return super()._optimal_sweep_angle() if angle is None else angle

    pl = _Planner(field, 7.0, 2.0, track_spacing_m=3.0, obstacles_xy=obstacles,
                  obstacle_clearance_m=0.5, cell_ordering=ordering)
    plan = pl.plan()
    samples = pl.time_parameterize(plan, dt=DT)
    xy = np.array([[t, x, y] for t, x, y, _, _ in samples])
    v = np.gradient(xy[:, 1:3], DT, axis=0)
    energy = float(pw.energy_from_speed_profile(np.hypot(v[:, 0], v[:, 1]), DT))
    w = plan.waypoints_xy
    inside = 0.0      # length of the plan inside any INFLATED obstacle
    for p, q in zip(w, w[1:]):
        if math.dist(p, q) > 1e-9:
            seg = LineString([p, q])
            inside += sum(seg.intersection(o.buffer(-1e-6)).length for o in pl.obstacles)
    m = plan.metrics
    return dict(cells=int(m.get("cells", 1)), path_m=float(m["path_length_m"]),
                transit_m=float(m.get("transit_length_m", 0.0)), time_s=float(xy[-1, 0]),
                free_area_m2=float(pl.poly.area), energy_wh=energy,
                energy_per_area_mwh_m2=1e3 * energy / pl.poly.area,
                coverage_field=float(m.get("coverage_field", np.nan)),
                coverage_accessible=float(m.get("coverage_accessible", np.nan)),
                inside_inflated_m=inside)


def _shift_job(dy):
    obs = [[(x, y + dy) for x, y in o] for o in OBST]
    return dict(dy=dy, exact=evaluate(FIELD, obs, "exact"), greedy=evaluate(FIELD, obs, "greedy"))


def _rect(cx, cy, w, h, th):
    c, s = math.cos(th), math.sin(th)
    return [(cx + c * x - s * y, cy + s * x + c * y)
            for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]


def random_instance(i, seed):
    """Field i of the random set: same generator as the audit, seeded per field."""
    from shapely.geometry import Polygon
    rng = np.random.default_rng([seed, i])
    W = rng.uniform(15, 60); H = W / rng.uniform(1, 3)
    field = [(0, 0), (W, 0), (W, -H), (0, -H)]
    obs = []
    for _ in range(int(rng.integers(1, 4))):
        for _try in range(50):
            w, h = rng.uniform(2, 0.25 * W), rng.uniform(2, 0.35 * H)
            cx, cy = rng.uniform(0.2 * W, 0.8 * W), -rng.uniform(0.2 * H, 0.8 * H)
            o = _rect(cx, cy, w, h, rng.uniform(0, math.pi) if rng.random() < 0.5 else 0.0)
            P = Polygon(o)
            if Polygon(field).buffer(-1.0).contains(P) and all(P.distance(Polygon(q)) > 1.5 for q in obs):
                obs.append(o)
                break
    return field, obs


def _random_job(args):
    i, seed, ang = args
    field, obs = random_instance(i, seed)
    try:
        return dict(i=i, angle_deg=round(math.degrees(ang)), n_obs=len(obs),
                    free=evaluate(field, None, "exact", ang),
                    exact=evaluate(field, obs, "exact", ang),
                    greedy=evaluate(field, obs, "greedy", ang))
    except Exception as exc:                     # a refused plan is data
        return dict(i=i, angle_deg=round(math.degrees(ang)), error=type(exc).__name__, msg=str(exc)[:160])


def q(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), p25=float(np.percentile(x, 25)), p75=float(np.percentile(x, 75)),
                p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                mean=float(x.mean()), min=float(x.min()), max=float(x.max()), n=int(x.size))


def changes(pairs):
    """Per-scenario relative changes (exclusions vs free) and ratios of totals."""
    dL = [e["path_m"] / f["path_m"] - 1 for f, e in pairs]
    dE = [e["energy_wh"] / f["energy_wh"] - 1 for f, e in pairs]
    dEa = [e["energy_per_area_mwh_m2"] / f["energy_per_area_mwh_m2"] - 1 for f, e in pairs]
    ratio = [a / b for a, b in zip(dE, dL) if abs(b) > 1e-3]
    return dict(dL=q(dL), dE=q(dE), dE_per_area=q(dEa), dE_over_dL=q(ratio),
                share_dL_negative=float(np.mean(np.array(dL) < 0)),
                ratio_of_totals_energy=float(sum(e["energy_wh"] for _, e in pairs)
                                             / sum(f["energy_wh"] for f, _ in pairs) - 1),
                ratio_of_totals_length=float(sum(e["path_m"] for _, e in pairs)
                                             / sum(f["path_m"] for f, _ in pairs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "coverage_study.json"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise FileExistsError(f"{a.out} exists; results are never overwritten")

    out = {"published": {"free": evaluate(FIELD, None, "exact"),
                         "exclusion_greedy": evaluate(FIELD, OBST, "greedy"),
                         "exclusion_exact": evaluate(FIELD, OBST, "exact")}}
    grid = [round(float(x), 2) for x in np.arange(-1.5, 1.5001, 0.01)]
    with Pool(a.jobs) as pool:
        shift = pool.map(_shift_job, grid)
        rnd = pool.map(_random_job, [(i, a.seed, ang) for i in range(a.n_random)
                                     for ang in (0.0, math.pi / 4, math.pi / 2)])
    free = out["published"]["free"]
    out["shift"] = dict(rows=shift,
                        summary=changes([(free, r["exact"]) for r in shift]),
                        greedy_over_exact=q([r["greedy"]["energy_wh"] / r["exact"]["energy_wh"] - 1 for r in shift]),
                        cells=sorted({r["exact"]["cells"] for r in shift}),
                        inside_inflated_plans=sum((r["exact"]["inside_inflated_m"] + r["greedy"]["inside_inflated_m"]) > 1e-6
                                                  for r in shift))
    ok = [r for r in rnd if "error" not in r]
    cov_ok = [r for r in ok if r["free"]["coverage_accessible"] >= 0.98 and r["exact"]["coverage_accessible"] >= 0.98]
    out["random"] = dict(rows=rnd, n_plans=len(rnd), n_refused=len(rnd) - len(ok),
                         summary=changes([(r["free"], r["exact"]) for r in ok]),
                         summary_coverage_matched=changes([(r["free"], r["exact"]) for r in cov_ok]),
                         n_coverage_matched=len(cov_ok),
                         greedy_over_exact=q([r["greedy"]["energy_wh"] / r["exact"]["energy_wh"] - 1 for r in ok]),
                         share_greedy_worse_1pct=float(np.mean([r["greedy"]["energy_wh"] / r["exact"]["energy_wh"] > 1.01 for r in ok])),
                         share_greedy_worse_5pct=float(np.mean([r["greedy"]["energy_wh"] / r["exact"]["energy_wh"] > 1.05 for r in ok])),
                         inside_inflated_plans=sum((r["exact"]["inside_inflated_m"] + r["greedy"]["inside_inflated_m"]) > 1e-6
                                                   for r in ok),
                         seed=a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    p = out["published"]
    print(json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in p.items()}, indent=1))
    for part in ("shift", "random"):
        s = out[part]["summary"]
        print(part, {k: round(s[k]["median"], 4) for k in ("dL", "dE", "dE_per_area", "dE_over_dL")},
              "ratio_of_totals E", round(s["ratio_of_totals_energy"], 4))


if __name__ == "__main__":
    main()
