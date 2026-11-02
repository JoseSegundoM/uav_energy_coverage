#!/usr/bin/env python3
"""eval_metrics.py — Métricas del paper a partir de artefactos YA generados.

NO simula nada. Lee los CSV georreferenciados que exportó la estación terrena
durante la misión SIL y los evalúa contra la verdad-terreno del gemelo digital.

La verdad-terreno se PARSEA del propio generador del mundo
(tools/generate_field_world.py) en vez de copiarse aquí: si alguien mueve un
foco de plaga en el mundo, las métricas siguen el cambio en lugar de mentir.

Salida: paper/data/metrics.json
"""
from __future__ import annotations

import ast
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORLD_GEN = os.path.join(ROOT, "tools", "generate_field_world.py")
ARTIFACTS = os.path.join(ROOT, "mission_artifacts")
OUT_DIR = os.path.join(os.path.dirname(HERE), "data")


# --------------------------------------------------------------- verdad-terreno
def literal_from_source(path: str, name: str):
    """Extrae una constante literal del fuente sin ejecutarlo."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} no encontrado en {path}")


def in_circles(x, y, circles) -> np.ndarray:
    m = np.zeros(len(x), bool)
    for cx, cy, r in circles:
        m |= ((x - cx) ** 2 + (y - cy) ** 2) <= r * r
    return m


def in_ellipses(x, y, ellipses) -> np.ndarray:
    m = np.zeros(len(x), bool)
    for cx, cy, rx, ry in ellipses:
        m |= (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1.0
    return m


# ------------------------------------------------------------------- métricas
def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def pr_curve(score: np.ndarray, y: np.ndarray, n: int = 60):
    """Curva P-R barriendo el umbral; devuelve puntos y AP (interp. de rectángulos)."""
    lo, hi = float(score.min()), float(score.max())
    taus = np.linspace(lo, hi, n)
    pts = []
    for t in taus:
        pred = score >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        d = prf(tp, fp, fn)
        pts.append((float(t), d["precision"], d["recall"], d["f1"]))
    # AP = suma de precision * delta(recall), con recall decreciente en tau
    rec = np.array([p[2] for p in pts])
    pre = np.array([p[1] for p in pts])
    order = np.argsort(rec)
    rec, pre = rec[order], pre[order]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * pre))
    best = max(pts, key=lambda p: p[3])
    return pts, ap, best


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pest_gt = literal_from_source(WORLD_GEN, "PEST_CLUSTERS")
    stress_gt = literal_from_source(WORLD_GEN, "STRESS_ZONES")

    out: Dict[str, object] = {
        "ground_truth": {"pest_clusters": pest_gt, "stress_zones": stress_gt},
    }

    # ---------------------------------------------------------- A) plagas
    det = pd.read_csv(os.path.join(ARTIFACTS, "pest_detections.csv"))
    x, y = det["x_m"].to_numpy(), det["y_m"].to_numpy()
    gt = in_circles(x, y, pest_gt)
    pred = det["class"].str.contains("pest").to_numpy()

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    pest = prf(tp, fp, fn)
    pest["tn"] = tn
    pest["n_detections"] = int(len(det))
    pest["accuracy"] = round((tp + tn) / len(det), 4)
    # curva P-R usando el score del detector sobre la clase "plaga"
    sc = np.where(pred, det["score"].to_numpy(), 0.0)
    pts, ap, best = pr_curve(sc, gt)
    pest["average_precision"] = round(ap, 4)
    pest["pr_curve"] = [[round(v, 4) for v in p] for p in pts]
    out["pest_detection"] = pest

    # --------------------------------------------------- B) estrés hídrico
    grid = pd.read_csv(os.path.join(ARTIFACTS, "cwsi_grid.csv"))
    gx, gy = grid["x_m"].to_numpy(), grid["y_m"].to_numpy()
    gt_s = in_ellipses(gx, gy, stress_gt)
    cw = grid["cwsi_mean"].to_numpy()
    pts_s, ap_s, best_s = pr_curve(cw, gt_s)
    tau = best_s[0]
    pred_s = cw >= tau
    stress = prf(int((pred_s & gt_s).sum()), int((pred_s & ~gt_s).sum()),
                 int((~pred_s & gt_s).sum()))
    stress.update({
        "n_cells": int(len(grid)),
        "best_threshold_cwsi": round(float(tau), 4),
        "average_precision": round(ap_s, 4),
        "cwsi_mean_inside_gt": round(float(cw[gt_s].mean()), 4) if gt_s.any() else None,
        "cwsi_mean_outside_gt": round(float(cw[~gt_s].mean()), 4) if (~gt_s).any() else None,
        "pr_curve": [[round(v, 4) for v in p] for p in pts_s],
    })
    out["water_stress"] = stress

    # ------------------------------------------------ C) cobertura y energía
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src", "mission_planner"))
    sys.path.insert(0, HERE)
    from mission_planner.coverage_planner import CoveragePlanner
    import power_model as pw

    field = [(0.0, 0.0), (20.0, 0.0), (20.0, -14.0), (0.0, -14.0)]
    obstacles = [[(6, -4), (10, -4), (10, -9), (6, -9)],
                 [(14, -1), (17, -1), (17, -4), (14, -4)]]
    cov = {"hover_power_w": round(pw.P_ELEC_HOVER, 2),
           "rotor_radius_eq_m": round(pw.ROTOR_R, 4),
           "cruise_power_w": round(float(pw.power_electrical(2.0)[0]), 2)}
    variants = (("free", None, "exact"),
                ("exclusion_greedy", obstacles, "greedy"),
                ("exclusion_exact", obstacles, "exact"))
    for tag, obs, order in variants:
        planner = CoveragePlanner(field, 7.0, 2.0, track_spacing_m=3.0,
                                  obstacles_xy=obs, obstacle_clearance_m=0.5,
                                  cell_ordering=order)
        pl = planner.plan()
        d = dict(pl.metrics)
        d["waypoints"] = len(pl.waypoints_xy)
        d["free_area_m2"] = round(planner.poly.area, 2)
        # energía con el modelo de vuelo de avance, integrando el perfil real
        samples = planner.time_parameterize(pl, dt=0.05)
        xy = np.array([[t, x, y] for t, x, y, _, _ in samples])
        v = np.gradient(xy[:, 1:3], 0.05, axis=0)
        speed = np.hypot(v[:, 0], v[:, 1])
        d["energy_fwd_wh"] = round(pw.energy_from_speed_profile(speed, 0.05), 4)
        d["energy_per_area_mwh_m2"] = round(
            1000 * d["energy_fwd_wh"] / planner.poly.area, 3)
        cov[tag] = d
    out["coverage"] = cov

    # -------------------------------------------- D) seguimiento (SIL previo)
    rep = os.path.join(ARTIFACTS, "validation_report.json")
    if os.path.exists(rep):
        out["tracking"] = json.load(open(rep, encoding="utf-8"))

    path = os.path.join(OUT_DIR, "metrics.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"-> {path}")
    print(f"plagas   P={pest['precision']:.3f} R={pest['recall']:.3f} "
          f"F1={pest['f1']:.3f} AP={pest['average_precision']:.3f} "
          f"(n={pest['n_detections']})")
    print(f"estres   P={stress['precision']:.3f} R={stress['recall']:.3f} "
          f"F1={stress['f1']:.3f} AP={stress['average_precision']:.3f} "
          f"tau={stress['best_threshold_cwsi']:.3f} (n={stress['n_cells']})")
    for tag, _, _ in (("free", 0, 0), ("exclusion_greedy", 0, 0),
                      ("exclusion_exact", 0, 0)):
        c = cov[tag]
        print(f"{tag:17s} celdas={c['cells']:.0f} L={c['path_length_m']:6.1f} m "
              f"transito={c.get('transit_length_m', 0):5.1f} "
              f"E={c['energy_fwd_wh']:.3f} Wh "
              f"({c['energy_per_area_mwh_m2']:.2f} mWh/m2) "
              f"campo={c['coverage_field']*100:.1f}% "
              f"acces={c['coverage_accessible']*100:.2f}%")


if __name__ == "__main__":
    main()
