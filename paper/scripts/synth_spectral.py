#!/usr/bin/env python3
"""synth_spectral.py — Campo multiespectral sintético procedural y su análisis.

Genera un mosaico de reflectancia de 5 bandas sobre la MISMA parcela del gemelo
digital (20 x 14 m), con surcos de papa, suelo desnudo y parches enfermos
situados en los mismos focos que el mundo de Gazebo. De ahí calcula NDVI y NDWI,
fusiona ambos en un mapa de salud y marca zonas sospechosas.

Por qué sintético y no un dataset real: mantiene el testbed autocontenido y
reproducible, y permite evaluar el pipeline contra una verdad-terreno EXACTA
(la máscara que generó los parches). Lo que mide es la cadena
índices -> mapa de salud -> zonas marcadas, no la capacidad de generalizar a
hojas de papa reales; eso se declara como limitación en el paper.

Firmas espectrales: se eligen dentro de rangos publicados para dosel sano,
dosel infectado y suelo. El comportamiento cualitativo que reproducen es el
descrito en la literatura de tizón tardío: el dosel infectado sube su
reflectancia en el visible y la baja en el NIR.

Salida: data/spectral_field.npz y data/spectral_metrics.json
"""
from __future__ import annotations

import ast
import json
import os
from typing import Dict, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
WORLD_GEN = os.path.join(ROOT, "tools", "generate_field_world.py")
OUT_DIR = os.path.join(PAPER, "data")

# parcela y muestreo
X0, X1, Y0, Y1 = 0.0, 20.0, -14.0, 0.0
GSD = 0.05                      # m/px -> 400 x 280
ROW_SPACING = 0.75              # m entre surcos de papa
ROW_WIDTH = 0.45                # m de dosel por surco

# Reflectancia media por clase: (green, red, red_edge, nir, swir1240)
SIG = {
    "soil":     np.array([0.20, 0.25, 0.28, 0.30, 0.35]),
    "healthy":  np.array([0.08, 0.04, 0.28, 0.50, 0.30]),
    "diseased": np.array([0.14, 0.12, 0.26, 0.30, 0.33]),
}
SIGMA_REL = 0.06                # ruido multiplicativo del sensor
BANDS = ("green", "red", "red_edge", "nir", "swir")


def literal_from_source(path: str, name: str):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(name)


def sample_config(seed: int) -> Dict:
    """Configuración ALEATORIA e independiente del mundo de Gazebo.

    Motivo (fuga conceptual): en la primera versión los focos de enfermedad se
    colocaban en las mismas coordenadas que los del gemelo, de modo que la
    verdad-terreno y el proceso que genera la señal compartían origen Y una
    única configuración fija. Un revisor puede objetar, con razón, que el
    benchmark está construido para funcionar. Aquí cada ensayo sortea su propia
    geometría, iluminación y ruido, y la evaluación se hace sobre la
    distribución de ensayos, no sobre un caso elegido a mano.
    """
    rng = np.random.default_rng(10_000 + seed)
    n_foci = int(rng.integers(2, 6))
    foci = []
    for _ in range(n_foci):
        foci.append((float(rng.uniform(2.0, 18.0)),
                     float(rng.uniform(-12.0, -2.0)),
                     float(rng.uniform(1.0, 2.4))))
    return {
        "seed": seed,
        "pest_clusters": foci,
        "row_spacing_m": float(rng.uniform(0.65, 0.90)),
        "row_width_m": float(rng.uniform(0.35, 0.55)),
        "sensor_noise_rel": float(rng.uniform(0.04, 0.10)),
        "illum_gain": float(rng.uniform(0.06, 0.22)),
        "illum_axis": int(rng.integers(0, 2)),
        "severity_exp": float(rng.uniform(1.0, 2.5)),
    }


def build_field(seed: int = 7, cfg: Dict | None = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    cfg = cfg or sample_config(seed)
    # RNG del SENSOR, separado del que sorteó la geometría: el ruido de medida
    # no debe estar correlacionado con dónde está la enfermedad.
    rng = np.random.default_rng(500_000 + seed)
    xs = np.arange(X0, X1, GSD)
    ys = np.arange(Y1, Y0, -GSD)          # de y=0 hacia y=-14
    X, Y = np.meshgrid(xs, ys)

    # --- surcos: dosel donde el resto del avance transversal cae en la banda
    row_sp, row_w = cfg["row_spacing_m"], cfg["row_width_m"]
    phase = np.mod(X, row_sp)
    canopy = phase < row_w
    # los surcos no son perfectos: borde irregular y algunos claros
    canopy &= rng.random(X.shape) > 0.06
    canopy &= (np.abs(np.sin(Y * 1.7)) < 0.985)

    # --- parches enfermos en los focos del gemelo digital
    clusters = cfg["pest_clusters"]
    diseased = np.zeros(X.shape, bool)
    for cx, cy, r in clusters:
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # borde difuso: la severidad decae con el radio, no es un disco duro
        p = np.clip(1.0 - (d / r) ** cfg["severity_exp"], 0.0, 1.0)
        diseased |= rng.random(X.shape) < p
    diseased &= canopy                     # solo hay enfermedad donde hay planta

    # --- mosaico de reflectancia
    cube = np.empty(X.shape + (len(BANDS),), np.float32)
    for k in range(len(BANDS)):
        base = np.where(canopy, SIG["healthy"][k], SIG["soil"][k])
        base = np.where(diseased, SIG["diseased"][k], base)
        cube[..., k] = base
    # ruido del sensor + gradiente de iluminación (sol oblicuo sobre la parcela)
    cube *= (1.0 + rng.normal(0.0, cfg["sensor_noise_rel"],
                              cube.shape)).astype(np.float32)
    g = cfg["illum_gain"]
    axis = (X - X0) / (X1 - X0) if cfg["illum_axis"] == 0 else (Y - Y0) / (Y1 - Y0)
    illum = (1.0 - g / 2 + g * axis).astype(np.float32)
    cube *= illum[..., None]
    np.clip(cube, 1e-4, 1.5, out=cube)

    meta = dict(cfg, gsd_m=GSD, shape=list(cube.shape),
                bands=list(BANDS))
    return cube, canopy, diseased, meta


def indices(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g, r, re, nir, swir = (cube[..., i] for i in range(5))
    ndvi = (nir - r) / (nir + r)           # Rouse et al. 1974
    ndwi = (nir - swir) / (nir + swir)     # Gao 1996
    return ndvi, ndwi


def health_map(ndvi: np.ndarray, ndwi: np.ndarray, canopy: np.ndarray
               ) -> np.ndarray:
    """Fusión de ambos índices en [0,1]: 1 = sano, 0 = severamente afectado.

    Se normaliza cada índice por sus percentiles DENTRO del dosel: el suelo
    tiene NDVI bajo por definición y arrastraría la escala si entrase.
    """
    def norm(a):
        lo, hi = np.percentile(a[canopy], [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    h = 0.65 * norm(ndvi) + 0.35 * norm(ndwi)
    return np.where(canopy, h, np.nan)


def flag_zones(health: np.ndarray, canopy: np.ndarray, tau: float,
               min_px: int = 400) -> Tuple[np.ndarray, int]:
    """Marca zonas sospechosas: umbral + cierre morfológico + componentes.

    El cierre elimina el moteado de píxeles sueltos, que a la escala del
    tratamiento agronómico no es accionable: lo que importa son focos.
    """
    from scipy import ndimage
    m = canopy & (health < tau)
    m = ndimage.binary_closing(m, np.ones((7, 7)))
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    lab, n = ndimage.label(m)
    out = np.zeros_like(m)
    kept = 0
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() >= min_px:
            out |= c
            kept += 1
    return out, kept


def prf(pred, gt) -> Dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f, 4)}


def gt_zone_mask(cfg, canopy):
    """Verdad-terreno de ÁREA: la zona a tratar, no la infección planta a planta.

    `diseased` es moteada: es lo que genera la reflectancia. Pero el producto
    accionable es la zona que un fumigador cubre entera, así que evaluar una
    salida rellena contra una máscara moteada penalizaría como falso positivo
    cada planta sana rodeada de enfermas, que es precisamente la que hay que
    tratar.
    """
    xs = np.arange(X0, X1, GSD)
    ys = np.arange(Y1, Y0, -GSD)
    X, Y = np.meshgrid(xs, ys)
    gt = np.zeros(X.shape, bool)
    for cx, cy, r in cfg["pest_clusters"]:
        gt |= ((X - cx) ** 2 + (Y - cy) ** 2) <= r * r
    return gt & canopy


def run_trial(seed: int, tau: float | None = None):
    """Un ensayo completo. Si tau es None, devuelve la curva para calibrar."""
    cfg = sample_config(seed)
    cube, canopy, diseased, meta = build_field(seed, cfg)
    ndvi, ndwi = indices(cube)
    health = health_map(ndvi, ndwi, canopy)
    gt = gt_zone_mask(meta, canopy)

    stats = {
        "ndvi_healthy": float(ndvi[canopy & ~diseased].mean()),
        "ndvi_diseased": float(ndvi[diseased].mean()) if diseased.any() else float("nan"),
        "ndvi_soil": float(ndvi[~canopy].mean()),
        "ndwi_healthy": float(ndwi[canopy & ~diseased].mean()),
        "ndwi_diseased": float(ndwi[diseased].mean()) if diseased.any() else float("nan"),
    }
    if tau is None:
        curve = []
        for t in np.linspace(0.05, 0.95, 37):
            pred, _ = flag_zones(health, canopy, float(t))
            curve.append((float(t), prf(pred, gt)["f1"]))
        return curve
    pred, n_zones = flag_zones(health, canopy, tau)
    d = prf(pred, gt)
    d.update(seed=seed, threshold=tau, zones_flagged=n_zones,
             zones_gt=len(meta["pest_clusters"]),
             area_gt_m2=round(float(gt.sum()) * GSD ** 2, 2),
             area_pred_m2=round(float(pred.sum()) * GSD ** 2, 2), **stats)
    return d, (ndvi, ndwi, health, canopy, diseased, pred, gt, meta)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=10, help="ensayos de calibración")
    ap.add_argument("--trials", type=int, default=30, help="ensayos de test")
    ap.add_argument("--figure-seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1) CALIBRACIÓN del umbral en semillas propias -----------------------
    # Elegir tau maximizando F1 sobre los mismos datos con los que luego se
    # reporta el resultado es ajustar sobre el test. Los conjuntos son disjuntos.
    curves = [run_trial(s) for s in range(a.calib)]
    taus = [t for t, _ in curves[0]]
    mean_f1 = [float(np.mean([c[i][1] for c in curves])) for i in range(len(taus))]
    tau = float(taus[int(np.argmax(mean_f1))])
    print(f"tau calibrado = {tau:.3f}  (F1 medio {max(mean_f1):.3f} "
          f"sobre {a.calib} ensayos de calibración)")

    # --- 2) TEST en semillas disjuntas --------------------------------------
    rows = []
    for s in range(1000, 1000 + a.trials):
        d, _ = run_trial(s, tau)
        rows.append(d)

    def agg(k):
        v = np.array([r[k] for r in rows], float)
        return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4),
                "p05": round(float(np.percentile(v, 5)), 4),
                "p95": round(float(np.percentile(v, 95)), 4)}

    detected = float(np.mean([min(r["zones_flagged"], r["zones_gt"]) / r["zones_gt"]
                              for r in rows]))
    summary = {
        "threshold": tau, "n_calib": a.calib, "n_trials": a.trials,
        "precision": agg("precision"), "recall": agg("recall"), "f1": agg("f1"),
        "zone_detection_rate": round(detected, 4),
        "ndvi_healthy": agg("ndvi_healthy"), "ndvi_diseased": agg("ndvi_diseased"),
        "ndvi_soil": agg("ndvi_soil"),
        "ndwi_healthy": agg("ndwi_healthy"), "ndwi_diseased": agg("ndwi_diseased"),
        "calibration_curve": [[round(t, 3), round(f, 4)]
                              for t, f in zip(taus, mean_f1)],
        "trials": rows,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "spectral_trials.json"), "w"),
              indent=1)

    # --- 3) figura: un ensayo canónico, al mismo umbral calibrado -----------
    d, arrays = run_trial(a.figure_seed, tau)
    ndvi, ndwi, health, canopy, diseased, pred, gt, meta = arrays
    np.savez_compressed(
        os.path.join(OUT_DIR, "spectral_field.npz"),
        ndvi=ndvi.astype(np.float32), ndwi=ndwi.astype(np.float32),
        health=health.astype(np.float32), canopy=canopy, diseased=diseased,
        flagged=pred, gt_zone=gt)
    json.dump({"meta": meta, "figure_trial": d, "summary": {
        k: summary[k] for k in ("threshold", "precision", "recall", "f1",
                                "zone_detection_rate", "n_trials")}},
        open(os.path.join(OUT_DIR, "spectral_metrics.json"), "w"), indent=1)

    print(f"NDVI sano={summary['ndvi_healthy']['mean']:.3f} "
          f"enfermo={summary['ndvi_diseased']['mean']:.3f} "
          f"suelo={summary['ndvi_soil']['mean']:.3f}")
    print(f"test n={a.trials}:  P={summary['precision']['mean']:.3f}"
          f"±{summary['precision']['std']:.3f}  "
          f"R={summary['recall']['mean']:.3f}±{summary['recall']['std']:.3f}  "
          f"F1={summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f}")
    print(f"tasa de deteccion de focos = {detected*100:.1f}%")


if __name__ == "__main__":
    main()
