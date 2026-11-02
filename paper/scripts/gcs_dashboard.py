#!/usr/bin/env python3
"""gcs_dashboard.py — Estación terrena (GCS) sobre los artefactos de la misión.

Lee lo que la misión YA exportó y lo presenta como lo vería un agrónomo: mapa
del campo con la traza volada, capas de índices, focos de plaga y zonas
marcadas para tratamiento, con sus coordenadas geodésicas.

No ejecuta la simulación ni el control: es la capa de visualización del
pipeline. Todo lo que muestra sale de:
    mission_artifacts/pest_detections.csv
    mission_artifacts/cwsi_grid.csv
    mission_artifacts/validation_report.json
    paper/data/spectral_field.npz  +  *_metrics.json

Uso:  streamlit run paper/scripts/gcs_dashboard.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
DATA = os.path.join(PAPER, "data")
ART = os.path.join(ROOT, "mission_artifacts")
EXTENT = (0.0, 20.0, -14.0, 0.0)
EARTH_R = 6378137.0


def local_to_geodetic(x, y, lat0, lon0):
    lat = lat0 + np.degrees(y / EARTH_R)
    lon = lon0 + np.degrees(x / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon


@st.cache_data
def load():
    det = pd.read_csv(os.path.join(ART, "pest_detections.csv"))
    grid = pd.read_csv(os.path.join(ART, "cwsi_grid.csv"))
    rep = json.load(open(os.path.join(ART, "validation_report.json")))
    met = json.load(open(os.path.join(DATA, "metrics.json")))
    spec = json.load(open(os.path.join(DATA, "spectral_metrics.json")))
    npz = np.load(os.path.join(DATA, "spectral_field.npz"))
    return det, grid, rep, met, spec, {k: npz[k] for k in npz.files}


def main() -> None:
    st.set_page_config(page_title="UAV crop-health GCS", layout="wide")
    det, grid, rep, met, spec, z = load()

    st.title("UAV crop-health ground control station")
    st.caption("Post-mission products of a hardware-free SIL run — "
               "demo plot 20 x 14 m, 7 m AGL, 2 m/s, OU wind sigma = 0.55")

    c = st.columns(5)
    c[0].metric("Tracking RMSE", f"{rep['rmse_pos_m']:.3f} m", "tol 0.75 m")
    c[1].metric("Coverage", f"{rep['coverage_pct']:.1f} %")
    c[2].metric("Pest detections", f"{int((det['class'].str.contains('pest')).sum())}",
                f"of {len(det)} obs.")
    c[3].metric("Mean CWSI", f"{grid['cwsi_mean'].mean():.2f}")
    c[4].metric("Flagged zones", f"{spec['zones_flagged']}",
                f"GT {spec['zones_ground_truth']}")

    st.sidebar.header("Layer")
    layer = st.sidebar.radio(
        "Raster", ["NDVI", "NDWI", "Health index", "Flagged zones", "None"])
    show_track = st.sidebar.checkbox("Flown track", True)
    show_pest = st.sidebar.checkbox("Pest detections", True)
    show_gt = st.sidebar.checkbox("Ground-truth foci", True)
    show_cwsi = st.sidebar.checkbox("CWSI grid", False)
    lat0 = st.sidebar.number_input("Reference latitude", value=-6.77, format="%.6f")
    lon0 = st.sidebar.number_input("Reference longitude", value=-79.84, format="%.6f")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(8, 5.6))
        if layer != "None":
            img, cmap, lim = {
                "NDVI": (z["ndvi"], "RdYlGn", (-0.1, 0.95)),
                "NDWI": (z["ndwi"], "BrBG", (-0.35, 0.45)),
                "Health index": (np.where(z["canopy"], z["health"], np.nan),
                                 "RdYlGn", (0.0, 1.0)),
                "Flagged zones": (np.where(z["flagged"], 1.0, np.nan),
                                  "autumn", (0.0, 1.0)),
            }[layer]
            im = ax.imshow(img, extent=EXTENT, cmap=cmap, vmin=lim[0],
                           vmax=lim[1], interpolation="nearest")
            if layer != "Flagged zones":
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        if show_cwsi:
            ax.scatter(grid["x_m"], grid["y_m"], c=grid["cwsi_mean"], s=110,
                       cmap="coolwarm", marker="s", alpha=0.65,
                       edgecolors="k", linewidths=0.3)
        if show_track:
            ax.plot(det["x_m"], det["y_m"], "-", color="#2e7d32", lw=0.7,
                    alpha=0.6, label="flown track")
        if show_pest:
            p = det[det["class"].str.contains("pest")]
            ax.plot(p["x_m"], p["y_m"], ".", color="#c0392b", ms=3,
                    label="pest detections")
        if show_gt:
            for cx, cy, r in met["ground_truth"]["pest_clusters"]:
                ax.add_patch(Circle((cx, cy), r, fill=False, ec="#1a5276",
                                    lw=1.2, ls="--"))
        ax.set_xlabel("x [m] (east)")
        ax.set_ylabel("y [m] (north)")
        ax.set_aspect("equal")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Detection performance")
        pest, ws = met["pest_detection"], met["water_stress"]
        px = spec["pixel_level"]
        st.dataframe(pd.DataFrame([
            {"layer": "Pest detector (RGB)", "precision": pest["precision"],
             "recall": pest["recall"], "F1": pest["f1"]},
            {"layer": "Water stress (CWSI)", "precision": ws["precision"],
             "recall": ws["recall"], "F1": ws["f1"]},
            {"layer": "Disease zones (NDVI+NDWI)", "precision": px["precision"],
             "recall": px["recall"], "F1": px["f1"]},
        ]), hide_index=True, use_container_width=True)

        st.subheader("Treatment zones (geodetic)")
        lab, n = ndimage.label(z["flagged"])
        gsd = spec["meta"]["gsd_m"]
        rows = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            cx = EXTENT[0] + xs.mean() * gsd
            cy = EXTENT[3] - ys.mean() * gsd
            lat, lon = local_to_geodetic(cx, cy, lat0, lon0)
            rows.append({"zone": i, "x [m]": round(cx, 2), "y [m]": round(cy, 2),
                         "lat": round(lat, 6), "lon": round(lon, 6),
                         "area [m2]": round(len(xs) * gsd ** 2, 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        st.download_button("Download zones as CSV",
                           pd.DataFrame(rows).to_csv(index=False),
                           "treatment_zones.csv", "text/csv")

        with st.expander("Index separability"):
            st.json(spec["index_stats"])


if __name__ == "__main__":
    main()
