"""
validate_coverage.py — Validación offline del planificador de cobertura.
Genera métricas + figura (sin ROS). Pensado para revisión del supervisor.

Uso:
    python3 validate_coverage.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coverage_planner import CoveragePlanner, Camera, geodetic_to_local


def main():
    # --- Campo de ejemplo (georreferenciado, escala agrícola) ---------------
    # Polígono rotado y no alineado a ejes para demostrar la dirección óptima.
    lat0, lon0 = -6.7700, -79.8400  # ref. local (zona arrocera, Lambayeque, Perú)
    field_latlon = [
        (-6.7690, -79.8415),
        (-6.7665, -79.8380),
        (-6.7700, -79.8345),
        (-6.7728, -79.8392),
    ]
    poly_xy = [geodetic_to_local(la, lo, lat0, lon0) for la, lo in field_latlon]

    # --- Cámara multiespectral típica de UAV agrícola -----------------------
    cam = Camera(focal_length_mm=8.0, sensor_width_mm=6.3, sensor_height_mm=4.7,
                 image_width_px=1600, image_height_px=1200)

    planner = CoveragePlanner(
        polygon_xy=poly_xy,
        altitude_m=80.0,
        speed_mps=6.0,
        camera=cam,
        side_overlap=0.70,
        forward_overlap=0.80,
        accel_mps2=2.0,
        yaw_mode="course",
    )
    plan = planner.plan()
    traj = planner.time_parameterize(plan, dt=0.1)

    print("=" * 60)
    print(" PLAN DE COBERTURA — métricas")
    print("=" * 60)
    print(f"  Dirección de barrido óptima : {math.degrees(plan.sweep_angle_rad):.1f} deg")
    for k, v in plan.metrics.items():
        print(f"  {k:22s}: {v}")
    print(f"  muestras de trayectoria (dt=0.1s): {len(traj)}")
    print("=" * 60)

    # --- Figura -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    px = [p[0] for p in poly_xy] + [poly_xy[0][0]]
    py = [p[1] for p in poly_xy] + [poly_xy[0][1]]
    ax.fill(px, py, color="#8a6d3b", alpha=0.20, zorder=0)
    ax.plot(px, py, color="#5a4327", lw=2.0, label="Límite del campo")

    wx = [w[0] for w in plan.waypoints_xy]
    wy = [w[1] for w in plan.waypoints_xy]
    ax.plot(wx, wy, color="#3ddc3d", lw=2.2, zorder=3,
            label="Trayectoria boustrophedon (zigzag)")
    ax.scatter([wx[0]], [wy[0]], c="blue", s=70, zorder=4, label="Inicio")
    ax.scatter([wx[-1]], [wy[-1]], c="red", s=70, zorder=4, label="Fin")

    ax.set_aspect("equal")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_title("Planificación de cobertura UAV — barrido óptimo\n"
                 f"{plan.metrics['field_area_ha']} ha · {int(plan.metrics['passes'])} pasadas · "
                 f"GSD {plan.metrics['gsd_m_per_px']} m/px · "
                 f"{plan.metrics['est_time_min']} min")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out = "coverage_validation.png"
    fig.savefig(out, dpi=150)
    print(f"Figura guardada: {out}")


if __name__ == "__main__":
    main()
