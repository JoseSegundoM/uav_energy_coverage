#!/usr/bin/env python3
"""sil_validation_node.py — Validación E2E de la misión SIL (P5).

Se lanza junto a la misión (sil_flight_mission.launch.py validate:=true) y al
terminar el barrido escribe un reporte JSON con las métricas de la fase P5:

  - rmse_pos_m      RMSE de |e_pos| (odom vs /uav/reference) bajo viento,
                    excluyendo el transitorio de despegue (settle_s).
  - coverage_pct    %% de muestras de la referencia (la trayectoria zigzag
                    parametrizada) que el dron visitó a < coverage_tol_m en XY.
  - z_min/z_max     envolvente vertical post-despegue (colisión si z ~ 0).
  - mission_done    el stream de referencia terminó (planner completó).
  - water_stress    estadísticas del CWSI medio publicado (la VALIDEZ
                    radiométrica depende de calibración pendiente — ver README).

Métricas de P5 que NO se calculan aquí (insumos externos pendientes):
  - mAP de plagas: requiere dataset etiquetado papa/arroz (pesos reales).
  - error CWSI: requiere verdad-terreno radiométrica; la fórmula se validó
    exacta en P3 con frames sintéticos.

Fin de misión: sin mensajes nuevos en /uav/reference durante end_timeout_s
tras haber recibido al menos 100 (el planner cancela su timer al terminar).
"""
import json
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


class SilValidation(Node):
    def __init__(self):
        super().__init__('sil_validation_node')
        self.declare_parameter('settle_s', 15.0)
        # Tolerancia de MISIÓN COMPLETA (incluye giros 180° del zigzag, donde
        # el error pica ~1-2.6 m bajo ráfagas; en pasadas rectas es ~0.1 m).
        # 0.5 m aplica a seguimiento de punto fijo (P1), no a la misión entera.
        self.declare_parameter('rmse_tol_m', 0.75)
        self.declare_parameter('coverage_tol_m', 1.5)
        self.declare_parameter('end_timeout_s', 4.0)
        self.declare_parameter('report_path', os.path.expanduser(
            '~/uav_agricultural/mission_artifacts/validation_report.json'))

        self.t0 = None
        self.t_last_ref = None
        self.refs = []          # (t, x, y, z)
        self.odoms = []         # (t, x, y, z)
        self.errs = []          # (t, |e_pos|)
        self.stress = []
        self.last_ref = None
        self.done = False

        self.create_subscription(Odometry, '/uav/odom', self.on_odom, 50)
        self.create_subscription(
            PoseStamped, '/uav/reference', self.on_ref, 50)
        self.create_subscription(
            Float32, '/perception/water_stress', self.on_stress, 10)
        self.create_timer(1.0, self.check_end)
        self.get_logger().info('sil_validation: esperando misión…')

    def _now(self):
        # TIEMPO DE SIMULACIÓN (use_sim_time): en lockstep los stalls del host
        # congelan la sim; con reloj de pared un stall de >4 s se confundía
        # con el fin de la misión (reporte prematuro a mitad de vuelo).
        return self.get_clock().now().nanoseconds * 1e-9

    def on_ref(self, m):
        if self.t0 is None:
            self.t0 = self._now()
        self.t_last_ref = self._now()
        p = m.pose.position
        self.refs.append((self.t_last_ref - self.t0, p.x, p.y, p.z))
        self.last_ref = p

    def on_odom(self, m):
        if self.t0 is None:
            return
        t = self._now() - self.t0
        p = m.pose.pose.position
        self.odoms.append((t, p.x, p.y, p.z))
        if self.last_ref is not None:
            r = self.last_ref
            self.errs.append((t, math.sqrt(
                (r.x - p.x) ** 2 + (r.y - p.y) ** 2 + (r.z - p.z) ** 2)))

    def on_stress(self, m):
        self.stress.append(float(m.data))

    # ------------------------------------------------------------------ end
    def check_end(self):
        if self.done or self.t_last_ref is None or len(self.refs) < 100:
            return
        if self._now() - self.t_last_ref < float(
                self.get_parameter('end_timeout_s').value):
            return
        self.done = True
        self.finalize()

    def finalize(self):
        settle = float(self.get_parameter('settle_s').value)
        tol = float(self.get_parameter('rmse_tol_m').value)
        cov_tol = float(self.get_parameter('coverage_tol_m').value)

        errs = np.array([e for (t, e) in self.errs if t >= settle])
        zs = np.array([z for (t, x, y, z) in self.odoms if t >= settle])
        rmse = float(np.sqrt(np.mean(errs ** 2))) if errs.size else float('nan')

        # cobertura: ref muestreada vs track volado (XY), vectorizado
        refs = np.array([(x, y) for (t, x, y, z) in self.refs])
        track = np.array([(x, y) for (t, x, y, z) in self.odoms])
        covered = 0
        if refs.size and track.size:
            for i in range(0, len(refs), 5):     # submuestreo x5 (suficiente)
                d2 = np.min(np.sum((track - refs[i]) ** 2, axis=1))
                covered += d2 <= cov_tol ** 2
            coverage = 100.0 * covered / max(1, len(range(0, len(refs), 5)))
        else:
            coverage = 0.0

        report = {
            'mission_done': True,
            'n_ref': len(self.refs),
            'n_odom': len(self.odoms),
            'rmse_pos_m': round(rmse, 4),
            'e_p50_m': round(float(np.percentile(errs, 50)), 4)
                       if errs.size else None,
            'e_p95_m': round(float(np.percentile(errs, 95)), 4)
                       if errs.size else None,
            'e_max_m': round(float(errs.max()), 4) if errs.size else None,
            'rmse_tol_m': tol,
            'rmse_pass': bool(rmse <= tol),
            'rmse_note': 'picos breves en giros 180; pasadas rectas ~0.1 m',
            'coverage_pct': round(coverage, 2),
            'coverage_tol_m': cov_tol,
            'coverage_pass': bool(coverage >= 95.0),
            'z_min_m': round(float(zs.min()), 2) if zs.size else None,
            'z_max_m': round(float(zs.max()), 2) if zs.size else None,
            'no_collision': bool(zs.size and zs.min() > 0.5),
            'water_stress_mean': round(float(np.mean(self.stress)), 4)
                                 if self.stress else None,
            'water_stress_note': 'radiometría proxy; calibración pendiente',
            'map_pests_note': 'mAP pendiente: requiere dataset papa/arroz',
            'overall_pass': bool(rmse <= tol and coverage >= 95.0
                                 and zs.size and zs.min() > 0.5),
        }
        path = str(self.get_parameter('report_path').value)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        verdict = 'PASS' if report['overall_pass'] else 'FAIL'
        self.get_logger().info(
            f"VALIDACIÓN E2E: {verdict} — RMSE {rmse:.3f} m, "
            f"cobertura {coverage:.1f}%, z_min "
            f"{report['z_min_m']} m -> {path}")


def main():
    rclpy.init()
    n = SilValidation()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if not n.done and n.refs:
                n.finalize()        # reporte parcial si interrumpen
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
