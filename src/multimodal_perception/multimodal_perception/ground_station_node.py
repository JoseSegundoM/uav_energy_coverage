"""ground_station_node.py — Estación terrena / telemetría (P4).

Georreferencia las detecciones de plaga y el estrés hídrico con /uav/odom,
agrega un mapa de estado del cultivo y lo exporta como CSV georreferenciado.

  suscribe:  /uav/odom                      nav_msgs/Odometry
             /perception/pest_detections   vision_msgs/Detection2DArray
             /perception/water_stress      std_msgs/Float32
  exporta:   <output_dir>/pest_detections.csv   (focos de plaga, lat/lon)
             <output_dir>/cwsi_grid.csv         (grilla de estrés hídrico)

Georreferenciación de plagas (modelo nadir documentado):
  cámara nadir con f = (W/2)/tan(hfov/2) px; un píxel (u,v) se proyecta al
  suelo (terreno plano) desplazado del nadir del dron:
     d_body = ( -(v-cy)*z/f ,  -(u-cx)*z/f )      [x_body, y_body]
     d_world = Rz(yaw) * d_body
  Supuestos: cámara nadir, actitud ~nivelada, terreno plano. El mapeo de ejes
  imagen->body corresponde al montaje del model.sdf (pitch +90 sobre Y).
  Para fotogrametría rigurosa (gimbal, DEM) se extendería con la pose completa.

Conversión local->geodésica: se REUTILIZA local_to_geodetic de
mission_planner.coverage_planner (misma transformación que planificó la misión).

Enlace aire->tierra con latencia/pérdida (relé /air/* -> /ground/*): extensión
opcional, no implementada — el SIL actual asume enlace ideal.
"""
import csv
import math
import os
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray

# Reutilizar la transformación geodésica del planificador (no duplicar fórmulas)
from mission_planner.coverage_planner import local_to_geodetic


def quat_to_yaw(qw, qx, qy, qz):
    """Yaw (Z) desde cuaternión ROS."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def quat_to_rot(qw, qx, qy, qz):
    """Matriz de rotación body->mundo desde cuaternión ROS."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)]])


class GroundStation(Node):
    def __init__(self):
        super().__init__('ground_station_node')
        # marco geodésico: MISMO ref que el planificador de la misión
        self.declare_parameter('ref_latlon', [-6.7700, -79.8400])
        self.declare_parameter('output_dir', os.path.expanduser(
            '~/uav_agricultural/mission_artifacts'))
        self.declare_parameter('grid_cell_m', 2.0)
        self.declare_parameter('export_period_s', 15.0)
        # cámara RGB del gemelo (model.sdf): hfov 0.7505 rad, 640x480
        self.declare_parameter('cam_hfov_rad', 0.7505)
        self.declare_parameter('cam_width_px', 640)
        self.declare_parameter('cam_height_px', 480)
        # límites de la parcela [xmin, xmax, ymin, ymax]: detecciones
        # proyectadas FUERA se descartan (artefactos del borde del mundo;
        # una estación terrena real solo acepta hits dentro del predio).
        self.declare_parameter('field_bounds_xy', [-1e9, 1e9, -1e9, 1e9])

        gp = self.get_parameter
        self.ref = [float(v) for v in gp('ref_latlon').value]
        self.out_dir = str(gp('output_dir').value)
        self.cell = float(gp('grid_cell_m').value)
        w = int(gp('cam_width_px').value)
        h = int(gp('cam_height_px').value)
        self.cx, self.cy = w / 2.0, h / 2.0
        self.f_px = (w / 2.0) / math.tan(float(gp('cam_hfov_rad').value) / 2.0)

        os.makedirs(self.out_dir, exist_ok=True)

        self.last_odom = None
        self.last_stress_mean = None             # media global (telemetría)
        self.pest_rows = []                      # filas CSV de focos de plaga
        self.cwsi_cells = defaultdict(lambda: [0.0, 0])   # (i,j) -> [suma, n]

        self.create_subscription(Odometry, '/uav/odom', self.on_odom, 10)
        self.create_subscription(
            Detection2DArray, '/perception/pest_detections', self.on_pest, 10)
        self.create_subscription(
            Float32, '/perception/water_stress', self.on_cwsi, 10)
        # Mapa CWSI por píxel: para la grilla se usa el RECORTE CENTRAL del
        # frame (el área directamente bajo el dron). El promedio de TODO el
        # frame (water_stress) mezcla varios metros alrededor y satura la
        # grilla con el suelo — sin contraste espacial (medido).
        self.create_subscription(
            Image, '/perception/cwsi_map', self.on_cwsi_map, 10)
        self.create_timer(float(gp('export_period_s').value), self.export)
        self.get_logger().info(
            f'ground_station: ref={self.ref}, salida={self.out_dir}, '
            f'celda={self.cell} m, f={self.f_px:.1f} px')

    # ------------------------------------------------------------------ subs
    def on_odom(self, m: Odometry):
        self.last_odom = m

    def on_pest(self, m: Detection2DArray):
        if self.last_odom is None or not m.detections:
            return
        p = self.last_odom.pose.pose.position
        q = self.last_odom.pose.pose.orientation
        # Proyección con ACTITUD COMPLETA: el rayo de cámara se rota con el
        # cuaternión y se interseca con el suelo (z=0). La versión nadir
        # (solo yaw) desplazaba los hits 2-4 m cuando el dron volaba con
        # ~10 grados de tilt y el blob caía en el borde del frame.
        R = quat_to_rot(q.w, q.x, q.y, q.z)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        for d in m.detections:
            u = d.bbox.center.position.x
            v = d.bbox.center.position.y
            # rayo en body: cámara nadir, u -> -y_body, v -> -x_body
            d_b = np.array([-(v - self.cy) / self.f_px,
                            -(u - self.cx) / self.f_px, -1.0])
            d_w = R @ d_b
            if d_w[2] > -0.1:
                continue                  # rayo casi horizontal: descartar
            s = -p.z / d_w[2]             # intersección con el suelo z=0
            gx = p.x + s * d_w[0]
            gy = p.y + s * d_w[1]
            bx = [float(v) for v in
                  self.get_parameter('field_bounds_xy').value]
            if not (bx[0] <= gx <= bx[1] and bx[2] <= gy <= bx[3]):
                continue                 # fuera de la parcela: descartar
            lat, lon = local_to_geodetic(gx, gy, self.ref[0], self.ref[1])
            hyp = d.results[0].hypothesis if d.results else None
            self.pest_rows.append({
                't': round(t, 3),
                'x_m': round(gx, 2), 'y_m': round(gy, 2),
                'lat': round(lat, 7), 'lon': round(lon, 7),
                'class': hyp.class_id if hyp else 'unknown',
                'score': round(hyp.score, 3) if hyp else 0.0,
                'src': d.id,
            })

    def on_cwsi(self, m: Float32):
        # media global del frame: solo telemetría (la grilla usa on_cwsi_map)
        self.last_stress_mean = float(m.data)

    def on_cwsi_map(self, m: Image):
        if self.last_odom is None or m.encoding != '32FC1':
            return
        arr = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.width)
        # recorte central 25%: la zona directamente bajo el dron (nadir)
        h4, w4 = m.height // 4, m.width // 4
        center = arr[m.height // 2 - h4:m.height // 2 + h4,
                     m.width // 2 - w4:m.width // 2 + w4]
        if not np.any(np.isfinite(center)):
            return
        val = float(np.nanmean(center))
        p = self.last_odom.pose.pose.position
        key = (int(math.floor(p.x / self.cell)),
               int(math.floor(p.y / self.cell)))
        acc = self.cwsi_cells[key]
        acc[0] += val
        acc[1] += 1

    # ---------------------------------------------------------------- export
    def export(self):
        if self.pest_rows:
            path = os.path.join(self.out_dir, 'pest_detections.csv')
            with open(path, 'w', newline='') as fh:
                wr = csv.DictWriter(fh, fieldnames=list(self.pest_rows[0]))
                wr.writeheader()
                wr.writerows(self.pest_rows)
        if self.cwsi_cells:
            path = os.path.join(self.out_dir, 'cwsi_grid.csv')
            with open(path, 'w', newline='') as fh:
                wr = csv.writer(fh)
                wr.writerow(['x_m', 'y_m', 'lat', 'lon', 'cwsi_mean', 'n'])
                for (i, j), (s, n) in sorted(self.cwsi_cells.items()):
                    cx = (i + 0.5) * self.cell
                    cy = (j + 0.5) * self.cell
                    lat, lon = local_to_geodetic(cx, cy, self.ref[0], self.ref[1])
                    wr.writerow([round(cx, 2), round(cy, 2),
                                 round(lat, 7), round(lon, 7),
                                 round(s / n, 4), n])
        self.get_logger().info(
            f'export: {len(self.pest_rows)} focos de plaga, '
            f'{len(self.cwsi_cells)} celdas CWSI -> {self.out_dir}')


def main():
    rclpy.init()
    n = GroundStation()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            n.export()              # artefacto final garantizado al cerrar
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
