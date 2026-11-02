"""mission_dashboard.py — Panel en vivo de percepción (P3/P4).

Abre dos ventanas OpenCV (requiere GUI/WSLg):
  - "Deteccion de plagas":  cámara RGB con las bounding boxes y clases de
    /perception/pest_detections dibujadas encima.
  - "Estres hidrico (CWSI)": mapa CWSI coloreado (azul=sin estrés, rojo=alto)
    con la media del frame y la escala.

  suscribe: /uav/camera/rgb/image_raw, /perception/pest_detections,
            /perception/cwsi_map, /perception/water_stress

Uso: se lanza con `dashboard:=true` en sil_flight_mission.launch.py, o solo:
  ros2 run multimodal_perception mission_dashboard
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray

import cv2
from cv_bridge import CvBridge


class MissionDashboard(Node):
    def __init__(self):
        super().__init__('mission_dashboard')
        self.bridge = CvBridge()
        self.dets = []
        self.stress = None
        self.gui_ok = True
        self.create_subscription(
            Image, '/uav/camera/rgb/image_raw', self.on_rgb, 10)
        self.create_subscription(
            Detection2DArray, '/perception/pest_detections', self.on_det, 10)
        self.create_subscription(
            Image, '/perception/cwsi_map', self.on_cwsi, 10)
        self.create_subscription(
            Float32, '/perception/water_stress', self.on_stress, 10)
        self.get_logger().info(
            'mission_dashboard: ventanas "Deteccion de plagas" y '
            '"Estres hidrico (CWSI)" (requiere GUI).')

    # ------------------------------------------------------------------ subs
    def on_det(self, m: Detection2DArray):
        self.dets = m.detections

    def on_stress(self, m: Float32):
        self.stress = float(m.data)

    def on_rgb(self, m: Image):
        if not self.gui_ok:
            return
        frame = self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8').copy()
        n_sana, n_plaga = 0, 0
        for d in self.dets:
            cx = int(d.bbox.center.position.x)
            cy = int(d.bbox.center.position.y)
            w2 = int(d.bbox.size_x / 2)
            h2 = int(d.bbox.size_y / 2)
            hyp = d.results[0].hypothesis if d.results else None
            cls = hyp.class_id if hyp else '?'
            # VERDE = planta sana | ROJO = plaga/enfermedad
            if 'healthy' in cls:
                color, tag = (0, 200, 0), 'sana'
                n_sana += 1
            else:
                color, tag = (0, 0, 255), 'PLAGA'
                n_plaga += 1
            label = f'{tag} {hyp.score:.2f}' if hyp else tag
            cv2.rectangle(frame, (cx - w2, cy - h2), (cx + w2, cy + h2),
                          color, 2)
            cv2.putText(frame, label, (cx - w2, max(12, cy - h2 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f'sanas: {n_sana}  plagas: {n_plaga}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self._show('Deteccion de plagas', frame)

    def on_cwsi(self, m: Image):
        if not self.gui_ok:
            return
        cwsi = self.bridge.imgmsg_to_cv2(m, desired_encoding='passthrough')
        u8 = np.nan_to_num(np.clip(cwsi, 0.0, 1.0) * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        color = cv2.resize(color, (480, 360), interpolation=cv2.INTER_NEAREST)
        if self.stress is not None:
            cv2.putText(color, f'CWSI medio: {self.stress:.2f}', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color, 'azul = sin estres   rojo = estres alto',
                    (8, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._show('Estres hidrico (CWSI)', color)

    # ------------------------------------------------------------------- gui
    def _show(self, win: str, img) -> None:
        try:
            cv2.imshow(win, img)
            cv2.waitKey(1)
        except cv2.error as e:
            self.gui_ok = False
            self.get_logger().error(
                f'Sin display para el panel ({e}). Use rqt_image_view o los '
                'CSV de mission_artifacts/.')


def main():
    rclpy.init()
    n = MissionDashboard()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
