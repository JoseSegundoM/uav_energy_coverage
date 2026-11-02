"""yolo_pest_detector.py — Detección de plagas sobre RGB aéreo (YOLOv8).

  suscribe:  /uav/camera/rgb/image_raw      sensor_msgs/Image (bgr8/rgb8)
  publica:   /perception/pest_detections    vision_msgs/Detection2DArray

PESOS / DATASET (pendiente de Jose — NO incluido en el repo):
  El modelo de producción debe entrenarse con un dataset de plagas de papa y
  arroz de la zona de despliegue (Perú). Clases candidatas a anotar:
    papa : polilla de la papa (Phthorimaea operculella), pulguilla (Epitrix
           spp.), gorgojo de los Andes (Premnotrypes spp.), pulgones.
    arroz: sogata (Tagosodes orizicolus), chinche (Oebalus spp.), Hydrellia,
           Spodoptera frugiperda.
  (Lista orientativa para el diseño del dataset; las clases finales las define
  el dataset real anotado — no presentar como taxonomía verificada en campo.)
  Colocar los pesos entrenados en weights/best.pt (parámetro `weights`).

FALLBACK de validación de pipeline: si no hay pesos de plagas, con
`allow_generic_fallback:=true` se usa YOLOv8n (COCO, genérico) SOLO para
verificar el flujo imagen->detecciones->mensaje. Sus clases NO son plagas:
queda registrado en un WARN y en el campo `id` de cada detección.

MODO SIM (detector_mode:=sim_color): detector clásico por segmentación HSV
para el GEMELO DIGITAL — las plantas infestadas del mundo tienen follaje
pálido moteado (firma visual de mosca blanca/oídio) y este modo las encuentra
desde el aire. Ejercita el pipeline completo (bbox -> georef -> CSV -> panel)
con verdad-terreno conocida; cada detección queda etiquetada "sim-color".
detector_mode:=auto usa YOLO (pesos reales o fallback).
"""
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D, Detection2DArray, ObjectHypothesisWithPose,
)
from cv_bridge import CvBridge


class YoloPestDetector(Node):
    def __init__(self):
        super().__init__('yolo_pest_detector')
        self.declare_parameter('weights', 'weights/best.pt')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('allow_generic_fallback', True)
        # 'auto' = YOLO (pesos reales o fallback) | 'sim_color' = segmentación
        # HSV del follaje pálido de las plantas infestadas del gemelo
        self.declare_parameter('detector_mode', 'auto')
        self.declare_parameter('sim_min_area_px', 60)
        # inferir 1 de cada N frames: YOLO en CPU compite con gzserver y, en
        # lockstep, un pico de CPU congela la simulación entera.
        self.declare_parameter('detect_every', 2)
        self.bridge = CvBridge()
        self.model = None
        self.model_tag = ''     # 'pest' | 'generic-coco' | 'sim-color'
        self._frame_count = 0
        self.mode = str(self.get_parameter('detector_mode').value)
        if self.mode != 'sim_color':
            # PRECARGA + warm-up AHORA (al arrancar, con la sim quieta): la
            # carga perezosa en el primer frame congelaba el lockstep 2-4 s.
            self._ensure_model()
            if self.model:
                self.model.predict(np.zeros((480, 640, 3), dtype='uint8'),
                                   verbose=False)
        else:
            self.model_tag = 'sim-color'
            self.get_logger().info(
                'detector_mode=sim_color: segmentación HSV del follaje '
                'infestado del gemelo (pipeline-check con verdad-terreno).')
        self.sub = self.create_subscription(
            Image, '/uav/camera/rgb/image_raw', self.on_image, 10)
        self.pub = self.create_publisher(
            Detection2DArray, '/perception/pest_detections', 10)
        self.get_logger().info('yolo_pest_detector listo (modelo precargado).')

    # ------------------------------------------------------------------ model
    def _ensure_model(self):
        if self.model is not None:
            return
        weights = str(self.get_parameter('weights').value)
        try:
            from ultralytics import YOLO
        except Exception as e:  # ultralytics no instalado
            self.get_logger().error(f'ultralytics no disponible: {e}')
            self.model = False
            return
        if os.path.isfile(weights):
            self.model = YOLO(weights)
            self.model_tag = 'pest'
            self.get_logger().info(f'Pesos de plagas cargados: {weights}')
        elif bool(self.get_parameter('allow_generic_fallback').value):
            # Pipeline-check con COCO genérico. NO detecta plagas.
            self.model = YOLO('yolov8n.pt')
            self.model_tag = 'generic-coco'
            self.get_logger().warn(
                f'No hay pesos de plagas en "{weights}". Usando YOLOv8n COCO '
                'genérico SOLO para validar el pipeline. Entrenar dataset '
                'papa/arroz (ver docstring) antes de cualquier resultado.')
        else:
            self.get_logger().warn(
                f'Sin pesos ("{weights}") y fallback deshabilitado: '
                'se publican Detection2DArray vacíos.')
            self.model = False

    # ------------------------------------------------------------------ image
    # ----------------------------------------------------- detector sim-color
    def _blobs(self, mask, min_area, max_aspect=3.5):
        """Blobs válidos: filtra borde del frame, franjas y no-compactos."""
        H, W = mask.shape
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        found = []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x0 = stats[k, cv2.CC_STAT_LEFT]
            y0 = stats[k, cv2.CC_STAT_TOP]
            bw = int(stats[k, cv2.CC_STAT_WIDTH])
            bh = int(stats[k, cv2.CC_STAT_HEIGHT])
            # borde del frame: georreferenciación del recorte no fiable
            if (x0 <= 1 or y0 <= 1 or x0 + bw >= W - 2 or y0 + bh >= H - 2):
                continue
            # planta nadir ~redonda; franjas (antialiasing) muy alargadas
            if max(bw, bh) > max_aspect * max(min(bw, bh), 1):
                continue
            if area < 0.30 * bw * bh:      # compacidad mínima
                continue
            found.append((float(cents[k][0]), float(cents[k][1]),
                          float(bw), float(bh), area))
        return found

    def detect_sim_color(self, frame, out: Detection2DArray):
        """Clasifica el cultivo del gemelo por color:
          - SANA (verde vivo, saturación alta)       -> healthy_plant_sim
          - INFESTADA (verde-pálido moteado, S baja) -> pest_focus_sim
        El panel pinta VERDE lo sano y ROJO lo infestado.
        H>=36 separa el verde-pálido del amarillo clorótico (estrés H~22-32)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_area = int(self.get_parameter('sim_min_area_px').value)
        # el dosel sano denso se fusiona a lo largo del surco: aspecto
        # permisivo (8). Para plaga se mantiene estricto (3.5: franjas).
        classes = [
            ('pest_focus_sim',
             cv2.inRange(hsv, (36, 26, 130), (95, 115, 255)), min_area, 3.5),
            ('healthy_plant_sim',
             cv2.inRange(hsv, (40, 120, 40), (85, 255, 210)), min_area * 2, 8.0),
        ]
        for class_id, mask, m_area, m_asp in classes:
            for (cx, cy, bw, bh, area) in self._blobs(mask, m_area, m_asp):
                det = Detection2D()
                det.header = out.header
                det.id = f'sim-color:{class_id}'
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = bw
                det.bbox.size_y = bh
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = class_id
                hyp.hypothesis.score = float(min(0.99, 0.3 + area / 4000.0))
                det.results.append(hyp)
                out.detections.append(det)

    def on_image(self, msg: Image):
        out = Detection2DArray()
        out.header = msg.header
        self._frame_count += 1
        every = max(1, int(self.get_parameter('detect_every').value))
        if self._frame_count % every:
            return                      # frame saltado (decimación de CPU)
        if self.mode == 'sim_color':
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.detect_sim_color(frame, out)
            self.pub.publish(out)
            return
        if not self.model:
            self.pub.publish(out)
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        conf = float(self.get_parameter('conf').value)
        iou = float(self.get_parameter('iou').value)
        results = self.model.predict(frame, conf=conf, iou=iou, verbose=False)

        for r in results:
            names = r.names  # id -> nombre de clase del modelo
            if r.boxes is None:
                continue
            # xywh = centro (cx, cy) + tamaño (w, h) en píxeles
            for box in r.boxes:
                cx, cy, w, h = (float(v) for v in box.xywh[0])
                cls_id = int(box.cls[0])
                score = float(box.conf[0])

                det = Detection2D()
                det.header = msg.header
                det.id = f'{self.model_tag}:{names.get(cls_id, cls_id)}'
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.center.theta = 0.0
                det.bbox.size_x = w
                det.bbox.size_y = h

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(names.get(cls_id, cls_id))
                hyp.hypothesis.score = score
                det.results.append(hyp)
                out.detections.append(det)

        self.pub.publish(out)


def main():
    rclpy.init()
    n = YoloPestDetector()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
