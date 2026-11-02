#!/usr/bin/env python3
"""mission_recorder_node.py — Grabador documental de la misión SIL.

Compone en UN SOLO cuadro 1920x1080 las tres vistas del sistema y lo escribe a
vídeo (y a PNG cada `still_period_s`), sin capturar la pantalla: todo llega por
tópicos, así que funciona headless y es reproducible.

    +----------------------------------------+------------------+
    |  Vista de persecución (/sim/chase)      |  Plagas (RGB +   |
    |  el vehículo sobre el cultivo           |  Detection2DArray)|
    |----------------------------------------+------------------+
    |  Telemetría: mapa de cobertura + KPIs   |  CWSI (32FC1)     |
    +----------------------------------------+------------------+

suscribe: /sim/chase/image_raw, /uav/camera/rgb/image_raw,
          /perception/pest_detections, /perception/cwsi_map,
          /perception/water_stress, /uav/odom, /uav/reference,
          /uav/coverage_path, /uav/wind

Uso:  ros2 launch crop_uav_simulator sil_flight_mission.launch.py record:=true
      ros2 run crop_uav_simulator mission_recorder_node --ros-args \
           -p out_dir:=/tmp/media -p video_fps:=25
"""
import math
import os
import shutil
import subprocess
import threading
import time
from queue import Full, Queue

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray

import cv2
from cv_bridge import CvBridge

# ---------------------------------------------------------------- paleta (BGR)
BG = (20, 17, 13)          # #0d1117 fondo
PANEL = (33, 28, 22)       # #161b22 tarjeta
EDGE = (60, 52, 43)        # #2b343c borde
FG = (243, 237, 230)       # #e6edf3 texto
MUTE = (165, 152, 139)     # #8b98a5 texto secundario
OK = (100, 200, 60)        # verde
WARN = (60, 160, 240)      # ámbar
BAD = (60, 60, 235)        # rojo
REF = (200, 170, 60)       # cian/azul referencia
TRACK = (110, 210, 90)     # traza recorrida

W, H = 1920, 1080
HEAD = 64                  # franja de cabecera
COL = 1280                 # ancho de la columna izquierda
CHASE_H = 720              # alto de la vista de persecución
F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def put(img, text, org, scale=0.5, color=FG, thick=1, font=F):
    # Las fuentes Hershey de OpenCV son ASCII: cualquier otro glifo saldría '?'.
    cv2.putText(img, text.encode('ascii', 'replace').decode(), org, font,
                scale, color, thick, cv2.LINE_AA)


def _ffmpeg_exe():
    """ffmpeg del sistema o el binario que trae imageio-ffmpeg; None si no hay."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoSink:
    """Escribe cuadros BGR a H.264 por ffmpeg; si no hay, cae a cv2 (mp4v).

    Dos decisiones que importan:
      - El encoder corre en un HILO con cola acotada. A 1920x1080 un cuadro son
        6 MB: escribir a la tubería desde el callback bloquea el ejecutor de ROS
        y, en lockstep, eso congela la simulación entera. Si la cola se llena se
        DESCARTA el cuadro — antes perder fluidez del vídeo que falsear la
        dinámica.
      - `-preset veryfast`: `medium` a 1080p satura la CPU que Gazebo necesita
        para renderizar y para el propio lazo de control.
    mp4v (el fallback sin ffmpeg) produce archivos ~10x mayores; por encima de
    100 MB GitHub ya no acepta el fichero sin LFS.
    """

    def __init__(self, path, size, fps, crf=23, preset='veryfast', queue=6):
        self.path, self.size, self.fps = path, size, fps
        self.proc = self.writer = None
        exe = _ffmpeg_exe()
        if exe:
            self.proc = subprocess.Popen(
                [exe, '-hide_banner', '-loglevel', 'error', '-y',
                 '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                 '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-',
                 '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
                 '-pix_fmt', 'yuv420p',
                 # MP4 FRAGMENTADO: el índice se escribe de forma incremental,
                 # así que el fichero es reproducible aunque el proceso muera
                 # sin finalizar. Con el mp4 clásico, un SIGKILL deja el
                 # archivo sin `moov atom` y por tanto ilegible.
                 '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
                 path],
                stdin=subprocess.PIPE)
            self.codec = f'H.264 (libx264, {preset})'
        else:
            self.writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f'no se pudo abrir {path} para escribir')
            self.codec = 'MPEG-4 part 2 (sin ffmpeg)'

        self.q = Queue(maxsize=queue)
        self.dropped = 0
        self._stop = False
        self.th = threading.Thread(target=self._pump, daemon=True)
        self.th.start()

    def _pump(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                if self.proc is not None:
                    self.proc.stdin.write(item)
                else:
                    self.writer.write(item)
            except (BrokenPipeError, ValueError):
                break

    def write(self, frame):
        payload = frame.tobytes() if self.proc is not None else frame
        try:
            self.q.put_nowait(payload)
            return True
        except Full:
            self.dropped += 1
            return False

    def close(self):
        if self._stop:
            return
        self._stop = True
        self.q.put(None)
        self.th.join(timeout=30.0)
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait(timeout=60)
            self.proc = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def card(img, x0, y0, x1, y1, title=None):
    """Tarjeta con borde y barra de título; devuelve el rect interior."""
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), PANEL, -1)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), EDGE, 1)
    if title is None:
        return x0 + 1, y0 + 1, x1 - 1, y1 - 1
    put(img, title, (x0 + 12, y0 + 20), 0.52, FG, 1, FD)
    cv2.line(img, (x0 + 1, y0 + 28), (x1 - 2, y0 + 28), EDGE, 1)
    return x0 + 1, y0 + 29, x1 - 1, y1 - 1


def fit(dst, img, rect):
    """Encaja `img` dentro de `rect` conservando la relación de aspecto."""
    x0, y0, x1, y1 = rect
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0 or img is None:
        return
    s = min(bw / img.shape[1], bh / img.shape[0])
    w, h = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    r = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ox, oy = x0 + (bw - w) // 2, y0 + (bh - h) // 2
    dst[oy:oy + h, ox:ox + w] = r


def placeholder(w, h, text):
    img = np.full((h, w, 3), PANEL, np.uint8)
    (tw, _), _ = cv2.getTextSize(text, F, 0.6, 1)
    put(img, text, ((w - tw) // 2, h // 2), 0.6, MUTE)
    return img


class MissionRecorder(Node):
    def __init__(self):
        super().__init__('mission_recorder')
        self.bridge = CvBridge()

        p = self.declare_parameter
        self.out_dir = os.path.expanduser(
            str(p('out_dir', '~/uav_agricultural/media').value))
        self.video_fps = float(p('video_fps', 25.0).value)   # fps del archivo
        self.grab_hz = float(p('grab_hz', 15.0).value)       # composición [Hz]
        self.still_period_s = float(p('still_period_s', 12.0).value)
        # La misión termina cuando el planificador deja de emitir referencia.
        # Cerrar aquí evita depender de que la señal de apagado llegue a tiempo.
        self.stop_after_idle_s = float(p('stop_after_idle_s', 10.0).value)
        self.crop_label = str(p('crop_label', 'Tomato | demo plot 20x14 m').value)
        os.makedirs(self.out_dir, exist_ok=True)

        # --- estado de los tópicos
        self.chase = None
        self.rgb = None
        self.dets = []
        self.cwsi = None
        self.stress = None
        self.odom = None
        self.ref = None
        self.wind = (0.0, 0.0, 0.0)
        self.path_xy = None          # (N,2) referencia de cobertura
        self.visited = None          # (N,) bool
        self.track = []              # traza recorrida (submuestreada)
        self.t0_sim = None
        self.n_det_frame = (0, 0)
        self.t_last_ref = None
        self.finished = False

        latched = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(Image, '/sim/chase/image_raw', self.on_chase, 1)
        self.create_subscription(
            Image, '/uav/camera/rgb/image_raw', self.on_rgb, 1)
        self.create_subscription(
            Detection2DArray, '/perception/pest_detections', self.on_det, 10)
        self.create_subscription(Image, '/perception/cwsi_map', self.on_cwsi, 1)
        self.create_subscription(
            Float32, '/perception/water_stress', self.on_stress, 10)
        self.create_subscription(Odometry, '/uav/odom', self.on_odom, 10)
        self.create_subscription(
            PoseStamped, '/uav/reference', self.on_ref, 10)
        self.create_subscription(
            Path, '/uav/coverage_path', self.on_path, latched)
        self.create_subscription(
            Vector3Stamped, '/uav/wind', self.on_wind, 10)

        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.video_path = os.path.join(self.out_dir, f'mission_{stamp}.mp4')
        self.sink = VideoSink(self.video_path, (W, H), self.video_fps)

        self.n_frames = 0
        self.n_stills = 0
        self.t_last_still = -1e9
        self.timer = self.create_timer(1.0 / self.grab_hz, self.compose)
        self.get_logger().info(
            f'grabando -> {self.video_path} ({W}x{H} @ {self.video_fps:g} fps, '
            f'composicion {self.grab_hz:g} Hz, {self.sink.codec})')

    # ---------------------------------------------------------------- callbacks
    def on_chase(self, m):
        self.chase = self.bridge.imgmsg_to_cv2(m, 'bgr8')

    def on_rgb(self, m):
        self.rgb = self.bridge.imgmsg_to_cv2(m, 'bgr8')

    def on_det(self, m):
        self.dets = m.detections

    def on_cwsi(self, m):
        self.cwsi = self.bridge.imgmsg_to_cv2(m, 'passthrough')

    def on_stress(self, m):
        self.stress = float(m.data)

    def on_wind(self, m):
        self.wind = (m.vector.x, m.vector.y, m.vector.z)

    def on_ref(self, m):
        self.ref = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
        self.t_last_ref = time.monotonic()

    def on_path(self, m):
        self.path_xy = np.array(
            [[ps.pose.position.x, ps.pose.position.y] for ps in m.poses],
            dtype=float)
        self.visited = np.zeros(len(self.path_xy), dtype=bool)

    def on_odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.t0_sim is None:
            self.t0_sim = t
        self.t_sim = t - self.t0_sim
        pp = m.pose.pose.position
        self.odom = (pp.x, pp.y, pp.z,
                     m.twist.twist.linear.x, m.twist.twist.linear.y,
                     m.twist.twist.linear.z, m.pose.pose.orientation)
        if self.path_xy is not None and len(self.path_xy):
            d = np.hypot(self.path_xy[:, 0] - pp.x, self.path_xy[:, 1] - pp.y)
            self.visited |= d < 1.5      # mismo criterio que sil_validation_node
        if not self.track or (pp.x - self.track[-1][0]) ** 2 + \
                (pp.y - self.track[-1][1]) ** 2 > 0.04:
            self.track.append((pp.x, pp.y))

    # ------------------------------------------------------------------ paneles
    def panel_pests(self, w, h):
        """RGB nadir con las cajas de /perception/pest_detections."""
        if self.rgb is None:
            return placeholder(w, h, 'waiting for /uav/camera/rgb ...')
        img = self.rgb.copy()
        n_ok, n_bad = 0, 0
        for d in self.dets:
            b = d.bbox
            cx, cy = int(b.center.position.x), int(b.center.position.y)
            w2, h2 = int(b.size_x / 2), int(b.size_y / 2)
            hyp = d.results[0].hypothesis if d.results else None
            cls = hyp.class_id if hyp else '?'
            if 'healthy' in cls:
                col, tag = OK, 'healthy'
                n_ok += 1
            else:
                col, tag = BAD, 'PEST'
                n_bad += 1
            cv2.rectangle(img, (cx - w2, cy - h2), (cx + w2, cy + h2), col, 2)
            lbl = f'{tag} {hyp.score:.2f}' if hyp else tag
            put(img, lbl, (cx - w2, max(12, cy - h2 - 5)), 0.45, col, 1)
        self.n_det_frame = (n_ok, n_bad)
        return img

    def panel_cwsi(self, w, h):
        """Mapa CWSI en falso color + barra de escala."""
        if self.cwsi is None:
            return placeholder(w, h, 'waiting for /perception/cwsi_map ...')
        u8 = np.nan_to_num(np.clip(self.cwsi, 0.0, 1.0) * 255.0).astype(np.uint8)
        img = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        img = cv2.resize(img, (w, h - 40), interpolation=cv2.INTER_NEAREST)
        # franja de escala BAJO el mapa, para no tapar dato con la leyenda
        strip = np.full((40, w, 3), PANEL, np.uint8)
        bx0, bx1, by = 14, w - 14, 20
        ramp = np.linspace(0, 255, bx1 - bx0).astype(np.uint8)[None, :]
        strip[by:by + 14, bx0:bx1] = cv2.applyColorMap(
            np.repeat(ramp, 14, axis=0), cv2.COLORMAP_JET)
        cv2.rectangle(strip, (bx0, by), (bx1 - 1, by + 14), EDGE, 1)
        put(strip, '0.0  no stress', (bx0, by - 5), 0.42, MUTE)
        put(strip, 'high stress  1.0', (bx1 - 132, by - 5), 0.42, MUTE)
        return np.vstack([img, strip])

    def panel_telemetry(self, w, h):
        """Mapa de cobertura (izquierda) + indicadores clave (derecha)."""
        img = np.full((h, w, 3), PANEL, np.uint8)
        mw = 470                                   # ancho del mapa
        cv2.line(img, (mw, 8), (mw, h - 8), EDGE, 1)

        # --- mapa: barrido planificado, traza real y posiciones
        if self.path_xy is not None and len(self.path_xy) > 1:
            pad = 26
            xs, ys = self.path_xy[:, 0], self.path_xy[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            sx = (mw - 2 * pad) / max(1e-6, x1 - x0)
            sy = (h - 2 * pad) / max(1e-6, y1 - y0)
            s = min(sx, sy)
            offx = (mw - s * (x1 - x0)) / 2.0
            offy = (h - s * (y1 - y0)) / 2.0

            def to_px(x, y):
                # y del mundo hacia arriba en pantalla
                return (int(offx + (x - x0) * s),
                        int(h - (offy + (y - y0) * s)))

            pts = np.array([to_px(x, y) for x, y in self.path_xy], np.int32)
            cv2.polylines(img, [pts], False, EDGE, 1, cv2.LINE_AA)
            if self.visited is not None and self.visited.any():
                seen = pts[self.visited]
                for px, py in seen[::3]:
                    cv2.circle(img, (px, py), 1, (90, 120, 70), -1)
            if len(self.track) > 1:
                tp = np.array([to_px(x, y) for x, y in self.track], np.int32)
                cv2.polylines(img, [tp], False, TRACK, 2, cv2.LINE_AA)
            if self.ref is not None:
                rx, ry = to_px(self.ref[0], self.ref[1])
                cv2.drawMarker(img, (rx, ry), REF, cv2.MARKER_CROSS, 13, 2)
            if self.odom is not None:
                dx, dy = to_px(self.odom[0], self.odom[1])
                cv2.circle(img, (dx, dy), 6, (255, 255, 255), -1)
                cv2.circle(img, (dx, dy), 6, BG, 1)
            put(img, 'COVERAGE PATH (boustrophedon)', (14, 20), 0.46, MUTE)
            put(img, 'planned', (14, h - 42), 0.42, MUTE)
            cv2.line(img, (78, h - 46), (104, h - 46), EDGE, 2)
            put(img, 'flown', (118, h - 42), 0.42, MUTE)
            cv2.line(img, (168, h - 46), (194, h - 46), TRACK, 2)
            put(img, 'reference', (208, h - 42), 0.42, MUTE)
            cv2.drawMarker(img, (272, h - 46), REF, cv2.MARKER_CROSS, 11, 2)
        else:
            put(img, 'waiting for /uav/coverage_path ...', (14, h // 2), 0.55, MUTE)

        # --- indicadores
        err = float('nan')
        if self.odom is not None and self.ref is not None:
            err = math.dist(self.odom[:3], self.ref)
        spd = (math.dist((0, 0, 0), self.odom[3:6])
               if self.odom is not None else float('nan'))
        cov = (100.0 * float(self.visited.mean())
               if self.visited is not None and len(self.visited) else 0.0)
        wmag = math.dist((0, 0, 0), self.wind)
        n_ok, n_bad = self.n_det_frame

        ecol = OK if err < 0.5 else (WARN if err < 0.75 else BAD)
        rows = [
            ('TRACKING ERROR', f'{err:5.3f} m', ecol),
            ('ALTITUDE', f'{self.odom[2]:5.2f} m' if self.odom else '--', FG),
            ('GROUND SPEED', f'{spd:4.2f} m/s', FG),
            ('WIND (OU)', f'{wmag:4.2f} N', FG),
            ('PESTS / HEALTHY (frame)', f'{n_bad} / {n_ok}',
             BAD if n_bad else FG),
            ('MEAN CWSI',
             f'{self.stress:4.2f}' if self.stress is not None else '--', FG),
        ]
        cx, cy = mw + 34, 40
        for i, (k, v, col) in enumerate(rows):
            yy = cy + (i % 3) * 74
            xx = cx + (i // 3) * 390
            put(img, k, (xx, yy), 0.44, MUTE)
            put(img, v, (xx, yy + 34), 0.95, col, 2, FD)

        # --- barra de cobertura
        by = h - 46
        put(img, 'COVERAGE', (cx, by - 12), 0.44, MUTE)
        bx0, bx1 = cx, w - 34
        cv2.rectangle(img, (bx0, by), (bx1, by + 18), EDGE, 1)
        fillx = int(bx0 + (bx1 - bx0) * cov / 100.0)
        cv2.rectangle(img, (bx0 + 1, by + 1), (max(bx0 + 1, fillx), by + 17),
                      OK if cov > 95 else WARN, -1)
        put(img, f'{cov:5.1f} %', (bx1 - 92, by + 14), 0.55, FG, 1, FD)
        return img

    # ------------------------------------------------------------------ compose
    def compose(self):
        if self.finished:
            return
        # fin de misión: el planificador agotó la trayectoria
        if (self.t_last_ref is not None and
                time.monotonic() - self.t_last_ref > self.stop_after_idle_s):
            self.finished = True
            self.get_logger().info(
                'sin referencia nueva: mision terminada, cerrando video')
            self.close()
            return

        f = np.full((H, W, 3), BG, np.uint8)

        # cabecera
        cv2.rectangle(f, (0, 0), (W, HEAD), PANEL, -1)
        cv2.line(f, (0, HEAD - 1), (W, HEAD - 1), EDGE, 1)
        put(f, 'uav_agricultural', (22, 41), 0.86, FG, 1, FD)
        put(f, 'Software-in-the-Loop  |  embedded NMPC (acados C) + '
               'boustrophedon coverage + multimodal perception',
            (272, 40), 0.52, MUTE)
        tsim = getattr(self, 't_sim', 0.0)
        put(f, f'MISSION t = {tsim:6.1f} s', (W - 300, 40), 0.62, FG, 1, FD)
        put(f, self.crop_label, (W - 300, 20), 0.40, MUTE)

        # vista de persecución
        r = card(f, 0, HEAD, COL, HEAD + CHASE_H, None)
        fit(f, self.chase if self.chase is not None
            else placeholder(960, 540, 'waiting for /sim/chase ...'), r)
        put(f, 'GAZEBO  |  CHASE VIEW', (18, HEAD + 28), 0.55, FG, 1, FD)
        put(f, 'digital twin  |  agricultural_field.world',
            (18, HEAD + 50), 0.44, MUTE)

        # telemetría
        f[HEAD + CHASE_H:H, 0:COL] = self.panel_telemetry(COL, H - HEAD - CHASE_H)

        # columna derecha
        mid = HEAD + (H - HEAD) // 2
        r = card(f, COL, HEAD, W, mid, 'PEST DETECTION  |  /perception/pest_detections')
        fit(f, self.panel_pests(640, 480), r)
        r = card(f, COL, mid, W, H, 'WATER STRESS  |  CWSI from thermal imagery')
        fit(f, self.panel_cwsi(560, 420), r)

        self.sink.write(f)
        self.n_frames += 1

        if tsim - self.t_last_still >= self.still_period_s and self.chase is not None:
            self.t_last_still = tsim
            self.n_stills += 1
            p = os.path.join(self.out_dir, f'mission_still_{self.n_stills:02d}.png')
            cv2.imwrite(p, f, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            self.get_logger().info(f'still -> {p}  (t={tsim:.1f} s)')

    def close(self):
        if self.sink is not None:
            drop = self.sink.dropped
            self.sink.close()
            self.sink = None
            mb = os.path.getsize(self.video_path) / 1e6
            self.get_logger().info(
                f'video cerrado: {self.video_path}  ({self.n_frames} cuadros, '
                f'{drop} descartados, {mb:.1f} MB, {self.n_stills} stills)')


def main():
    rclpy.init()
    n = MissionRecorder()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.close()
        try:
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
