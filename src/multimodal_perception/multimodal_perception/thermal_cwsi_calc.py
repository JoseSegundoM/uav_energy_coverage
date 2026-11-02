"""thermal_cwsi_calc.py — Índice de estrés hídrico del cultivo (CWSI empírico).

  suscribe:  /uav/camera/thermal/image_raw   sensor_msgs/Image (32FC1, Kelvin)
  publica:   /perception/cwsi_map            sensor_msgs/Image (32FC1, [0,1])
             /perception/water_stress        std_msgs/Float32  (media del mapa)

Fórmula (CWSI empírico — fuentes canónicas: Idso et al. 1981, Agricultural
Meteorology; Jackson et al. 1981, Water Resources Research. VERIFICAR DOI y
páginas antes de citarlas en la tesis; no copiar a ciegas):

    CWSI = ((Tc - Ta) - dT_LL) / (dT_UL - dT_LL),  recortado a [0, 1]

  Tc    = temperatura de canopia (térmica, K -> °C)
  Ta    = temperatura del aire [°C]
  dT_LL = línea base sin estrés. Modo VPD (Idso): dT_LL = a + b * VPD
  dT_UL = límite superior no-transpirante

CALIBRACIÓN (obligatoria antes de usar resultados):
  Los coeficientes (a, b) y dT_UL son ESPECÍFICOS de cultivo y ambiente. Deben
  calibrarse para papa y arroz en condiciones de Perú (mediciones de campo con
  lisímetro/porómetro o parcelas de referencia bien regadas / sin riego).
  Los valores por defecto de este nodo son PLACEHOLDERS de simulación: el nodo
  lo advierte en el log y NO deben presentarse como calibrados.

  VPD [kPa] se calcula con Tetens: es(T) = 0.6108*exp(17.27*T/(T+237.3)),
  VPD = es(Ta) * (1 - RH/100).
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge


def tetens_es_kpa(t_c: float) -> float:
    """Presión de vapor de saturación [kPa] (Tetens)."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


class ThermalCwsi(Node):
    def __init__(self):
        super().__init__('thermal_cwsi_calc')
        # --- ambiente ---
        self.declare_parameter('air_temp_c', 22.0)
        self.declare_parameter('rel_humidity_pct', 60.0)
        # --- baselines (PLACEHOLDERS — calibrar por cultivo, ver docstring) ---
        self.declare_parameter('use_vpd_baseline', False)
        self.declare_parameter('idso_a', 1.0)    # dT_LL = a + b*VPD  [°C]
        self.declare_parameter('idso_b', -1.5)   # [°C/kPa] (pendiente típica <0)
        self.declare_parameter('dT_LL', 1.0)     # modo fijo (si use_vpd=false)
        self.declare_parameter('dT_UL', 5.0)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/uav/camera/thermal/image_raw', self.on_thermal, 10)
        self.pub_map = self.create_publisher(Image, '/perception/cwsi_map', 10)
        self.pub_mean = self.create_publisher(
            Float32, '/perception/water_stress', 10)
        self.get_logger().warn(
            'thermal_cwsi_calc: baselines dT_LL/dT_UL son PLACEHOLDERS sin '
            'calibrar (papa/arroz, Perú). No usar resultados como válidos '
            'hasta calibrar (ver docstring).')

    def _baseline_ll(self) -> float:
        if bool(self.get_parameter('use_vpd_baseline').value):
            ta = float(self.get_parameter('air_temp_c').value)
            rh = float(self.get_parameter('rel_humidity_pct').value)
            vpd = tetens_es_kpa(ta) * (1.0 - rh / 100.0)   # [kPa]
            a = float(self.get_parameter('idso_a').value)
            b = float(self.get_parameter('idso_b').value)
            return a + b * vpd
        return float(self.get_parameter('dT_LL').value)

    def on_thermal(self, msg: Image):
        tc = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough').astype(np.float32)
        if np.nanmax(tc) > 200.0:   # heurística Kelvin -> °C
            tc = tc - 273.15
        ta = float(self.get_parameter('air_temp_c').value)
        ll = self._baseline_ll()
        ul = float(self.get_parameter('dT_UL').value)
        denom = max(ul - ll, 1e-6)
        cwsi = np.clip(((tc - ta) - ll) / denom, 0.0, 1.0).astype(np.float32)
        # NaN del sensor -> NaN en el mapa (no inventar datos); la media los ignora
        cwsi[np.isnan(tc)] = np.nan

        out = self.bridge.cv2_to_imgmsg(cwsi, encoding='32FC1')
        out.header = msg.header
        self.pub_map.publish(out)
        m = Float32()
        m.data = float(np.nanmean(cwsi)) if np.any(np.isfinite(cwsi)) else 0.0
        self.pub_mean.publish(m)


def main():
    rclpy.init()
    n = ThermalCwsi()
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
