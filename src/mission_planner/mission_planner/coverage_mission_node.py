"""
coverage_mission_node.py — Nodo ROS 2 (Humble) de planificación de cobertura.

Construye la trayectoria boustrophedon (zigzag) sobre el polígono del campo,
la parametriza en el tiempo y la PUBLICA como referencia continua para el NMPC:

  publica:
    /uav/reference        geometry_msgs/PoseStamped   (flujo en tiempo real)
    /uav/coverage_path    nav_msgs/Path               (una vez, LATCHED)

Parámetros (declare/get):
  field_latlon   double[]  lat0,lon0,lat1,lon1,...  (vértices del polígono)
  ref_latlon     double[]  [lat0, lon0]  origen del marco local ENU
  altitude_m     double
  speed_mps      double
  side_overlap   double    (0..1)
  forward_overlap double   (0..1)
  dt             double    paso de muestreo de la referencia
  cam_focal_mm, cam_sw_mm, cam_sh_mm, cam_w_px, cam_h_px  (modelo de cámara)
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from .coverage_planner import CoveragePlanner, Camera, geodetic_to_local


def yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class CoverageMissionNode(Node):
    def __init__(self):
        super().__init__("coverage_mission_node")

        # --- parámetros ---
        self.declare_parameter("field_latlon", [
            -6.7690, -79.8415, -6.7665, -79.8380,
            -6.7700, -79.8345, -6.7728, -79.8392])
        self.declare_parameter("ref_latlon", [-6.7700, -79.8400])
        self.declare_parameter("altitude_m", 80.0)
        self.declare_parameter("speed_mps", 6.0)
        self.declare_parameter("side_overlap", 0.70)
        self.declare_parameter("forward_overlap", 0.80)
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("cam_focal_mm", 8.0)
        self.declare_parameter("cam_sw_mm", 6.3)
        self.declare_parameter("cam_sh_mm", 4.7)
        self.declare_parameter("cam_w_px", 1600)
        self.declare_parameter("cam_h_px", 1200)
        self.declare_parameter("frame_id", "map")
        # "fixed": sin maniobras de yaw (cámara nadir cubre igual; el footprint
        # en mundo no depende del rumbo). "course": cámara alineada al avance,
        # exige giros de yaw de 90-180 grados en cada esquina del zigzag.
        self.declare_parameter("yaw_mode", "fixed")

        gp = self.get_parameter
        flat = list(gp("field_latlon").value)
        self.ref = list(gp("ref_latlon").value)
        self.frame = gp("frame_id").value
        self.dt = float(gp("dt").value)

        poly_xy = [
            geodetic_to_local(flat[i], flat[i + 1], self.ref[0], self.ref[1])
            for i in range(0, len(flat), 2)
        ]
        cam = Camera(
            focal_length_mm=float(gp("cam_focal_mm").value),
            sensor_width_mm=float(gp("cam_sw_mm").value),
            sensor_height_mm=float(gp("cam_sh_mm").value),
            image_width_px=int(gp("cam_w_px").value),
            image_height_px=int(gp("cam_h_px").value),
        )
        self.planner = CoveragePlanner(
            polygon_xy=poly_xy,
            altitude_m=float(gp("altitude_m").value),
            speed_mps=float(gp("speed_mps").value),
            camera=cam,
            side_overlap=float(gp("side_overlap").value),
            forward_overlap=float(gp("forward_overlap").value),
            yaw_mode=str(gp("yaw_mode").value),
        )

        plan = self.planner.plan()
        self.traj = self.planner.time_parameterize(plan, dt=self.dt)
        self.idx = 0

        for k, v in plan.metrics.items():
            self.get_logger().info(f"[plan] {k} = {v}")
        self.get_logger().info(
            f"[plan] dirección óptima = {math.degrees(plan.sweep_angle_rad):.1f} deg, "
            f"{len(self.traj)} muestras")

        # --- publicadores ---
        self.ref_pub = self.create_publisher(PoseStamped, "/uav/reference", 10)
        # LATCHED (TRANSIENT_LOCAL): el path se publica una sola vez, así que un
        # suscriptor tardío (RViz, grabador) debe seguir recibiéndolo.
        self.path_pub = self.create_publisher(
            Path, "/uav/coverage_path",
            QoSProfile(depth=1,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self._publish_path(plan)

        # --- gate de integración SIL (añadido en P4, opt-out con parámetros):
        # sin esto, si Gazebo/spawn tardan, la referencia "se escapa" antes del
        # despegue y el dron persigue el zigzag desde decenas de metros atrás.
        #   wait_for_odom: no iniciar el stream hasta la primera /uav/odom
        #   start_hold_s : retener traj[0] unos segundos (ascenso inicial)
        self.declare_parameter("wait_for_odom", True)
        self.declare_parameter("start_hold_s", 3.0)
        self._wait_odom = bool(self.get_parameter("wait_for_odom").value)
        self._hold_ticks = max(
            0, int(float(self.get_parameter("start_hold_s").value) / self.dt))
        self._have_odom = not self._wait_odom
        if self._wait_odom:
            from nav_msgs.msg import Odometry
            self._odom_sub = self.create_subscription(
                Odometry, "/uav/odom", self._on_first_odom, 1)

        # --- flujo de referencia en tiempo real ---
        self.timer = self.create_timer(self.dt, self._tick)

    def _on_first_odom(self, _msg):
        if not self._have_odom:
            self._have_odom = True
            self.get_logger().info(
                "Odometría detectada: iniciando stream de referencia "
                f"(retención inicial {self._hold_ticks * self.dt:.1f} s).")

    def _publish_path(self, plan):
        path = Path()
        path.header.frame_id = self.frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in plan.waypoints_xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(plan.altitude_m)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def _tick(self):
        if not self._have_odom:
            return                      # esperar al gemelo (gate SIL)
        if self.idx >= len(self.traj):
            self.get_logger().info("Misión de cobertura completada.")
            self.timer.cancel()
            return
        if self._hold_ticks > 0:        # retención del primer punto (ascenso)
            self._hold_ticks -= 1
            _, x, y, z, yaw = self.traj[0]
        else:
            _, x, y, z, yaw = self.traj[self.idx]
            self.idx += 1
        msg = PoseStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.ref_pub.publish(msg)


def main():
    rclpy.init()
    node = CoverageMissionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
