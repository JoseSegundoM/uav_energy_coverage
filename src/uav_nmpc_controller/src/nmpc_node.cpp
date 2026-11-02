#include <array>
#include <vector>
#include <chrono>
#include <cmath>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "uav_nmpc_controller/nmpc_solver.hpp"

using namespace std::chrono_literals;

// Constantes del modelo (uav_dynamics_casadi.py — NO cambiar sin regenerar el C):
//   OMEGA_HOVER=292.0 rad/s, V_NOM=16.8 V, TAU_M=0.03 s, ALPHA=OMEGA_MAX/V_NOM.
// Estado acados (18): [x,y,z, dx,dy,dz, q0(w),q1,q2,q3, p,q,r, Om1..Om4, V_bat]
// Convención de cuaternión del modelo: [w,x,y,z]; ROS msg usa (x,y,z,w).
namespace {
constexpr double kOmegaHover = 292.0;    // rad/s
constexpr double kVNom       = 16.8;     // V
constexpr double kTauM       = 0.03;     // s
constexpr double kAlpha      = 1500.0 / 16.8;  // OMEGA_MAX / V_NOM [rad/(s·V)]
constexpr double kVRefMax    = 12.0;     // m/s — clamp del v_ref por diferencias finitas
}  // namespace

class NmpcNode : public rclcpp::Node {
public:
  NmpcNode() : Node("nmpc_node") {
    // dt del lazo = paso de muestreo del OCP (DT=0.01 en generate_acados_c_code.py).
    dt_ = declare_parameter<double>("dt", 0.01);
    // espaciado temporal entre muestras de /uav/reference (dt del planner)
    ref_dt_ = declare_parameter<double>("ref_dt", 0.05);
    // error de yaw máximo entregado al OCP [rad] (gobernador anti-tumbos)
    yaw_err_max_ = declare_parameter<double>("yaw_err_max_rad", 0.3);
    // COMPENSACIÓN DE RETARDO: el NMPC validado exige retardo de actuación
    // < ~5 ms (validado en Python determinista: 10 ms lo tumban). Con el
    // gemelo en LOCKSTEP (default del plugin) el lazo es síncrono y el
    // retardo es CERO -> delay_cycles=0. Para plantas en tiempo real
    // (lockstep=false o hardware), fijar delay_cycles≈retardo/ciclo: x0 se
    // predice a través de los comandos EN VUELO con el integrador exportado
    // por acados y el preview se adelanta el mismo tiempo.
    delay_cycles_ = static_cast<int>(
        declare_parameter<int>("delay_cycles", 0));
    inflight_.assign(std::max(delay_cycles_, 0), kHoverU);
    v_bat_ = declare_parameter<double>("v_bat", kVNom);  // sin sensor: V_bat nominal

    if (!solver_.init()) {
      RCLCPP_FATAL(get_logger(), "No se pudo crear el solver de acados. Abortando.");
      throw std::runtime_error("acados create failed");
    }

    omega_hat_.fill(kOmegaHover);  // estimación de rotores: arranque en hover

    // RELIABLE + keep_last(1): en lockstep la cadena cmd->step->odom->cmd es
    // el ÚNICO motor de la simulación — un odom perdido (best-effort bajo
    // pico de CPU) la deja congelada para siempre. Reliable garantiza la
    // entrega; keep_last(1) evita procesar backlog tras un stall.
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/uav/odom", rclcpp::QoS(1).reliable(),
      [this](nav_msgs::msg::Odometry::SharedPtr m) { onOdom(*m); });
    ref_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/uav/reference", 10,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr m) { onReference(*m); });
    // Telemetría de rotores de la planta (tipo RPM de ESC). Si está presente,
    // x0 usa el estado VERDADERO de los rotores; si no, la estimación 1er orden.
    rotor_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/uav/rotor_speeds", rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
        if (m->data.size() >= 4) {
          for (int i = 0; i < 4; ++i) omega_meas_[i] = m->data[i];
          have_rotor_meas_ = true;
        }
      });
    cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/uav/cmd_actuators", 10);

    // LAZO DISPARADO POR ODOMETRÍA (no por timer): la planta publica
    // /uav/odom a 100 Hz de TIEMPO DE SIMULACIÓN exactos, así que ejecutar el
    // solve en cada odom sincroniza control<->planta sin depender de relojes.
    //   - wall-timer: si el factor de tiempo real baja (GUI/WSL2), el control
    //     corre "más rápido" que la planta -> seguimiento oscilante.
    //   - timer en sim time: /clock de gazebo_ros llega a 10 Hz -> el timer de
    //     10 ms dispara en ráfagas de 10 y la planta queda 100 ms en lazo
    //     abierto entre ráfagas -> inestable.
    // (controlStep se invoca al final de onOdom; no hay timer.)
    RCLCPP_INFO(get_logger(),
                "nmpc_node listo: acados uav_nmpc_18states, dt=%.0f ms (100 Hz).",
                dt_ * 1000.0);
  }

private:
  void onOdom(const nav_msgs::msg::Odometry& m) {
    // dt real entre muestras de la planta (stamps de sim del plugin), para
    // propagar la estimación de rotores aunque varíe el rate.
    const rclcpp::Time t_m(m.header.stamp);
    bool gap = false;
    if (have_odom_) {
      const double d = (t_m - rclcpp::Time(last_odom_.header.stamp)).seconds();
      if (d > 1e-4 && d < 0.1) dt_meas_ = d;
      gap = d > 0.1;   // hueco (stall del host): el warm-start RTI es basura
    }
    last_odom_ = m;
    if (!have_odom_) {
      // Hasta recibir /uav/reference: mantener la primera posición conocida (hover).
      hold_pos_ = {m.pose.pose.position.x, m.pose.pose.position.y,
                   m.pose.pose.position.z};
      have_odom_ = true;
    }
    if (gap) {
      // Recuperación tras stall: resetear el solver (el warm-start de hace
      // >100 ms degenera el QP), re-sincronizar la estimación de rotores al
      // hover y ceder un ciclo publicando hover neutro.
      RCLCPP_WARN(get_logger(),
                  "hueco de odometría > 100 ms: reset del solver + hover");
      solver_.reset();
      omega_hat_.fill(kOmegaHover);
      std_msgs::msg::Float64MultiArray cmd;
      cmd.data = {kHoverU[0], kHoverU[1], kHoverU[2], kHoverU[3]};
      cmd_pub_->publish(cmd);
      return;
    }
    controlStep();   // lazo síncrono con la planta (ver nota en el constructor)
  }

  void onReference(const geometry_msgs::msg::PoseStamped& m) {
    if (have_ref_) {
      // v_ref por diferencias finitas con el ESPACIADO DETERMINISTA de la
      // trayectoria del planner (parámetro ref_dt = dt de la misión), no con
      // relojes: las muestras consecutivas son puntos consecutivos de la
      // trayectoria, así que la diferencia es EXACTA (sin ruido) y NO se
      // filtra: un EMA retrasa el cambio de dirección en las esquinas del
      // zigzag y el preview extrapola hacia atrás -> error artificial ->
      // maniobra violenta (medido: tilt 40-114 grados en cada giro).
      const std::array<double, 3> v_prev = v_ref_;
      for (int i = 0; i < 3; ++i) {
        const double d = (i == 0) ? m.pose.position.x - last_ref_.pose.position.x
                       : (i == 1) ? m.pose.position.y - last_ref_.pose.position.y
                                  : m.pose.position.z - last_ref_.pose.position.z;
        v_ref_[i] = std::clamp(d / ref_dt_, -kVRefMax, kVRefMax);
        // a_ref exacta del perfil trapezoidal (para el preview cuadrático)
        a_ref_[i] = std::clamp((v_ref_[i] - v_prev[i]) / ref_dt_, -15.0, 15.0);
      }
    }
    last_ref_ = m;
    last_ref_time_ = now();
    have_ref_ = true;
  }

  void controlStep() {
    if (!have_odom_) return;  // sin estado no hay control

    // --- x0 (18): orden EXACTO del plant de acados ---------------------------
    std::array<double, uav::NmpcSolver::kNx> x0{};
    const auto& P = last_odom_.pose.pose.position;
    const auto& Q = last_odom_.pose.pose.orientation;   // ROS: (x,y,z,w)
    const auto& V = last_odom_.twist.twist.linear;      // marco mundo (ver nota)
    const auto& W = last_odom_.twist.twist.angular;     // body rates
    // NOTA: la planta SIL publica twist.linear en marco MUNDO (igual que el
    // modelo MIL). Al integrar Gazebo en P2, verificar el marco del plugin.
    x0[0] = P.x;  x0[1] = P.y;  x0[2] = P.z;
    x0[3] = V.x;  x0[4] = V.y;  x0[5] = V.z;
    // Cuaternión reordenado a [w,x,y,z] + renormalización (higiene numérica).
    double qn = std::sqrt(Q.w * Q.w + Q.x * Q.x + Q.y * Q.y + Q.z * Q.z);
    if (qn < 1e-9) qn = 1.0;
    x0[6] = Q.w / qn;  x0[7] = Q.x / qn;  x0[8] = Q.y / qn;  x0[9] = Q.z / qn;
    x0[10] = W.x;  x0[11] = W.y;  x0[12] = W.z;
    // Rotores: estimación CAUSAL con el modelo de actuador 1er orden del MIL,
    // propagada con los comandos propios. NO usar la telemetría /uav/rotor_speeds
    // en x0: llega con ~1 ciclo (10 ms) de retraso y, con tau_m=30 ms, ese
    // desfase crea un lazo algebraico inestable (medido: tumbo continuo).
    // La telemetría queda para logging/validación offline.
    for (int i = 0; i < 4; ++i) x0[13 + i] = omega_hat_[i];
    x0[17] = v_bat_;

    // Compensación de retardo: predecir x0 a través de los comandos en vuelo
    // (los publicados que la planta aún no ha aplicado al instante de medida).
    // Cada paso de predicción dura un CICLO DE LAZO real (dt_meas_, 5 ms a
    // 200 Hz), no el dt del OCP.
    for (const auto& u_inf : inflight_) {
      if (!solver_.predict(x0, u_inf, dt_meas_)) break;
    }

    // Referencia obsoleta (stream detenido): anular v_ref/a_ref para que el
    // preview no extrapole por delante de una referencia congelada.
    if (have_ref_ && (now() - last_ref_time_).seconds() > 0.3) {
      v_ref_ = {0.0, 0.0, 0.0};
      a_ref_ = {0.0, 0.0, 0.0};
    }

    // --- p_ref (13): [p_ref, v_ref, q_ref(wxyz), w_ref] ----------------------
    std::array<double, uav::NmpcSolver::kNp> p_ref{};
    if (have_ref_) {
      p_ref[0] = last_ref_.pose.position.x;
      p_ref[1] = last_ref_.pose.position.y;
      p_ref[2] = last_ref_.pose.position.z;
      p_ref[3] = v_ref_[0];  p_ref[4] = v_ref_[1];  p_ref[5] = v_ref_[2];
      // GOBERNADOR de yaw: el planner (yaw_mode=course) salta 180 grados en
      // cada giro del zigzag. Con error de yaw grande, el NMPC de cuaternión
      // (pesos de actitud isotrópicos, roll/pitch con ~4x mas autoridad que
      // yaw) reduce el error geodésico más rápido "volteándose" que girando
      // -> tumbos violentos (medido: roll std 50 deg en transiciones).
      // Solución: el yaw de referencia sigue al yaw MEDIDO + offset acotado
      // (yaw_err_max), así el error de actitud nunca habilita el camino tumbo
      // y el giro sale coordinado al ritmo que el lazo pueda seguir.
      const auto& qr = last_ref_.pose.orientation;
      const double yaw_target = std::atan2(
          2.0 * (qr.w * qr.z + qr.x * qr.y),
          1.0 - 2.0 * (qr.y * qr.y + qr.z * qr.z));
      const double yaw_meas = std::atan2(
          2.0 * (Q.w * Q.z + Q.x * Q.y),
          1.0 - 2.0 * (Q.y * Q.y + Q.z * Q.z));
      double dyaw = yaw_target - yaw_meas;
      while (dyaw > M_PI)  dyaw -= 2.0 * M_PI;   // camino corto
      while (dyaw < -M_PI) dyaw += 2.0 * M_PI;
      const double yaw_cmd =
          yaw_meas + std::clamp(dyaw, -yaw_err_max_, yaw_err_max_);
      p_ref[6] = std::cos(yaw_cmd / 2.0);        // q_ref = yaw puro (wxyz)
      p_ref[9] = std::sin(yaw_cmd / 2.0);
      // w_ref = 0 (índices 10..12 ya en cero)
    } else {
      // Sin referencia todavía: hover en la posición inicial, actitud identidad.
      p_ref[0] = hold_pos_[0];  p_ref[1] = hold_pos_[1];  p_ref[2] = hold_pos_[2];
      p_ref[6] = 1.0;
    }

    // --- solve (preview adelantado el retardo compensado) ---------------------
    std::array<double, uav::NmpcSolver::kNu> u{};
    const double preview_offset = delay_cycles_ * dt_meas_;
    const bool ok = solver_.solve(x0, p_ref, preview_offset, u);
    if (!ok) {
      // Un QP degenerado (p.ej. ACADOS_MINSTEP) deja iterados basura y SQP_RTI
      // no se recupera solo: descartar u, resetear memoria y publicar HOVER
      // neutro. (NO el último u: suele ser un transitorio agresivo saturado y
      // retenerlo durante el reset volteaba el dron — medido tilt ~180.)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "acados status != 0 -> reset + hover neutro");
      solver_.reset();
      u = kHoverU;
    }
    const double solve_ms = solver_.lastSolveTimeMs();
    if (solve_ms > dt_ * 1000.0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "solve %.3f ms > dt %.1f ms (deadline incumplido)",
                           solve_ms, dt_ * 1000.0);
    }

    // --- publicar PWM + propagar estimación de rotores -------------------------
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data = {u[0], u[1], u[2], u[3]};   // delta_i in [0,1]
    cmd_pub_->publish(cmd);
    // Rotores 1er orden, DISCRETIZACIÓN EXACTA (estable para cualquier dt):
    // Omega+ = Omega + (1 - exp(-dt/tau)) * (alpha*V_bat*delta - Omega).
    // Con retardo, la planta aplica AHORA el comando publicado hace
    // delay_cycles (inflight_.front()), no el recién calculado.
    const std::array<double, 4>& u_applied =
        inflight_.empty() ? u : inflight_.front();
    const double a = 1.0 - std::exp(-dt_meas_ / kTauM);
    for (int i = 0; i < 4; ++i) {
      omega_hat_[i] += a * (kAlpha * v_bat_ * u_applied[i] - omega_hat_[i]);
    }
    // rotar el buffer de comandos en vuelo
    if (!inflight_.empty()) {
      inflight_.erase(inflight_.begin());
      inflight_.push_back(u);
    }

    // Telemetría de seguimiento (throttled, no satura el lazo).
    const double ex = p_ref[0] - x0[0], ey = p_ref[1] - x0[1], ez = p_ref[2] - x0[2];
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                         "|e_pos|=%.3f m  solve=%.3f ms  u=[%.2f %.2f %.2f %.2f]",
                         std::sqrt(ex * ex + ey * ey + ez * ez), solve_ms,
                         u[0], u[1], u[2], u[3]);
  }

  uav::NmpcSolver solver_;
  double dt_{0.01};
  double ref_dt_{0.05};     // espaciado de las muestras de /uav/reference
  double dt_meas_{0.01};    // dt real entre odoms (stamps de sim)
  double yaw_err_max_{0.3};  // rad — gobernador de yaw (anti-tumbos)
  int delay_cycles_{1};      // retardo de actuación a compensar [ciclos de dt]
  std::vector<std::array<double, 4>> inflight_;  // comandos en vuelo
  double v_bat_{kVNom};
  std::array<double, 4> omega_hat_{};       // estimación de velocidad de rotores
  std::array<double, 4> omega_meas_{};      // telemetría de rotores (planta)
  bool have_rotor_meas_{false};
  // Fallback ante fallo del solver: hover neutro (delta = Omega_hover/Omega_max).
  static constexpr std::array<double, 4> kHoverU{
      kOmegaHover / 1500.0, kOmegaHover / 1500.0,
      kOmegaHover / 1500.0, kOmegaHover / 1500.0};
  std::array<double, 3> v_ref_{};           // v_ref por diferencias finitas
  std::array<double, 3> a_ref_{};           // a_ref del perfil (preview cuadrático)
  std::array<double, 3> hold_pos_{};        // posición de hover pre-misión

  nav_msgs::msg::Odometry last_odom_;
  geometry_msgs::msg::PoseStamped last_ref_;
  rclcpp::Time last_ref_time_;
  bool have_odom_{false};
  bool have_ref_{false};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ref_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rotor_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NmpcNode>());
  rclcpp::shutdown();
  return 0;
}
