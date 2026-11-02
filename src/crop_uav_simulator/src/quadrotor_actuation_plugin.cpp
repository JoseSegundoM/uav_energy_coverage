// quadrotor_actuation_plugin.cpp — Plugin de motores del gemelo digital.
//
// Cierra el contrato de tópicos (docs/TECHNICAL_SPEC.md) del lado Gazebo:
//   suscribe:  /uav/cmd_actuators  std_msgs/Float64MultiArray  [d1..d4] in [0,1]
//              /uav/wind           geometry_msgs/Vector3Stamped (FUERZA [N], OU MIL)
//   publica:   /uav/odom           nav_msgs/Odometry (100 Hz)
//
// Modelo de fuerzas = el del MIL (uav_dynamics_casadi.py), aplicado sobre el
// cuerpo rígido que integra Gazebo (ODE):
//   - rotores 1er orden:  dOm_i = (ALPHA*V_NOM*d_i - Om_i)/TAU_M
//   - mezcla X:  U1 = kf*sum(Om^2);  U2,U3 = kf*l/sqrt(2)*(...);  U4 = km*(...)
//   - drag traslacional:  F = -CD*|v|*v   (marco mundo)
//   - viento: fuerza externa en marco mundo (acc = w/m, igual que el MIL)
// Se omiten (igual que el modelo nominal del NMPC): efecto suelo, blade
// flapping y batería (V_bat = V_NOM). Documentado: vuelo a >5 m, despreciables.
//
// Convención de /uav/odom (la que consume nmpc_node):
//   twist.linear  = velocidad en marco MUNDO; twist.angular = body rates.

#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

#include <array>
#include <chrono>
#include <mutex>

namespace crop_uav
{

// Parámetros físicos del MIL (uav_dynamics_casadi.py) — NO cambiar sin
// regenerar el C de acados; el gemelo debe coincidir con el modelo de control.
constexpr int    kStepsPerCycle = 5;      // pasos de 1 ms por ciclo de lazo (5 ms)
constexpr double kMass      = 0.80;       // kg
constexpr double kArm       = 0.175;      // m
constexpr double kKf        = 2.3e-5;     // N s^2/rad^2
constexpr double kKm        = 7.0e-7;     // N m s^2/rad^2
constexpr double kTauM      = 0.03;       // s
constexpr double kOmegaMax  = 1500.0;     // rad/s  (= ALPHA*V_NOM)
constexpr double kCd        = 0.15;       // kg/m   (drag traslacional)
constexpr double kCdw       = 5.0e-4;     // kg·m^2/rad (drag rotacional, MIL)
constexpr double kOmegaHover = 292.0;     // rad/s

class QuadrotorActuationPlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    link_ = model->GetLink("base_link");
    if (!link_) {
      gzerr << "[quadrotor_actuation] no existe link 'base_link'\n";
      return;
    }

    ros_node_ = gazebo_ros::Node::Get(sdf);
    lockstep_ = ros_node_->declare_parameter<bool>("lockstep", true);

    // LOCKSTEP (patrón PX4 SITL): el lazo NMPC validado exige retardo < ~5 ms
    // (medido: 10 ms lo desestabiliza) y un host WSL2 no garantiza eso en
    // tiempo real. En lockstep, la física avanza EXACTAMENTE un ciclo de lazo
    // (5 ms) por cada comando recibido: control<->planta quedan síncronos por
    // construcción, igual que el MIL validado. La sim corre a la velocidad
    // del lazo (~tiempo real o más rápido). Desactivable: lockstep=false.
    cmd_sub_ = ros_node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/uav/cmd_actuators", 10,
      [this](std_msgs::msg::Float64MultiArray::SharedPtr m) {
        if (m->data.size() < 4) return;
        {
          std::lock_guard<std::mutex> lk(mtx_);
          for (int i = 0; i < 4; ++i) {
            u_[i] = std::clamp(m->data[i], 0.0, 1.0);
          }
          last_cmd_time_ = model_->GetWorld()->SimTime();
          last_cmd_wall_ = std::chrono::steady_clock::now();
        }
        if (lockstep_) {
          auto world = model_->GetWorld();
          if (!world->IsPaused()) world->SetPaused(true);
          world->Step(kStepsPerCycle);   // avanza 5 ms de física (lazo 200 Hz)
        }
      });
    wind_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/uav/wind", 10,
      [this](geometry_msgs::msg::Vector3Stamped::SharedPtr m) {
        std::lock_guard<std::mutex> lk(mtx_);
        wind_ = {m->vector.x, m->vector.y, m->vector.z};
      });
    odom_pub_ = ros_node_->create_publisher<nav_msgs::msg::Odometry>(
      "/uav/odom", 10);
    // Telemetría de rotores (análoga al RPM de los ESC reales): permite al
    // NMPC usar el estado VERDADERO de los rotores en x0 en vez de una
    // estimación en lazo abierto que diverge en transitorios agresivos.
    rotor_pub_ = ros_node_->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/uav/rotor_speeds", 10);

    // Antes del primer comando: rotores en hover (el dron no cae en el spawn).
    omega_.fill(kOmegaHover);
    u_.fill(kOmegaHover / kOmegaMax);

    last_sim_time_ = model_->GetWorld()->SimTime();
    update_conn_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&QuadrotorActuationPlugin::OnUpdate, this,
                std::placeholders::_1));

    // KEEP-ALIVE del lockstep: la cadena cmd->step->odom->cmd es el único
    // motor de la sim; si un mensaje se pierde en un pico de CPU (WSL2), la
    // sim queda congelada "de la nada". Este timer de PARED detecta la
    // cadena rota (mundo pausado, sin comandos >0.5 s) y re-publica la
    // odometría actual para re-disparar al controlador.
    keepalive_timer_ = ros_node_->create_wall_timer(
      std::chrono::milliseconds(500), [this] {
        if (!lockstep_) return;
        auto world = model_->GetWorld();
        if (!world->IsPaused()) return;
        const auto now_w = std::chrono::steady_clock::now();
        if (now_w - last_cmd_wall_ > std::chrono::milliseconds(500)) {
          RCLCPP_WARN(ros_node_->get_logger(),
                      "lockstep sin comandos >0.5 s: re-publicando odom "
                      "(keep-alive)");
          PublishOdomNow();
        }
      });
    RCLCPP_INFO(ros_node_->get_logger(),
                "quadrotor_actuation_plugin: modelo de fuerzas MIL activo");
  }

private:
  void OnUpdate(const gazebo::common::UpdateInfo& info)
  {
    const double dt = (info.simTime - last_sim_time_).Double();
    last_sim_time_ = info.simTime;
    if (dt <= 0.0 || dt > 0.1) return;  // pausa/reset del mundo

    std::array<double, 4> u;
    std::array<double, 3> wind;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      u = u_;
      wind = wind_;
      // FAILSAFE (como el de un FC real): si el controlador se atasca y no
      // llegan comandos frescos (>250 ms — medido: stalls de WSL2 de hasta
      // 3.6 s), retener el último u agresivo voltea el dron. Decaer a hover.
      // (El umbral deja margen al jitter normal del lazo a 100 Hz.)
      if (last_cmd_time_.Double() > 0.0 &&
          (info.simTime - last_cmd_time_).Double() > 0.25) {
        u.fill(kOmegaHover / kOmegaMax);
      }
    }

    // 1) dinámica de rotores (1er orden, V_bat = V_NOM absorbido)
    for (int i = 0; i < 4; ++i) {
      omega_[i] += dt * (kOmegaMax * u[i] - omega_[i]) / kTauM;
    }
    const double o1 = omega_[0] * omega_[0], o2 = omega_[1] * omega_[1];
    const double o3 = omega_[2] * omega_[2], o4 = omega_[3] * omega_[3];

    // 2) mezcla X (idéntica al MIL)
    const double s2 = kArm / std::sqrt(2.0);
    const double U1 = kKf * (o1 + o2 + o3 + o4);
    const double U2 = kKf * s2 * (-o1 + o2 + o3 - o4);
    const double U3 = kKf * s2 * (-o1 + o2 - o3 + o4);
    const double U4 = kKm * (o1 + o2 - o3 - o4);

    // 3) fuerzas/pares sobre el cuerpo (Gazebo integra el rígido)
    link_->AddRelativeForce({0.0, 0.0, U1});        // empuje en z del body
    link_->AddRelativeTorque({U2, U3, U4});         // pares body
    // drag rotacional del MIL (-CDw * omega_body): amortigua la actitud igual
    // que la planta de 18 estados validada (antes omitido — sin él la actitud
    // queda subamortiguada y el RTI la excita).
    const auto wb = link_->RelativeAngularVel();
    link_->AddRelativeTorque(-kCdw * wb);
    const auto v = link_->WorldLinearVel();
    link_->AddForce(-kCd * v.Length() * v);          // drag mundo (MIL)
    // Viento = fuerza (MIL), con filtro 1er orden (tau 0.15 s): el mensaje
    // llega muestreado/retenido (50 Hz, y en ráfagas si /clock va a 10 Hz);
    // sin filtro los escalones de fuerza excitan la actitud y se ve "vibrar".
    constexpr double kTauWind = 0.15;
    for (int i = 0; i < 3; ++i) {
      wind_f_[i] += dt * (wind[i] - wind_f_[i]) / kTauWind;
    }
    link_->AddForce({wind_f_[0], wind_f_[1], wind_f_[2]});

    // 4) /uav/odom a 200 Hz (cada 5 pasos de física de 1 ms). El lazo NMPC
    // se dispara por odom: a 200 Hz el retardo de 1 ciclo es 5 ms, por debajo
    // del umbral de inestabilidad del lazo (validado: 5 ms estable, 10 tumba).
    if ((info.simTime - last_pub_time_).Double() >= 0.005) {
      last_pub_time_ = info.simTime;
      PublishOdom(info);
    }
  }

  void PublishOdomNow()
  {
    gazebo::common::UpdateInfo info;
    info.simTime = model_->GetWorld()->SimTime();
    PublishOdom(info);
  }

  void PublishOdom(const gazebo::common::UpdateInfo& info)
  {
    const auto pose = link_->WorldPose();
    const auto vel = link_->WorldLinearVel();
    const auto rates = link_->RelativeAngularVel();   // body rates

    nav_msgs::msg::Odometry m;
    m.header.stamp.sec = static_cast<int32_t>(info.simTime.sec);
    m.header.stamp.nanosec = static_cast<uint32_t>(info.simTime.nsec);
    m.header.frame_id = "map";
    m.child_frame_id = "base_link";
    m.pose.pose.position.x = pose.Pos().X();
    m.pose.pose.position.y = pose.Pos().Y();
    m.pose.pose.position.z = pose.Pos().Z();
    m.pose.pose.orientation.w = pose.Rot().W();
    m.pose.pose.orientation.x = pose.Rot().X();
    m.pose.pose.orientation.y = pose.Rot().Y();
    m.pose.pose.orientation.z = pose.Rot().Z();
    m.twist.twist.linear.x = vel.X();                 // marco MUNDO
    m.twist.twist.linear.y = vel.Y();
    m.twist.twist.linear.z = vel.Z();
    m.twist.twist.angular.x = rates.X();              // body rates
    m.twist.twist.angular.y = rates.Y();
    m.twist.twist.angular.z = rates.Z();
    odom_pub_->publish(m);

    std_msgs::msg::Float64MultiArray rs;
    rs.data = {omega_[0], omega_[1], omega_[2], omega_[3]};
    rotor_pub_->publish(rs);
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::LinkPtr link_;
  gazebo::event::ConnectionPtr update_conn_;
  gazebo::common::Time last_sim_time_;
  gazebo::common::Time last_pub_time_;
  gazebo::common::Time last_cmd_time_;   // watchdog de comandos (failsafe)
  std::chrono::steady_clock::time_point last_cmd_wall_{
      std::chrono::steady_clock::now()};
  rclcpp::TimerBase::SharedPtr keepalive_timer_;

  gazebo_ros::Node::SharedPtr ros_node_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr wind_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rotor_pub_;

  std::mutex mtx_;
  bool lockstep_{true};
  std::array<double, 4> u_{};
  std::array<double, 3> wind_{};
  std::array<double, 3> wind_f_{};   // viento filtrado (1er orden)
  std::array<double, 4> omega_{};
};

GZ_REGISTER_MODEL_PLUGIN(QuadrotorActuationPlugin)

}  // namespace crop_uav
