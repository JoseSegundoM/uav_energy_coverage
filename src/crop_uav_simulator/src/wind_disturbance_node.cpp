#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include <random>
// Viento: MISMO modelo Ornstein-Uhlenbeck del MIL (drl_residual_env.py, línea
// "Ornstein-Uhlenbeck turbulence update"), para coherencia train<->eval:
//   w += -theta*w*dt + sigma*sqrt(dt)*N(0,1)   por eje, theta = 1.0
// El vector publicado es una FUERZA [N] que entra a la planta como acc = w/m
// (igual que en el MIL: "wind as EXTERNAL FORCE ... acc = wind/mass").
// NOTA: se descarta Dryden/von Kármán a propósito. El modelo VALIDADO en la
// tesis (MIL/DRL) es este OU; usar otro rompería la coherencia train<->eval.
// El sesgo persistente de proximidad (efecto pared/suelo de mina) se omite en
// campo abierto: bias = 0 por defecto, configurable si se requiere.
using namespace std::chrono_literals;

class WindNode : public rclcpp::Node {
public:
  WindNode() : Node("wind_disturbance_node"), gen_(std::random_device{}()) {
    // sigma del OU [N·s^-1/2] — el MIL entrena con sigma ~ U(0,1); la tesis
    // evalúa con sigma fijo. 0.55 es el valor de evaluación heredado del kit.
    sigma_ = declare_parameter<double>("sigma", 0.55);
    theta_ = declare_parameter<double>("theta", 1.0);   // = MIL theta_ou
    dt_    = declare_parameter<double>("dt", 0.02);     // 50 Hz como el stub
    bias_  = declare_parameter<std::vector<double>>("bias", {0.0, 0.0, 0.0});
    w_[0] = w_[1] = w_[2] = 0.0;
    pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>("/uav/wind", 10);
    // Reloj del nodo (no wall): con use_sim_time la integración OU usa el dt
    // de simulación correcto aunque el factor de tiempo real baje (GUI/WSL2).
    timer_ = rclcpp::create_timer(this, get_clock(),
                                  rclcpp::Duration::from_seconds(dt_),
                                  [this] { tick(); });
    RCLCPP_INFO(get_logger(),
                "wind OU (MIL): theta=%.2f sigma=%.2f dt=%.0f ms (fuerza [N])",
                theta_, sigma_, dt_ * 1000.0);
  }

private:
  void tick() {
    // dt MEDIDO con el reloj del nodo: con use_sim_time el /clock de gazebo
    // llega a 10 Hz y el timer dispara en ráfagas; integrar con dt fijo
    // multiplicaría la varianza del OU. Con dt medido la integral es correcta
    // (los ticks de una misma ráfaga tienen dt~0 y no aportan).
    const rclcpp::Time t = now();
    double dt = dt_;
    if (have_t_) {
      dt = (t - t_last_).seconds();
      if (dt < 0.0) dt = 0.0;          // reset del sim
      if (dt > 0.5) dt = dt_;          // hueco largo: no integrar de golpe
    }
    t_last_ = t;
    have_t_ = true;
    if (dt > 1e-6) {
      std::normal_distribution<double> n01(0.0, 1.0);
      const double sq = std::sqrt(dt);
      for (int i = 0; i < 3; ++i) {
        w_[i] += -theta_ * w_[i] * dt + sigma_ * sq * n01(gen_);  // OU del MIL
      }
    }
    geometry_msgs::msg::Vector3Stamped m;
    m.header.stamp = now();
    m.header.frame_id = "map";
    m.vector.x = w_[0] + bias_[0];
    m.vector.y = w_[1] + bias_[1];
    m.vector.z = w_[2] + bias_[2];
    pub_->publish(m);
  }

  double sigma_, theta_, dt_;
  double w_[3];
  std::vector<double> bias_;
  rclcpp::Time t_last_;
  bool have_t_{false};
  std::mt19937 gen_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WindNode>());
  rclcpp::shutdown();
  return 0;
}
