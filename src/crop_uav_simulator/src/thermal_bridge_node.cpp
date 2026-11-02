// thermal_bridge_node.cpp — Puente térmico del gemelo digital.
//
// Gazebo classic (11) NO tiene sensor térmico nativo (gz/Ignition sí). El
// gemelo usa una cámara mono8 nadir y este nodo la convierte al contrato:
//   suscribe:  /uav/camera/thermal_8u/image_raw   sensor_msgs/Image (mono8)
//   publica:   /uav/camera/thermal/image_raw      sensor_msgs/Image (32FC1, K)
//
// Mapeo lineal placeholder:  T[K] = t_min_k + (v/255)*(t_max_k - t_min_k).
// Defaults 293.15..313.15 K (20..40 °C, rango típico de canopia). Es un PROXY
// de simulación: la radiometría real se calibra en P3 (CWSI) — no presentar
// estos valores como temperaturas verificadas.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cstring>

class ThermalBridge : public rclcpp::Node {
public:
  ThermalBridge() : Node("thermal_bridge_node") {
    t_min_k_ = declare_parameter<double>("t_min_k", 293.15);
    t_max_k_ = declare_parameter<double>("t_max_k", 313.15);
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/uav/camera/thermal_8u/image_raw", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::SharedPtr m) { convert(*m); });
    pub_ = create_publisher<sensor_msgs::msg::Image>(
      "/uav/camera/thermal/image_raw", 10);
    RCLCPP_INFO(get_logger(),
                "thermal_bridge: mono8 -> 32FC1 [%.2f, %.2f] K (proxy sim)",
                t_min_k_, t_max_k_);
  }

private:
  void convert(const sensor_msgs::msg::Image& in) {
    if (in.encoding != "mono8" && in.encoding != "8UC1") {
      RCLCPP_WARN_ONCE(get_logger(), "encoding inesperado: %s",
                       in.encoding.c_str());
      return;
    }
    sensor_msgs::msg::Image out;
    out.header = in.header;
    out.height = in.height;
    out.width = in.width;
    out.encoding = "32FC1";
    out.is_bigendian = false;
    out.step = in.width * sizeof(float);
    out.data.resize(out.step * out.height);

    const double span = t_max_k_ - t_min_k_;
    auto* dst = reinterpret_cast<float*>(out.data.data());
    for (size_t r = 0; r < in.height; ++r) {
      const uint8_t* src = in.data.data() + r * in.step;
      for (size_t c = 0; c < in.width; ++c) {
        dst[r * in.width + c] =
            static_cast<float>(t_min_k_ + (src[c] / 255.0) * span);
      }
    }
    pub_->publish(out);
  }

  double t_min_k_, t_max_k_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ThermalBridge>());
  rclcpp::shutdown();
  return 0;
}
