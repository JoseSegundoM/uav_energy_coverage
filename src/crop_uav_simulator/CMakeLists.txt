cmake_minimum_required(VERSION 3.8)
project(crop_uav_simulator)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)

# Viento OU del MIL (theta=1.0, fuerza [N])
add_executable(wind_disturbance_node src/wind_disturbance_node.cpp)
ament_target_dependencies(wind_disturbance_node rclcpp geometry_msgs)

# Puente térmico: mono8 (Gazebo classic) -> 32FC1 [K] (contrato §2)
add_executable(thermal_bridge_node src/thermal_bridge_node.cpp)
ament_target_dependencies(thermal_bridge_node rclcpp sensor_msgs)

# Plugin de motores del gemelo (modelo de fuerzas del MIL + /uav/odom)
add_library(quadrotor_actuation_plugin SHARED src/quadrotor_actuation_plugin.cpp)
ament_target_dependencies(quadrotor_actuation_plugin
  rclcpp nav_msgs std_msgs geometry_msgs gazebo_dev gazebo_ros)

install(TARGETS wind_disturbance_node thermal_bridge_node
  DESTINATION lib/${PROJECT_NAME})
# Validador E2E (P5): RMSE + cobertura + reporte JSON
install(PROGRAMS scripts/sil_validation_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sil_validation_node)
# Grabador documental: compone las 3 vistas en un cuadro 1920x1080 -> mp4 + PNG
install(PROGRAMS scripts/mission_recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mission_recorder_node)
install(TARGETS quadrotor_actuation_plugin
  LIBRARY DESTINATION lib)
install(DIRECTORY launch worlds models DESTINATION share/${PROJECT_NAME})
ament_package()
