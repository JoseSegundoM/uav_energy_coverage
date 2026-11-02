cmake_minimum_required(VERSION 3.8)
project(uav_nmpc_controller)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# --- Puente con el C generado por acados --------------------------------------
# El C fue exportado por nmpc_model/generate_acados_c_code.py
# (modelo uav_nmpc_18states). Aquí SOLO se enlaza; la matemática no se reimplementa.
set(ACADOS_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/acados_generated")
if(DEFINED ENV{ACADOS_SOURCE_DIR})
  set(ACADOS_ROOT "$ENV{ACADOS_SOURCE_DIR}")
else()
  set(ACADOS_ROOT "$ENV{HOME}/acados")
endif()

find_library(ACADOS_OCP_SOLVER_LIB
  NAMES acados_ocp_solver_uav_nmpc_18states
  PATHS "${ACADOS_GEN_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(ACADOS_LIB acados PATHS "${ACADOS_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(nmpc_node
  src/nmpc_node.cpp
  src/nmpc_solver.cpp
  # integrador exportado por acados (mismo modelo): compensación de retardo.
  # Las funciones del modelo (expl_vde_*) las aporta el .so del ocp_solver.
  "${ACADOS_GEN_DIR}/acados_sim_solver_uav_nmpc_18states.c")
target_include_directories(nmpc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  "${ACADOS_GEN_DIR}"
  "${ACADOS_ROOT}/include"
  "${ACADOS_ROOT}/include/blasfeo/include"
  "${ACADOS_ROOT}/include/hpipm/include")
target_link_libraries(nmpc_node
  "${ACADOS_OCP_SOLVER_LIB}"
  "${ACADOS_LIB}"
  m)
# RPATH: el nodo instalado encuentra las .so de acados sin LD_LIBRARY_PATH.
set_target_properties(nmpc_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${ACADOS_GEN_DIR};${ACADOS_ROOT}/lib")
ament_target_dependencies(nmpc_node
  rclcpp nav_msgs geometry_msgs sensor_msgs std_msgs)

install(TARGETS nmpc_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
ament_package()
