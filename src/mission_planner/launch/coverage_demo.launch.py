# Lanza solo el planificador de cobertura (para verificar la referencia en RViz).
# Integración completa: añadir a crop_uav_simulator/launch/sil_flight_mission.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='mission_planner', executable='coverage_mission_node',
             output='screen',
             parameters=[{
                 'altitude_m': 80.0,
                 'speed_mps': 6.0,
                 'side_overlap': 0.70,
                 'forward_overlap': 0.80,
                 'dt': 0.05,
             }]),
    ])
