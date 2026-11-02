# sil_flight_mission.launch.py — Orquestador SIL end-to-end (P2).
#
# Lanza: Gazebo (gzserver [+gzclient]) con agricultural_field.world, spawnea el
# gemelo custom_quadrotor (plugin de motores + cámaras), y arranca el lazo
# completo: coverage_mission_node (zigzag) -> nmpc_node (acados C) ->
# plugin Gazebo -> /uav/odom, con viento OU del MIL y percepción/telemetría.
#
# Args útiles:
#   gui:=true|false          gzclient (default false: headless WSL2)
#   wind_sigma:=0.55         sigma del OU [N·s^-1/2]
#   altitude_m / speed_mps   misión
#   field_latlon / ref_latlon  polígono geodésico del campo
#   spawn_x/y/z              pose inicial del dron (ponerlo DENTRO del campo,
#                            cerca del inicio del barrido: evita persecuciones
#                            largas fuera de la envolvente validada del NMPC)
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('crop_uav_simulator')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg_share, 'worlds', 'agricultural_field.world')
    model_sdf = os.path.join(pkg_share, 'models', 'custom_quadrotor', 'model.sdf')

    args = [
        DeclareLaunchArgument('gui', default_value='false'),
        # 0.3 = demo fluida (menos balanceo visible); la validación de tesis
        # usa el sigma de evaluación: pasar wind_sigma:=0.55.
        DeclareLaunchArgument('wind_sigma', default_value='0.3'),
        # 7 m: vuelo bajo (se aprecia el cultivo); la huella se angosta y el
        # planificador añade pasadas para mantener el traslape lateral.
        DeclareLaunchArgument('altitude_m', default_value='7.0'),
        DeclareLaunchArgument('speed_mps', default_value='2.0'),
        DeclareLaunchArgument('mission_dt', default_value='0.05'),
        # DEFAULT: el campo demo 20x14 m del mundo agricultural_field.world
        # (coincide con tools/generate_field_world.py). Para el campo real de
        # ~6 ha pasar field_latlon/ref_latlon/altitude/speed por argumento.
        DeclareLaunchArgument(
            'field_latlon',
            default_value='[-6.77, -79.84, -6.77, -79.8398191, '
                          '-6.7701258, -79.8398191, -6.7701258, -79.84]'),
        DeclareLaunchArgument('ref_latlon', default_value='[-6.7700, -79.8400]'),
        # spawn = inicio del barrido para el campo default (a 7 m: y=-13.17)
        DeclareLaunchArgument('spawn_x', default_value='0.0'),
        DeclareLaunchArgument('spawn_y', default_value='-13.17'),
        DeclareLaunchArgument('spawn_z', default_value='0.15'),
        # P5: validación E2E (RMSE, % cobertura, reporte JSON al terminar)
        DeclareLaunchArgument('validate', default_value='false'),
        # Panel en vivo de percepción (ventanas OpenCV; requiere GUI/WSLg)
        DeclareLaunchArgument('dashboard', default_value='false'),
        # Grabación documental: 3 vistas en un cuadro 1920x1080 -> mp4 + PNG.
        # No necesita display: compone desde los tópicos.
        # OJO: NO llamarlo 'record'. gzserver.launch.py declara su propio
        # 'record' (state logging de Gazebo) y, al compartir contexto, un
        # record:=true de la CLI lo activaría también — sim más lenta y logs
        # gigantes.
        DeclareLaunchArgument('record_video', default_value='false'),
        DeclareLaunchArgument('media_dir', default_value='~/uav_agricultural/media'),
    ]

    # model://field_ground (suelo texturizado) vive en el share del paquete
    model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        os.path.join(pkg_share, 'models') + ':' +
        os.environ.get('GAZEBO_MODEL_PATH', ''))

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items())
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(LaunchConfiguration('gui')))

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py', output='screen',
        arguments=['-entity', 'custom_quadrotor', '-file', model_sdf,
                   '-x', LaunchConfiguration('spawn_x'),
                   '-y', LaunchConfiguration('spawn_y'),
                   '-z', LaunchConfiguration('spawn_z'),
                   # el mundo denso (plantas) puede tardar >30 s en cargar
                   '-timeout', '120'])

    nodes = [
        # Planificador de cobertura zigzag (entregado) -> /uav/reference
        Node(package='mission_planner', executable='coverage_mission_node',
             output='screen',
             parameters=[{
                 'use_sim_time': True,
                 # retener el primer punto 6 s: da margen a que gzclient,
                 # texturas y YOLO terminen de cargar antes del despegue
                 # (en lockstep esa carga se ve como "congelado al inicio")
                 'start_hold_s': 6.0,
                 'field_latlon': LaunchConfiguration('field_latlon'),
                 'ref_latlon': LaunchConfiguration('ref_latlon'),
                 'altitude_m': LaunchConfiguration('altitude_m'),
                 'speed_mps': LaunchConfiguration('speed_mps'),
                 'side_overlap': 0.70, 'forward_overlap': 0.80,
                 'dt': LaunchConfiguration('mission_dt')}]),
        # Control NMPC (acados C embebido), dt = paso del OCP (100 Hz).
        # TODO el lazo va en SIM TIME (/clock de Gazebo): si el factor de
        # tiempo real baja (GUI/WSL2), el control sigue síncrono con la planta.
        Node(package='uav_nmpc_controller', executable='nmpc_node',
             output='screen', parameters=[{'use_sim_time': True}]),
        # Viento OU del MIL (fuerza [N], theta=1.0)
        Node(package='crop_uav_simulator', executable='wind_disturbance_node',
             output='screen',
             parameters=[{'use_sim_time': True,
                          'sigma': LaunchConfiguration('wind_sigma')}]),
        # Puente térmico mono8 -> 32FC1 [K]
        Node(package='crop_uav_simulator', executable='thermal_bridge_node',
             output='screen', parameters=[{'use_sim_time': True}]),
        # Detector de plagas en MODO SIM por defecto: segmentación HSV de las
        # plantas infestadas del gemelo (2 focos con verdad-terreno) -> el
        # pipeline completo funciona y el panel/CSV marcan los focos. Con
        # pesos reales: detector_mode:=auto (y weights/best.pt). El fallback
        # COCO queda desactivado (veía las plantas como "potted plant").
        Node(package='multimodal_perception', executable='yolo_pest_detector',
             output='screen',
             parameters=[{'use_sim_time': True,
                          'detector_mode': 'sim_color',
                          'allow_generic_fallback': False}]),
        # Parámetros CWSI ajustados al RANGO DEL PROXY térmico del gemelo
        # (mono8 -> [293,313] K): follaje sano ~26 C, clorótico/suelo ~29-31 C.
        # Son valores de SIMULACIÓN para que el índice no sature — la
        # calibración real por cultivo sigue pendiente (insumo de Jose).
        Node(package='multimodal_perception', executable='thermal_cwsi_calc',
             output='screen',
             parameters=[{'use_sim_time': True, 'air_temp_c': 24.0,
                          'dT_LL': 2.0, 'dT_UL': 9.0}]),
        Node(package='multimodal_perception', executable='ground_station_node',
             output='screen',
             parameters=[{'use_sim_time': True,
                          'ref_latlon': LaunchConfiguration('ref_latlon'),
                          # parcela demo 20x14 + 0.5 m de margen
                          'field_bounds_xy': [-0.5, 20.5, -14.5, 0.5]}]),
        # Validador E2E (P5) — solo con validate:=true. En SIM TIME: con reloj
        # de pared, un stall del lockstep fingía el fin de la misión.
        Node(package='crop_uav_simulator', executable='sil_validation_node',
             output='screen', parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('validate'))),
        # Panel de percepción en vivo — solo con dashboard:=true
        Node(package='multimodal_perception', executable='mission_dashboard',
             output='screen', parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('dashboard'))),
        # Grabador documental — solo con record:=true
        Node(package='crop_uav_simulator', executable='mission_recorder_node',
             output='screen',
             parameters=[{'use_sim_time': True,
                          'out_dir': LaunchConfiguration('media_dir')}],
             condition=IfCondition(LaunchConfiguration('record_video'))),
    ]

    return LaunchDescription(args + [model_path, gzserver, gzclient, spawn]
                             + nodes)
