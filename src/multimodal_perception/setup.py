from setuptools import setup
package_name = 'multimodal_perception'
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='José Luis Segundo Manayay, Luis Ángel Vilcamiche Fernández',
    maintainer='José Luis Segundo Manayay',
    maintainer_email='jose.segundo.m@uni.pe',
    description='Percepción multimodal y estación terrena para UAV agrícola.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_pest_detector = multimodal_perception.yolo_pest_detector:main',
            'thermal_cwsi_calc = multimodal_perception.thermal_cwsi_calc:main',
            'ground_station_node = multimodal_perception.ground_station_node:main',
            'mission_dashboard = multimodal_perception.mission_dashboard:main',
        ],
    },
)
