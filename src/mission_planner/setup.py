from setuptools import setup
package_name = 'mission_planner'
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/coverage_demo.launch.py']),
    ],
    install_requires=['setuptools', 'numpy', 'shapely'],
    zip_safe=True,
    author='José Luis Segundo Manayay, Luis Ángel Vilcamiche Fernández',
    maintainer='José Luis Segundo Manayay',
    maintainer_email='jose.segundo.m@uni.pe',
    description='Planificador de cobertura boustrophedon para UAV agrícola.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'coverage_mission_node = mission_planner.coverage_mission_node:main',
        ],
    },
)
