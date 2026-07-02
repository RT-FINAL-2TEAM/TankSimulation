from glob import glob
from setuptools import find_packages, setup

package_name = "phone_sim2real"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="Smartphone camera/IMU sim-to-real obstacle gateway for Tank Challenge ROS2 stack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "phone_yolo_gateway_node = phone_sim2real.phone_yolo_gateway_node:main",
            "phone_virtual_obstacle_node = phone_sim2real.phone_virtual_obstacle_node:main",
            "phone_cluster_mux_node = phone_sim2real.phone_cluster_mux_node:main",
            "phone_emergency_brake_node = phone_sim2real.phone_emergency_brake_node:main",
        ],
    },
)
