from glob import glob
from setuptools import find_packages, setup

package_name = "tank_visual_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="Team visual perception utilities for LiDAR-camera overlay and LiDAR DBSCAN clustering",
    license="MIT",
    entry_points={
        "console_scripts": [
            "lidar_camera_overlay_node = tank_visual_perception.lidar_camera_overlay_node:main",
            "lidar_dbscan_cluster_node = tank_visual_perception.lidar_dbscan_cluster_node:main",
        ],
    },
)
