from glob import glob
from setuptools import find_packages, setup

package_name = "rviz_visualization"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/map", glob("map/*.map")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="RViz2 marker visualization package for Tank Challenge simulator",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rviz_visualizer_node = rviz_visualization.rviz_visualizer_node:main",
            "static_map_loader_node = rviz_visualization.static_map_loader_node:main",
        ],
    },
)
