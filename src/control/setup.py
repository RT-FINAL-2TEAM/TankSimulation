from glob import glob
from setuptools import find_packages, setup

package_name = "control"

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
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="PD heading and PID speed controller for Tank Challenge ROS2 bridge",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tank_controller_node = control.tank_controller_node:main",
        ],
    },
)
