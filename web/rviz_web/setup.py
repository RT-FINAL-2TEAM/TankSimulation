from glob import glob
from setuptools import find_packages, setup

package_name = "rviz_web"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "flask"],
    zip_safe=True,
    maintainer="tankcc",
    maintainer_email="tankcc@example.com",
    description="Standalone web RViz-like 3D viewer for Tank Challenge ROS2 topics",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rviz_web_server = rviz_web.web_server_node:main",
        ],
    },
)
