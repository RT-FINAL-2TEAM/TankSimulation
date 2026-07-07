from setuptools import find_packages, setup

package_name = "tank_web_dashboard"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "flask",
        "numpy",
        "opencv-python",
    ],
    zip_safe=True,
    maintainer="tank",
    maintainer_email="tank@example.com",
    description="Separated web dashboard for TankSimulation ROS2 topics",
    license="MIT",
    entry_points={
        "console_scripts": [
            "web_dashboard = tank_web_dashboard.web_server_node:main",
        ],
    },
)