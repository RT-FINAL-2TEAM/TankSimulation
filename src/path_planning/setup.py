from glob import glob
from setuptools import find_packages, setup

package_name = "path_planning"

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
    description="A* global planner for Tank Challenge map files",
    license="MIT",
    entry_points={
        "console_scripts": [
            "map_astar_planner_node = path_planning.map_astar_planner_node:main",
            "local_path_node = path_planning.local_path_node:main",
        ],
    },
)
