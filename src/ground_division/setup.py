from glob import glob
from setuptools import find_packages, setup

package_name = "ground_division"

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
    description="Terrain/obstacle division and LiDAR terrain-map finalization package.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "terrain_record_finalize_node = ground_division.terrain_record_finalize_node:main",
        ],
    },
)
