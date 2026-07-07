from setuptools import find_packages, setup

package_name = "lidar"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="LiDAR preprocessing and map-frame detected point publisher.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "lidar_processor_node = lidar.lidar_processor_node:main",
        ],
    },
)
