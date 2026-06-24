from glob import glob
from setuptools import find_packages, setup

package_name = "vision"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/models", glob("models/*")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "PyYAML",
        "torch",
        "ultralytics",
        "flask",
    ],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="YOLO-based visual perception utilities for Tank Challenge simulator",
    license="MIT",
    entry_points={
        "console_scripts": [
            "yolo_debug_server = vision.yolo_debug_server:main",
        ],
    },
)
