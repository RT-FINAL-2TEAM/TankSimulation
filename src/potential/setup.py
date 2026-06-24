from glob import glob
from setuptools import find_packages, setup

package_name = "potential"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnuav",
    maintainer_email="pnuav@example.com",
    description="APF vector publisher for Tank Challenge",
    license="MIT",
    entry_points={
        "console_scripts": [
            "potential_field_node = potential.potential_field_node:main",
        ],
    },
)
