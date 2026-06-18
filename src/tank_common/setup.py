from setuptools import setup

package_name = 'tank_common'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jun',
    maintainer_email='wol00107@gmail.com',
    description='전차 워크스페이스 공용 헬퍼 (LiDAR PointCloud2→numpy 변환 등)',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
