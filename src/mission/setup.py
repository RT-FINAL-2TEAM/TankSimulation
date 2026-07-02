from setuptools import setup

package_name = 'mission'

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
    description='시나리오2 임무 두뇌 — 전술 decision FSM(돌파/교전/복귀) + mock turret.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'decision_node = mission.decision_node:main',
            'mock_turret_node = mission.mock_turret_node:main',
            'sudden_advisor_node = mission.sudden_advisor_node:main',
        ],
    },
)
