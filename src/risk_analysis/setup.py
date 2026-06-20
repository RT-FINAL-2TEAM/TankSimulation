from setuptools import setup

package_name = 'risk_analysis'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'requests'],
    zip_safe=True,
    maintainer='ldfha',
    maintainer_email='ldfha@example.com',
    description='Route risk analysis with local LLM',
    license='TODO',
    entry_points={
        'console_scripts': [
            'route_risk_node = risk_analysis.route_risk_node:main',
        ],
    },
)
