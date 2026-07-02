# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    phone_port = LaunchConfiguration("phone_port")
    phone_endpoint = LaunchConfiguration("phone_endpoint")
    use_phone_yaw = LaunchConfiguration("use_phone_yaw")

    default_config_file = PathJoinSubstitution(
        [
            FindPackageShare("phone_sim2real"),
            "config",
            "phone_sim2real.yaml",
        ]
    )

    phone_yolo_gateway_node = Node(
        package="phone_sim2real",
        executable="phone_yolo_gateway_node",
        name="phone_yolo_gateway_node",
        output="screen",
        parameters=[
            config_file,
            {
                "phone_port": ParameterValue(phone_port, value_type=int),
                "phone_endpoint": phone_endpoint,
            },
        ],
    )

    phone_virtual_obstacle_node = Node(
        package="phone_sim2real",
        executable="phone_virtual_obstacle_node",
        name="phone_virtual_obstacle_node",
        output="screen",
        parameters=[
            config_file,
            {
                "use_phone_yaw": ParameterValue(use_phone_yaw, value_type=bool),
            },
        ],
    )

    phone_cluster_mux_node = Node(
        package="phone_sim2real",
        executable="phone_cluster_mux_node",
        name="phone_cluster_mux_node",
        output="screen",
        parameters=[
            config_file,
        ],
    )

    phone_emergency_brake_node = Node(
        package="phone_sim2real",
        executable="phone_emergency_brake_node",
        name="phone_emergency_brake_node",
        output="screen",
        parameters=[
            config_file,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config_file,
                description="phone_sim2real YAML config file",
            ),
            DeclareLaunchArgument(
                "phone_port",
                default_value="5002",
                description="HTTP port for Android phone app",
            ),
            DeclareLaunchArgument(
                "phone_endpoint",
                default_value="/phone/detect",
                description="HTTP endpoint for phone image detection",
            ),
            DeclareLaunchArgument(
                "use_phone_yaw",
                default_value="false",
                description="Use Android phone yaw for obstacle bearing correction",
            ),
            phone_yolo_gateway_node,
            phone_virtual_obstacle_node,
            phone_cluster_mux_node,
            phone_emergency_brake_node,
        ]
    )
