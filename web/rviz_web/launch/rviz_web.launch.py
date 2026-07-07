from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    web_port = LaunchConfiguration("web_port")
    host = LaunchConfiguration("host")
    rosbridge_port = LaunchConfiguration("rosbridge_port")
    start_rosbridge = LaunchConfiguration("start_rosbridge")

    return LaunchDescription([
        DeclareLaunchArgument("web_port", default_value="5055"),
        DeclareLaunchArgument("host", default_value="0.0.0.0"),
        DeclareLaunchArgument("rosbridge_port", default_value="9090"),
        DeclareLaunchArgument("start_rosbridge", default_value="true"),
        ExecuteProcess(
            condition=IfCondition(start_rosbridge),
            cmd=["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml", ["port:=", rosbridge_port]],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "run", "rviz_web", "rviz_web_server",
                "--host", host,
                "--port", web_port,
                "--rosbridge-port", rosbridge_port,
            ],
            output="screen",
        ),
    ])
