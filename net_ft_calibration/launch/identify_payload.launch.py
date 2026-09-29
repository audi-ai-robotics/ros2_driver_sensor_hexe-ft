from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='Robot namespace'),
        DeclareLaunchArgument('velocity_scaling', default_value='0.1'),
        DeclareLaunchArgument('sensor_frame', default_value='ur16e_tool0'),
        DeclareLaunchArgument('world_frame', default_value='ur16e_base_link'),
        DeclareLaunchArgument('wrench_topic',
                              default_value='force_torque_sensor_broadcaster/wrench'),

        GroupAction(actions=[
            PushROSNamespace(LaunchConfiguration('ns')),
            Node(
                package='net_ft_calibration',
                executable='identify_payload_interactive.py',
                name='identify_payload',
                output='screen',
                prefix='xterm -e' if False else '',
                parameters=[{
                    'robot_ns': LaunchConfiguration('ns'),
                    'sensor_frame': LaunchConfiguration('sensor_frame'),
                    'world_frame': LaunchConfiguration('world_frame'),
                    'wrench_topic': LaunchConfiguration('wrench_topic'),
                    'velocity_scaling': LaunchConfiguration('velocity_scaling'),
                }],
            ),
        ]),
    ])
