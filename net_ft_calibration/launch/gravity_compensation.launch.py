from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='Robot namespace'),
        DeclareLaunchArgument('payload_params_file', default_value='',
                              description='Path to payload_params.yaml; auto-detected if empty'),
        DeclareLaunchArgument('sensor_frame', default_value='ur16e_tool0'),
        DeclareLaunchArgument('world_frame', default_value='ur16e_base_link'),
        DeclareLaunchArgument('wrench_topic',
                              default_value='force_torque_sensor_broadcaster/wrench'),
        DeclareLaunchArgument('compensated_topic', default_value='ft_compensated'),

        GroupAction(actions=[
            PushROSNamespace(LaunchConfiguration('ns')),
            Node(
                package='net_ft_calibration',
                executable='gravity_compensation_node.py',
                name='gravity_compensation',
                output='screen',
                parameters=[{
                    'payload_params_file': LaunchConfiguration('payload_params_file'),
                    'sensor_frame': LaunchConfiguration('sensor_frame'),
                    'world_frame': LaunchConfiguration('world_frame'),
                    'wrench_topic': LaunchConfiguration('wrench_topic'),
                    'compensated_topic': LaunchConfiguration('compensated_topic'),
                }],
            ),
        ]),
    ])
