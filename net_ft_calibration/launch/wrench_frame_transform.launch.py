from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace, SetRemap


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='Robot namespace'),
        DeclareLaunchArgument('sensor_frame', default_value='hex_e_fts_link'),
        DeclareLaunchArgument('tcp_frame', default_value='tcp'),
        DeclareLaunchArgument('input_topic', default_value='ft_compensated'),
        DeclareLaunchArgument('output_topic', default_value='ft_compensated_tcp'),

        GroupAction(actions=[
            PushROSNamespace(LaunchConfiguration('ns')),
            # robot publishes TF under the namespace; keep the listener's /tf in-namespace too
            SetRemap(src='/tf', dst='tf'),
            SetRemap(src='/tf_static', dst='tf_static'),
            Node(
                package='net_ft_calibration',
                executable='wrench_frame_transform_node.py',
                name='wrench_frame_transform',
                output='screen',
                parameters=[{
                    'sensor_frame': LaunchConfiguration('sensor_frame'),
                    'tcp_frame': LaunchConfiguration('tcp_frame'),
                    'input_topic': LaunchConfiguration('input_topic'),
                    'output_topic': LaunchConfiguration('output_topic'),
                }],
            ),
        ]),
    ])
