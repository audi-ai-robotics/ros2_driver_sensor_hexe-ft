from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace, SetRemap


# ros2 launch does not forward stdin, so the interactive keys don't reach this node; use
# `ros2 run ... --ros-args -r __ns:=/<ns> -r /tf:=tf -r /tf_static:=tf_static -p robot_ns:=<ns>`.
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='Robot namespace'),
        DeclareLaunchArgument('velocity_scaling', default_value='0.1'),
        DeclareLaunchArgument('sensor_frame', default_value='ur16e_tool0'),
        DeclareLaunchArgument('world_frame', default_value='ur16e_base_link'),
        DeclareLaunchArgument('wrench_topic',
                              default_value='force_torque_sensor_broadcaster/wrench'),
        DeclareLaunchArgument('ur_wrench_topic',
                              default_value='force_torque_sensor_broadcaster/wrench',
                              description='UR built-in wrench, used by the measuring-offset test'),

        DeclareLaunchArgument('joint_poses_file', default_value='',
                              description='Poses for auto-capture (a); empty = config/joint_poses.yaml'),

        GroupAction(actions=[
            PushROSNamespace(LaunchConfiguration('ns')),
            # robot publishes TF under the namespace; keep the listener's /tf in-namespace too
            SetRemap(src='/tf', dst='tf'),
            SetRemap(src='/tf_static', dst='tf_static'),
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
                    'ur_wrench_topic': LaunchConfiguration('ur_wrench_topic'),
                    'velocity_scaling': LaunchConfiguration('velocity_scaling'),
                    'joint_poses_file': LaunchConfiguration('joint_poses_file'),
                }],
            ),
        ]),
    ])
