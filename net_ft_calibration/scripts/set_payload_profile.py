#!/usr/bin/env python3
"""
Switch the active F/T payload profile (robot payload + RTDE streamer sensor_bias).

Profiles live in payload_params.yaml (top level = 'default', named ones under `profiles`).
Only switch while force_mode is inactive.

Usage:
  ros2 run net_ft_calibration set_payload_profile.py ecu --ros-args -p robot_ns:=Peach
  ros2 run net_ft_calibration set_payload_profile.py default --ros-args -p robot_ns:=Peach
  ros2 run net_ft_calibration set_payload_profile.py --list
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from net_ft_calibration import PayloadProfileSwitcher, list_payload_profiles


def main():
    rclpy.init(args=sys.argv)
    args = remove_ros_args(sys.argv)[1:]
    node = Node('set_payload_profile')
    node.declare_parameter('robot_ns', '')
    node.declare_parameter('payload_params_file', '')
    switcher = PayloadProfileSwitcher(
        node,
        robot_ns=node.get_parameter('robot_ns').value,
        params_file=node.get_parameter('payload_params_file').value,
    )
    ok = True
    try:
        if len(args) != 1 or args[0] == '--list':
            profiles = list_payload_profiles(switcher.params_file)
            print(f"Profiles in {switcher.params_file}: {', '.join(profiles)}")
            ok = len(args) == 1
        else:
            ok = switcher.apply(args[0])
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
