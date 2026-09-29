#!/usr/bin/env python3
"""
Interactive multi-pose payload identification for F/T sensor gravity compensation.

Moves the robot through diverse orientations, records F/T readings and TF,
then solves for tool mass, center of gravity, and sensor bias.

Assumes the robot driver (start_robot) and F/T sensor are already running.
MoveIt (move_group) must be running for robot motion.

Usage:
  ros2 run net_ft_calibration identify_payload_interactive.py \
      --ros-args -p robot_ns:=robot1

Keys:
  c   Capture current pose (manual)
  s   Save current joint pose to joint_poses.yaml for reuse
  a   Auto-capture all saved joint poses
  r   Run payload identification (least-squares solve)
  w   Save results to payload_params.yaml
  h   Move robot to home
  d   Delete all captured data
  q   Quit
"""

import sys
import select
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R

import tf2_ros

from net_ft_calibration import RobotController, identify_payload, save_payload_params


class WrenchBuffer:
    """Thread-safe rolling buffer for wrench readings."""

    def __init__(self, maxlen: int = 200):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, wrench: np.ndarray):
        with self._lock:
            self._buf.append(wrench)

    def average(self, n: int = 100) -> np.ndarray | None:
        with self._lock:
            samples = list(self._buf)[-n:]
        if len(samples) < 10:
            return None
        return np.mean(samples, axis=0)

    def count(self) -> int:
        with self._lock:
            return len(self._buf)


class IdentifyPayloadNode(Node):
    def __init__(self):
        super().__init__('identify_payload')

        self.declare_parameter('robot_ns', '')
        self.declare_parameter('base_frame', 'ur16e_base_link')
        self.declare_parameter('ee_frame', 'ur16e_tool0')
        self.declare_parameter('sensor_frame', 'ur16e_tool0')
        self.declare_parameter('world_frame', 'ur16e_base_link')
        self.declare_parameter('wrench_topic', 'force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('velocity_scaling', 0.1)
        self.declare_parameter('moveit_config_package', 'ur16e_cell_moveit_config')
        self.declare_parameter('joint_poses_file', '')
        self.declare_parameter('output_file', '')
        self.declare_parameter('capture_samples', 100)
        self.declare_parameter('settle_time', 1.5)

        robot_ns = self.get_parameter('robot_ns').value
        base_frame = self.get_parameter('base_frame').value
        ee_frame = self.get_parameter('ee_frame').value
        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        vel_scaling = self.get_parameter('velocity_scaling').value
        moveit_pkg = self.get_parameter('moveit_config_package').value
        self.capture_samples = self.get_parameter('capture_samples').value
        self.settle_time = self.get_parameter('settle_time').value

        wrench_topic = self.get_parameter('wrench_topic').value
        ns_prefix = f'/{robot_ns}' if robot_ns else ''
        if not wrench_topic.startswith('/'):
            wrench_topic = f'{ns_prefix}/{wrench_topic}'

        pkg_share = get_package_share_directory('net_ft_calibration')
        poses_file = self.get_parameter('joint_poses_file').value
        if not poses_file:
            poses_file = str(Path(pkg_share) / 'config' / 'joint_poses.yaml')
        self.poses_file = Path(poses_file)

        output_file = self.get_parameter('output_file').value
        if not output_file:
            output_file = '/tmp/ft_calibration/payload_params.yaml'
        self.output_file = Path(output_file)

        self.robot = RobotController(
            self, base_frame, ee_frame,
            velocity_scaling=vel_scaling,
            acceleration_scaling=vel_scaling,
            robot_ns=robot_ns,
            moveit_config_package=moveit_pkg,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._wrench_buf = WrenchBuffer(maxlen=500)
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, 10)

        self._captured_wrenches = []
        self._captured_rotations = []
        self._result = None
        self._stop = False

    def _wrench_cb(self, msg: WrenchStamped):
        w = msg.wrench
        self._wrench_buf.append(np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ]))

    def _get_sensor_rotation(self) -> np.ndarray | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self.sensor_frame, self.world_frame, rclpy.time.Time()
            )
            r = tf.transform.rotation
            return R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def _capture_pose(self) -> bool:
        wrench_avg = self._wrench_buf.average(self.capture_samples)
        if wrench_avg is None:
            self.get_logger().error("Not enough wrench samples")
            return False

        R_sw = self._get_sensor_rotation()
        if R_sw is None:
            return False

        self._captured_wrenches.append(wrench_avg)
        self._captured_rotations.append(R_sw)

        n = len(self._captured_wrenches)
        self.get_logger().info(
            f"Captured pose {n}: "
            f"F=[{wrench_avg[0]:.2f}, {wrench_avg[1]:.2f}, {wrench_avg[2]:.2f}] N  "
            f"T=[{wrench_avg[3]:.4f}, {wrench_avg[4]:.4f}, {wrench_avg[5]:.4f}] Nm"
        )
        return True

    def _save_joint_pose(self):
        state = self.robot.get_state()
        q = state['q']
        names = state['joint_names']
        arm_joints = [n for n in names if not n.startswith('hand_e')]
        arm_q = q[:len(arm_joints)]

        if self.poses_file.exists():
            with open(self.poses_file, 'r') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        poses = data.get('joint_poses', [])
        poses.append([round(v, 6) for v in arm_q])
        data['joint_poses'] = poses

        self.poses_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.poses_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        self.get_logger().info(f"Saved joint pose #{len(poses)} to {self.poses_file}")

    def _auto_capture(self):
        if not self.poses_file.exists():
            self.get_logger().error(f"No joint poses file: {self.poses_file}")
            return

        with open(self.poses_file, 'r') as f:
            data = yaml.safe_load(f)
        joint_poses = data.get('joint_poses', [])
        if len(joint_poses) < 2:
            self.get_logger().error(f"Need at least 2 poses, got {len(joint_poses)}")
            return

        self.get_logger().info(f"Auto-capturing {len(joint_poses)} poses...")
        for i, pose in enumerate(joint_poses):
            self.get_logger().info(f"--- Pose {i + 1}/{len(joint_poses)} ---")
            try:
                self.robot.move_joints(pose)
                time.sleep(self.settle_time)
                for _ in range(20):
                    rclpy.spin_once(self, timeout_sec=0.05)
                self._capture_pose()
            except Exception as e:
                self.get_logger().error(f"Failed at pose {i + 1}: {e}")

        self.get_logger().info(
            f"Auto-capture complete: {len(self._captured_wrenches)} total captures"
        )

    def _run_identification(self):
        n = len(self._captured_wrenches)
        if n < 3:
            self.get_logger().error(f"Need at least 3 captures, have {n}")
            return

        wrenches = np.array(self._captured_wrenches)
        self._result = identify_payload(wrenches, self._captured_rotations)

        self.get_logger().info("=== Payload Identification Result ===")
        self.get_logger().info(f"  Mass:             {self._result['mass']:.4f} kg")
        self.get_logger().info(
            f"  Center of gravity: [{self._result['cog'][0]:.4f}, "
            f"{self._result['cog'][1]:.4f}, {self._result['cog'][2]:.4f}] m"
        )
        bias = self._result['bias']
        self.get_logger().info(
            f"  Force bias:       [{bias[0]:.3f}, {bias[1]:.3f}, {bias[2]:.3f}] N"
        )
        self.get_logger().info(
            f"  Torque bias:      [{bias[3]:.5f}, {bias[4]:.5f}, {bias[5]:.5f}] Nm"
        )
        self.get_logger().info(f"  Residual norm:    {self._result['residual_norm']:.4f}")
        self.get_logger().info(f"  Condition number: {self._result['condition_number']:.1f}")
        self.get_logger().info(f"  Poses used:       {n}")

        if self._result['condition_number'] > 50:
            self.get_logger().warn(
                "High condition number! Poses lack orientation diversity. "
                "Add more poses with different tool orientations."
            )
        if self._result['mass'] < 0.01:
            self.get_logger().warn("Estimated mass is very low. Check sensor data.")

    def _save_result(self):
        if self._result is None:
            self.get_logger().error("No result to save. Press 'r' first.")
            return

        path = save_payload_params(
            self._result, self.output_file,
            sensor_frame=self.sensor_frame,
            world_frame=self.world_frame,
        )
        self.get_logger().info(f"Saved payload params to {path}")

    def _print_help(self):
        self.get_logger().info("--- Commands ---")
        self.get_logger().info("  c  Capture current pose")
        self.get_logger().info("  s  Save current joint pose to file")
        self.get_logger().info("  a  Auto-capture all saved joint poses")
        self.get_logger().info("  r  Run identification")
        self.get_logger().info("  w  Save results")
        self.get_logger().info("  h  Move to home")
        self.get_logger().info("  d  Delete all captured data")
        self.get_logger().info("  q  Quit")

    def run(self):
        self.get_logger().info("Waiting for F/T data and TF...")
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._wrench_buf.count() > 10:
                break
        else:
            self.get_logger().error("No wrench data received within timeout")
            return

        self.get_logger().info("Waiting for robot...")
        if not self.robot.wait_for_ready(timeout_sec=15.0):
            self.get_logger().error("Robot not ready")
            return

        self.get_logger().info("Ready. Type a command and press Enter (? for help):")
        self._print_help()

        spinner = threading.Thread(
            target=self._spin_loop, daemon=True
        )
        spinner.start()

        try:
            while not self._stop:
                try:
                    line = input("> ").strip().lower()
                except EOFError:
                    break

                if not line:
                    continue

                cmd = line[0]
                if cmd == 'c':
                    self._capture_pose()
                elif cmd == 's':
                    self._save_joint_pose()
                elif cmd == 'a':
                    self._auto_capture()
                elif cmd == 'r':
                    self._run_identification()
                elif cmd == 'w':
                    self._save_result()
                elif cmd == 'h':
                    self.get_logger().info("Moving to home...")
                    self.robot.go_home()
                elif cmd == 'd':
                    self._captured_wrenches.clear()
                    self._captured_rotations.clear()
                    self._result = None
                    self.get_logger().info("Cleared all captured data")
                elif cmd == 'q':
                    break
                elif cmd == '?':
                    self._print_help()
                else:
                    self.get_logger().info(f"Unknown command: '{cmd}'. Press '?' for help.")

        except KeyboardInterrupt:
            self.get_logger().info("Interrupted")

        self._stop = True
        spinner.join(timeout=2.0)

    def _spin_loop(self):
        while not self._stop:
            rclpy.spin_once(self, timeout_sec=0.02)


def main():
    rclpy.init()
    node = IdentifyPayloadNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
