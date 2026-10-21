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
  o   Measuring-offset push test (external sensor vs. UR built-in sensor)
  w   Save results to payload_params.yaml

Measuring-offset test (o): run the driver with ft_force_mode_via_rtde:=false so
ur_wrench_topic is the UR built-in sensor, set the pendant TCP to the flange, and keep
the robot still. Push and twist the tool by hand in several directions.
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
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R

from tf2_ros import LookupException, ExtrapolationException

from net_ft_calibration import (
    RobotController,
    estimate_measuring_offset,
    identify_payload,
    load_payload_params,
    save_payload_params,
    split_payload,
)


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

    def samples(self, n: int = 100) -> np.ndarray:
        with self._lock:
            return np.array(list(self._buf)[-n:])

    def clear(self):
        with self._lock:
            self._buf.clear()

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
        # UR built-in wrench, only used by the measuring-offset test.
        self.declare_parameter('ur_wrench_topic', 'force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('push_samples', 50)
        # Total flange payload from the teach pendant (flange frame) and the sensor measuring
        # point in flange frame. Unset = taken from the installed payload_params.yaml.
        self.declare_parameter('robot_payload_mass', 0.0)
        self.declare_parameter('robot_payload_cog', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('measuring_offset', Parameter.Type.DOUBLE_ARRAY)

        robot_ns = self.get_parameter('robot_ns').value
        base_frame = self.get_parameter('base_frame').value
        ee_frame = self.get_parameter('ee_frame').value
        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        vel_scaling = self.get_parameter('velocity_scaling').value
        moveit_pkg = self.get_parameter('moveit_config_package').value
        self.capture_samples = self.get_parameter('capture_samples').value
        pkg_share = get_package_share_directory('net_ft_calibration')
        self.installed_params_file = Path(pkg_share) / 'config' / 'payload_params.yaml'
        self.settle_time = self.get_parameter('settle_time').value
        self.ee_frame = ee_frame
        self.push_samples = self.get_parameter('push_samples').value
        self._load_payload_defaults()

        wrench_topic = self.get_parameter('wrench_topic').value
        ur_wrench_topic = self.get_parameter('ur_wrench_topic').value
        ns_prefix = f'/{robot_ns}' if robot_ns else ''
        if not wrench_topic.startswith('/'):
            wrench_topic = f'{ns_prefix}/{wrench_topic}'
        if not ur_wrench_topic.startswith('/'):
            ur_wrench_topic = f'{ns_prefix}/{ur_wrench_topic}'

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

        self._wrench_buf = WrenchBuffer(maxlen=500)
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, 10)
        self._ur_wrench_buf = WrenchBuffer(maxlen=500)
        if ur_wrench_topic != wrench_topic:
            self.create_subscription(
                WrenchStamped, ur_wrench_topic,
                lambda msg: self._ur_wrench_buf.append(self._to_array(msg)), 10,
            )

        self._captured_wrenches = []
        self._captured_rotations = []
        self._result = None
        self._stop = False

    def _optional_param(self, name):
        """Value of a typed parameter declared without default, or None if it was not set."""
        try:
            return self.get_parameter(name).value
        except ParameterUninitializedException:
            return None

    def _load_payload_defaults(self):
        stored = {}
        if self.installed_params_file.exists():
            try:
                stored = load_payload_params(self.installed_params_file)
            except (KeyError, TypeError) as e:
                self.get_logger().warn(f"Could not read {self.installed_params_file}: {e}")

        self.robot_payload = stored.get('robot_payload')
        mass = self.get_parameter('robot_payload_mass').value
        cog = self._optional_param('robot_payload_cog')
        if mass > 0.0:
            if cog is None:
                raise ValueError('robot_payload_mass needs robot_payload_cog [x, y, z]')
            self.robot_payload = {'mass': mass, 'cog': np.array(cog, dtype=float)}

        offset = self._optional_param('measuring_offset')
        if offset is not None:
            self.measuring_offset = np.array(offset, dtype=float)
        elif stored.get('measuring_offset') is not None:
            self.measuring_offset = stored['measuring_offset']
        else:
            # flange-to-sensor spacer (7 mm) + HEX-E height (37.5 mm); refine with the 'o' test
            self.measuring_offset = np.array([0.0, 0.0, 0.0445])

        if self.robot_payload is None:
            self.get_logger().warn(
                "No robot_payload (pendant total payload) known; RTDE sensor parameters will not "
                "be derived. Set robot_payload_mass/robot_payload_cog or add it to payload_params.yaml."
            )

    @staticmethod
    def _to_array(msg: WrenchStamped) -> np.ndarray:
        w = msg.wrench
        return np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ])

    def _wrench_cb(self, msg: WrenchStamped):
        self._wrench_buf.append(self._to_array(msg))

    def _get_flange_sensor_rotation(self) -> np.ndarray | None:
        try:
            tf = self.robot._tf_buffer.lookup_transform(
                self.ee_frame, self.sensor_frame, rclpy.time.Time()
            )
            r = tf.transform.rotation
            return R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def _get_sensor_rotation(self) -> np.ndarray | None:
        try:
            tf = self.robot._tf_buffer.lookup_transform(
                self.sensor_frame, self.world_frame, rclpy.time.Time()
            )
            r = tf.transform.rotation
            return R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def _capture_pose(self) -> bool:
        # only use samples received after this call, so a stopped topic can't hand back old data
        self._wrench_buf.clear()
        deadline = time.time() + 5.0
        while self._wrench_buf.count() < self.capture_samples and time.time() < deadline:
            time.sleep(0.02)
        samples = self._wrench_buf.samples(self.capture_samples)
        if len(samples) < 10:
            self.get_logger().error("Not enough wrench samples")
            return False
        # the net_ft hardware interface holds its last value when UDP packets stop, and the
        # broadcaster keeps publishing it at full rate; real sensor data is never constant
        if np.all(np.ptp(samples, axis=0) == 0.0):
            self.get_logger().error(
                "Wrench is frozen (all samples identical): sensor stream dropped, pose not captured. "
                "Check the driver log for 'No F/T packets received'.")
            return False
        wrench_avg = samples.mean(axis=0)
        wrench_std = samples.std(axis=0)

        R_sw = self._get_sensor_rotation()
        if R_sw is None:
            return False

        self._captured_wrenches.append(wrench_avg)
        self._captured_rotations.append(R_sw)

        n = len(self._captured_wrenches)
        self.get_logger().info(
            f"Captured pose {n}: "
            f"F=[{wrench_avg[0]:.2f}, {wrench_avg[1]:.2f}, {wrench_avg[2]:.2f}] N  "
            f"T=[{wrench_avg[3]:.4f}, {wrench_avg[4]:.4f}, {wrench_avg[5]:.4f}] Nm  "
            f"std F=[{wrench_std[0]:.2f}, {wrench_std[1]:.2f}, {wrench_std[2]:.2f}] N"
        )
        if np.any(wrench_std[:3] > 0.5):
            self.get_logger().warn(
                "Force std > 0.5 N while standing still: something (cable?) is moving the tool")
        return True

    def _save_joint_pose(self):
        state = self.robot.get_state()
        q = state['q']
        names = state['joint_names']
        _UR_SUFFIXES = (
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        )
        arm_indices = [i for i, n in enumerate(names) if any(n.endswith(s) for s in _UR_SUFFIXES)]
        arm_q = [q[i] for i in arm_indices]

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
                # the background spinner thread (started in run()) already keeps wrench/TF
                # data fresh; spinning the same node here too races with it and raises
                # "Executor is already spinning".
                time.sleep(self.settle_time)
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
        self.get_logger().info("  Per-pose residuals (|F| N, |T| Nm):")
        for i, (rf, rt) in enumerate(zip(self._result['pose_force_residuals'],
                                         self._result['pose_torque_residuals'])):
            self.get_logger().info(f"    pose {i + 1:2d}: {rf:6.3f}  {rt:7.4f}")
        self._log_rtde_params()

        if self._result['condition_number'] > 50:
            self.get_logger().warn(
                "High condition number! Poses lack orientation diversity. "
                "Add more poses with different tool orientations."
            )
        if self._result['mass'] < 0.01:
            self.get_logger().warn("Estimated mass is very low. Check sensor data.")

    def _log_rtde_params(self):
        if self._result is None or self.robot_payload is None:
            return
        R_fs = self._get_flange_sensor_rotation()
        if R_fs is None:
            return
        try:
            sensor_mass, sensor_cog = split_payload(
                self.robot_payload['mass'], self.robot_payload['cog'],
                self._result['mass'], np.array(self._result['cog']),
                self.measuring_offset, R_fs,
            )
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        off = self.measuring_offset
        self.get_logger().info(
            f"  RTDE: ft_rtde_input_enable(True, {sensor_mass:.4f}, "
            f"[{off[0]:.4f}, {off[1]:.4f}, {off[2]:.4f}], "
            f"[{sensor_cog[0]:.4f}, {sensor_cog[1]:.4f}, {sensor_cog[2]:.4f}])"
        )

    def _measuring_offset_test(self):
        if self._ur_wrench_buf.count() < 10:
            self.get_logger().error("No data on ur_wrench_topic; is the UR built-in sensor published?")
            return
        R_fs = self._get_flange_sensor_rotation()
        if R_fs is None:
            return

        input("Hands off the tool, then press Enter to capture the baseline... ")
        base_ext = self._wrench_buf.average(self.capture_samples)
        base_ur = self._ur_wrench_buf.average(self.capture_samples)
        if base_ext is None or base_ur is None:
            self.get_logger().error("Not enough samples for the baseline")
            return

        d_ext, d_ur = [], []
        print("Push or twist the tool and hold it; press Enter to capture while holding.")
        print("Vary direction and point of attack. 'f' = fit, 'x' = abort.")
        while True:
            line = input(f"[{len(d_ext)} pushes] > ").strip().lower()
            if line == 'x':
                return
            if line == 'f':
                break
            ext = self._wrench_buf.average(self.push_samples)
            ur = self._ur_wrench_buf.average(self.push_samples)
            if ext is None or ur is None:
                self.get_logger().error("Not enough samples")
                continue
            d_ext.append(ext - base_ext)
            d_ur.append(ur - base_ur)
            de = d_ext[-1]
            print(f"  dF_ext=[{de[0]:.1f}, {de[1]:.1f}, {de[2]:.1f}] N  "
                  f"dT_ext=[{de[3]:.2f}, {de[4]:.2f}, {de[5]:.2f}] Nm")

        if len(d_ext) < 3:
            self.get_logger().error("Need at least 3 pushes in different directions")
            return
        d_ext, d_ur = np.array(d_ext), np.array(d_ur)
        res = estimate_measuring_offset(d_ext[:, :3], d_ext[:, 3:], d_ur[:, :3], d_ur[:, 3:], R_fs)
        r = res['offset']
        self.get_logger().info("=== Measuring offset (flange frame) ===")
        self.get_logger().info(f"  offset:            [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}] m")
        self.get_logger().info(f"  UR sign:           {res['sign']:+.0f}")
        self.get_logger().info(f"  force agreement:   {res['force_agreement']:.3f} (relative error, small is good)")
        self.get_logger().info(f"  torque residual:   {res['residual_norm']:.4f} Nm")
        self.get_logger().info(f"  condition number:  {res['condition_number']:.1f}")
        if res['force_agreement'] > 0.3:
            self.get_logger().warn(
                "Sensors disagree on the pushes; check frames and that RTDE injection is off."
            )
        self.measuring_offset = r
        self._log_rtde_params()

    def _save_result(self):
        if self._result is None:
            self.get_logger().error("No result to save. Press 'r' first.")
            return

        path = save_payload_params(
            self._result, self.output_file,
            sensor_frame=self.sensor_frame,
            world_frame=self.world_frame,
            robot_payload=self.robot_payload,
            measuring_offset=self.measuring_offset,
        )
        self.get_logger().info(f"Saved payload params to {path}")

    def _print_help(self):
        self.get_logger().info("--- Commands ---")
        self.get_logger().info("  c  Capture current pose")
        self.get_logger().info("  s  Save current joint pose to file")
        self.get_logger().info("  a  Auto-capture all saved joint poses")
        self.get_logger().info("  r  Run identification")
        self.get_logger().info("  o  Measuring-offset push test")
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
                self.get_logger().info(f"Received command '{cmd}'")
                try:
                    if cmd == 'c':
                        self._capture_pose()
                    elif cmd == 's':
                        self._save_joint_pose()
                    elif cmd == 'a':
                        self._auto_capture()
                    elif cmd == 'r':
                        self._run_identification()
                    elif cmd == 'o':
                        self._measuring_offset_test()
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
                except Exception as e:
                    # surfaces otherwise-silent failures (e.g. TF/move_group not reachable)
                    self.get_logger().error(f"Command '{cmd}' raised an exception: {e}")

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
