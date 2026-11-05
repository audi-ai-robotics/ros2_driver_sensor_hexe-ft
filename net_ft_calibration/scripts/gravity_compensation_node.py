#!/usr/bin/env python3
"""
Gravity compensation node for network F/T sensors.

Subscribes to raw wrench data, subtracts the gravity of the identified payload, the sensor bias
and a runtime tare, and publishes the compensated wrench. Provides a ~/tare service for runtime
re-zeroing (e.g. after grasping).

Optionally (output_frame set) the result is rotated into output_frame and its torques are moved
from the sensor measuring point (payload_params.yaml measuring_offset, flange frame) to the TCP
(tcp_offset). With output_frame=ur16e_tool0 this reproduces what the UR controller reports on
force_torque_sensor_broadcaster/wrench when the HEX-E feeds force_mode via RTDE, but it also works
with ft_force_mode_via_rtde:=false (hand-guiding through a handle on the built-in sensor while
recording the HEX-E). start_robot.launch.py starts it as hex_e_compensation.

The payload_profile parameter can be changed at runtime (PayloadProfileSwitcher does this); it
reloads mass/CoG/bias and resets the tare.

This node is not in any control loop; it only feeds recording/logging.

Usage:
  ros2 run net_ft_calibration gravity_compensation_node.py \
      --ros-args -p payload_params_file:=/tmp/ft_calibration/payload_params.yaml
"""

import collections
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger

from net_ft_calibration import compute_gravity_wrench, load_payload_params, transform_wrench


class GravityCompensationNode(Node):
    def __init__(self):
        super().__init__('gravity_compensation')

        self.declare_parameter('payload_params_file', '')
        self.declare_parameter('payload_profile', '')
        self.declare_parameter('sensor_frame', 'ur16e_tool0')
        self.declare_parameter('world_frame', 'ur16e_base_link')
        # '' = publish in the sensor frame with torques about the measuring point (legacy behavior)
        self.declare_parameter('output_frame', '')
        # TCP in output_frame coordinates; torques are taken about it (UR convention)
        self.declare_parameter('tcp_offset', [0.0, 0.0, 0.0])
        self.declare_parameter('wrench_topic', 'force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('compensated_topic', 'ft_compensated')
        self.declare_parameter('tare_samples', 100)
        self.declare_parameter('max_sample_age', 0.5)
        self.declare_parameter('frozen_samples', 50)
        self.declare_parameter('gravity', 9.81)

        params_file = self.get_parameter('payload_params_file').value
        if not params_file:
            pkg_share = get_package_share_directory('net_ft_calibration')
            params_file = str(Path(pkg_share) / 'config' / 'payload_params.yaml')
        self._params_file = params_file
        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.output_frame = self.get_parameter('output_frame').value
        self._tcp_offset = np.asarray(self.get_parameter('tcp_offset').value, dtype=float)
        wrench_topic = self.get_parameter('wrench_topic').value
        compensated_topic = self.get_parameter('compensated_topic').value
        self.tare_n = int(self.get_parameter('tare_samples').value)
        self.max_age = float(self.get_parameter('max_sample_age').value)
        self.frozen_n = int(self.get_parameter('frozen_samples').value)
        self.gravity = self.get_parameter('gravity').value

        self.get_logger().info(f"Loading payload params from: {self._params_file}")
        try:
            self._load_profile(self.get_parameter('payload_profile').value)
        except Exception as e:
            self.get_logger().error(f"Failed to load payload params: {e}")
            raise SystemExit(1)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._R_out_sensor = None  # static, looked up once

        self._tare_offset = np.zeros(6)
        self._recent = collections.deque(maxlen=self.tare_n)  # compensated, for ~/tare
        self._last_stamp = 0.0
        self._last_raw = None
        self._same_count = 0

        self.add_on_set_parameters_callback(self._on_set_parameters)
        # reliable, so sync_node and rosbag (default QoS) receive it
        self._pub = self.create_publisher(WrenchStamped, compensated_topic, 10)
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, qos_profile_sensor_data)
        self.create_service(Trigger, '~/tare', self._tare_srv)

        out = self.output_frame or f'{self.sensor_frame} (input frame)'
        self.get_logger().info(
            f"Gravity compensation active: {wrench_topic} -> {compensated_topic}, output frame {out}"
        )

    def _load_profile(self, profile: str) -> None:
        params = load_payload_params(self._params_file, profile or None)
        self._mass = params['mass']
        self._cog = params['cog']
        self._bias = params['bias']
        offset = params['measuring_offset']
        if self.output_frame and offset is None:
            self.get_logger().warn('payload params have no measuring_offset; torques stay about '
                                   'the sensor origin')
            offset = np.zeros(3)
        self._lever = (np.zeros(3) if offset is None else np.asarray(offset)) - self._tcp_offset
        self.get_logger().info(
            f"Profile '{params['profile']}': mass={self._mass:.3f} kg, "
            f"cog=[{self._cog[0]:.4f}, {self._cog[1]:.4f}, {self._cog[2]:.4f}] m, "
            f"bias=[{', '.join(f'{b:.3f}' for b in self._bias)}]"
        )

    def _on_set_parameters(self, params):
        for p in params:
            if p.name == 'payload_profile':
                try:
                    self._load_profile(p.value)
                except (OSError, KeyError, ValueError) as e:
                    return SetParametersResult(successful=False, reason=str(e))
                self._tare_offset = np.zeros(6)
                self._recent.clear()
                self.get_logger().info('Tare reset (profile changed)')
        return SetParametersResult(successful=True)

    def _lookup_rotation(self, target: str, source: str) -> np.ndarray | None:
        try:
            tf = self._tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        r = tf.transform.rotation
        return R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()

    def _is_frozen(self, raw: np.ndarray) -> bool:
        # net_ft_driver repeats its last value when UDP packets stop; real data is never constant
        if self._last_raw is not None and np.array_equal(raw, self._last_raw):
            self._same_count += 1
        else:
            self._same_count = 0
        self._last_raw = raw
        return self.frozen_n > 0 and self._same_count >= self.frozen_n

    def _wrench_cb(self, msg: WrenchStamped):
        raw = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])
        if self._is_frozen(raw):
            self.get_logger().warn(
                f'Input wrench frozen ({self._same_count} identical samples); not publishing',
                throttle_duration_sec=2.0)
            return

        R_sw = self._lookup_rotation(self.sensor_frame, self.world_frame)
        if R_sw is None:
            self.get_logger().warn(f'No TF {self.world_frame} -> {self.sensor_frame}',
                                   throttle_duration_sec=5.0)
            return

        gravity_w = compute_gravity_wrench(R_sw, self._mass, self._cog, self.gravity)
        compensated = raw - gravity_w - self._bias

        frame_id = msg.header.frame_id
        if self.output_frame:
            if self._R_out_sensor is None:
                self._R_out_sensor = self._lookup_rotation(self.output_frame, self.sensor_frame)
                if self._R_out_sensor is None:
                    self.get_logger().warn(f'No TF {self.sensor_frame} -> {self.output_frame}',
                                           throttle_duration_sec=5.0)
                    return
            compensated = transform_wrench(compensated, self._R_out_sensor, self._lever)
            frame_id = self.output_frame

        compensated = compensated - self._tare_offset
        self._recent.append(compensated)
        self._last_stamp = time.monotonic()

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = frame_id
        out.wrench.force.x, out.wrench.force.y, out.wrench.force.z = (float(v) for v in compensated[:3])
        out.wrench.torque.x, out.wrench.torque.y, out.wrench.torque.z = (float(v) for v in compensated[3:])
        self._pub.publish(out)

    def _tare_srv(self, request, response):
        # Synchronous: averages the last tare_samples outputs, so the caller gets a real result.
        age = time.monotonic() - self._last_stamp
        if len(self._recent) < self.tare_n or age > self.max_age:
            response.success = False
            response.message = (f'not enough fresh samples ({len(self._recent)}/{self.tare_n}, '
                                f'last {age:.2f} s ago); is the sensor streaming and TF available?')
            self.get_logger().error(f'Tare failed: {response.message}')
            return response
        samples = np.array(self._recent)
        mean = samples.mean(axis=0)
        self._tare_offset = self._tare_offset + mean
        self._recent = collections.deque(samples - mean, maxlen=self.tare_n)
        std = samples[:, :3].std(axis=0)
        response.success = True
        response.message = (f'tared F=[{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}] N, '
                            f'T=[{mean[3]:.4f}, {mean[4]:.4f}, {mean[5]:.4f}] Nm '
                            f'(force std [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}] N)')
        self.get_logger().info(f'Tare: {response.message}')
        return response


def main():
    rclpy.init()
    node = GravityCompensationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
