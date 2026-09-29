#!/usr/bin/env python3
"""
Gravity compensation node for network F/T sensors.

Subscribes to raw wrench data, subtracts gravitational load using identified
payload parameters, and publishes compensated wrench. Provides a ~/tare service
for runtime re-zeroing (e.g. after gripper state change).

Assumes the robot driver is running (TF available) and the F/T sensor is publishing.

Usage:
  ros2 run net_ft_calibration gravity_compensation_node.py \
      --ros-args -p payload_params_file:=/tmp/ft_calibration/payload_params.yaml
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import tf2_ros

from net_ft_calibration import compute_gravity_wrench, load_payload_params


class GravityCompensationNode(Node):
    def __init__(self):
        super().__init__('gravity_compensation')

        self.declare_parameter('payload_params_file', '')
        self.declare_parameter('sensor_frame', 'ur16e_tool0')
        self.declare_parameter('world_frame', 'ur16e_base_link')
        self.declare_parameter('wrench_topic', 'force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('compensated_topic', 'ft_compensated')
        self.declare_parameter('tare_samples', 100)
        self.declare_parameter('gravity', 9.81)

        params_file = self.get_parameter('payload_params_file').value
        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        wrench_topic = self.get_parameter('wrench_topic').value
        compensated_topic = self.get_parameter('compensated_topic').value
        self.tare_n = self.get_parameter('tare_samples').value
        self.gravity = self.get_parameter('gravity').value

        if not params_file:
            pkg_share = get_package_share_directory('net_ft_calibration')
            params_file = str(Path(pkg_share) / 'config' / 'payload_params.yaml')

        self.get_logger().info(f"Loading payload params from: {params_file}")
        try:
            params = load_payload_params(params_file)
        except Exception as e:
            self.get_logger().error(f"Failed to load payload params: {e}")
            raise SystemExit(1)

        self._mass = params['mass']
        self._cog = params['cog']
        self._bias = params['bias']

        self.get_logger().info(
            f"Payload: mass={self._mass:.3f} kg, "
            f"cog=[{self._cog[0]:.4f}, {self._cog[1]:.4f}, {self._cog[2]:.4f}] m"
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._tare_offset = np.zeros(6)
        self._tare_collecting = False
        self._tare_samples = []
        self._tare_lock = threading.Lock()

        self._pub = self.create_publisher(WrenchStamped, compensated_topic, 10)
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, 10)
        self.create_service(Trigger, '~/tare', self._tare_srv)

        self.get_logger().info(
            f"Gravity compensation active: {wrench_topic} -> {compensated_topic}"
        )

    def _get_sensor_rotation(self) -> np.ndarray | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self.sensor_frame, self.world_frame, rclpy.time.Time()
            )
            r = tf.transform.rotation
            return R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None

    def _wrench_cb(self, msg: WrenchStamped):
        R_sw = self._get_sensor_rotation()
        if R_sw is None:
            return

        raw = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])

        gravity_w = compute_gravity_wrench(R_sw, self._mass, self._cog, self.gravity)
        compensated = raw - gravity_w - self._bias - self._tare_offset

        with self._tare_lock:
            if self._tare_collecting:
                self._tare_samples.append(compensated.copy())
                if len(self._tare_samples) >= self.tare_n:
                    self._tare_offset += np.mean(self._tare_samples, axis=0)
                    self._tare_samples.clear()
                    self._tare_collecting = False
                    self.get_logger().info("Tare complete")

        out = WrenchStamped()
        out.header = msg.header
        out.wrench.force.x = float(compensated[0])
        out.wrench.force.y = float(compensated[1])
        out.wrench.force.z = float(compensated[2])
        out.wrench.torque.x = float(compensated[3])
        out.wrench.torque.y = float(compensated[4])
        out.wrench.torque.z = float(compensated[5])
        self._pub.publish(out)

    def _tare_srv(self, request, response):
        with self._tare_lock:
            self._tare_samples.clear()
            self._tare_collecting = True
        self.get_logger().info(f"Tare started, collecting {self.tare_n} samples...")
        response.success = True
        response.message = f"Collecting {self.tare_n} samples for tare"
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
