#!/usr/bin/env python3
"""
Re-expresses a wrench measured at the F/T sensor frame at a different frame's origin (e.g. TCP).

Applies the standard wrench/screw transform: rotate force and torque into the
target frame, then add the lever-arm correction torque = r x F, where r is the
sensor origin's position expressed in the target frame. tf2_geometry_msgs has
no do_transform_wrench helper, so this is done manually via scipy.

Usage:
  ros2 run net_ft_calibration wrench_frame_transform_node.py \
      --ros-args -p sensor_frame:=hex_e_fts_link -p tcp_frame:=tcp
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R

import tf2_ros


class WrenchFrameTransformNode(Node):
    def __init__(self):
        super().__init__('wrench_frame_transform')

        self.declare_parameter('sensor_frame', 'hex_e_fts_link')
        self.declare_parameter('tcp_frame', 'tcp')
        self.declare_parameter('input_topic', 'ft_compensated')
        self.declare_parameter('output_topic', 'ft_compensated_tcp')

        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.tcp_frame = self.get_parameter('tcp_frame').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(WrenchStamped, output_topic, 10)
        self.create_subscription(WrenchStamped, input_topic, self._wrench_cb, 10)

        self.get_logger().info(
            f"Wrench frame transform active: {self.sensor_frame} -> {self.tcp_frame} "
            f"({input_topic} -> {output_topic})"
        )

    def _lookup_sensor_in_tcp(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                self.tcp_frame, self.sensor_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ):
            return None, None
        q = tf.transform.rotation
        t = tf.transform.translation
        rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        translation = np.array([t.x, t.y, t.z])
        return rotation, translation

    def _wrench_cb(self, msg: WrenchStamped):
        rotation, translation = self._lookup_sensor_in_tcp()
        if rotation is None:
            return

        force_sensor = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        torque_sensor = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        force_tcp = rotation @ force_sensor
        # lever-arm correction for the translation between sensor origin and TCP origin
        torque_tcp = rotation @ torque_sensor + np.cross(translation, force_tcp)

        out = WrenchStamped()
        out.header = msg.header
        out.header.frame_id = self.tcp_frame
        out.wrench.force.x, out.wrench.force.y, out.wrench.force.z = force_tcp.tolist()
        out.wrench.torque.x, out.wrench.torque.y, out.wrench.torque.z = torque_tcp.tolist()
        self._pub.publish(out)


def main():
    rclpy.init()
    node = WrenchFrameTransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
