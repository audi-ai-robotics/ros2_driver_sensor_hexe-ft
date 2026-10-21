#!/usr/bin/env python3
"""
Live comparison of the calibrated HEX-E wrench against the UR built-in F/T sensor, both in tool0.

Top row: HEX-E (gravity-compensated, transformed to tool0). Bottom row: UR built-in sensor.
Columns: Fx Fy Fz Tx Ty Tz. Each column shares its y-axis between the rows.
The Record button writes every received sample (long format: time_s, sensor, fx..tz) to a CSV in log_dir.

Usage (HEX-E pipeline from 01_Commands.md must be running):
  python3 ft_compare_gui.py --ros-args -p ns:=Peach -r /tf:=/Peach/tf -r /tf_static:=/Peach/tf_static
"""

import csv
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import WrenchStamped
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

LABELS = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]']
ROW_NAMES = ['HEX-E (compensated)', 'UR built-in']
COLORS = ['tab:blue', 'tab:red']
SENSOR_NAMES = ['hexe', 'ur']


class WrenchCollector(Node):
    def __init__(self):
        super().__init__('ft_compare_gui')
        self.declare_parameter('ns', '')
        self.declare_parameter('hexe_topic', 'ft_compensated_tool0')
        self.declare_parameter('ur_topic', 'force_torque_sensor_broadcaster/wrench')
        self.declare_parameter('tool_frame', 'ur16e_tool0')
        # Use if the UR driver labels its wrench with a frame that does not match the data.
        self.declare_parameter('ur_frame_override', '')
        self.declare_parameter('window_s', 10.0)
        self.declare_parameter('log_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))

        ns = self.get_parameter('ns').value.strip('/')
        prefix = f'/{ns}/' if ns else ''
        self.tool_frame = self.get_parameter('tool_frame').value
        self.ur_frame_override = self.get_parameter('ur_frame_override').value
        self.window_s = float(self.get_parameter('window_s').value)

        self.log_dir = self.get_parameter('log_dir').value

        self.lock = threading.Lock()
        self.buffers = [deque(), deque()]  # per row: (t, [fx..tz])
        self._csv_file = None
        self._csv_writer = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        hexe_topic = prefix + self.get_parameter('hexe_topic').value
        ur_topic = prefix + self.get_parameter('ur_topic').value
        self.create_subscription(WrenchStamped, hexe_topic, lambda m: self._cb(0, m, ''), 50)
        self.create_subscription(
            WrenchStamped, ur_topic, lambda m: self._cb(1, m, self.ur_frame_override), 50
        )
        self.get_logger().info(f'HEX-E: {hexe_topic} | UR: {ur_topic} | tool frame: {self.tool_frame}')

    def _to_tool_frame(self, msg, frame_override):
        f = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        t = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        src = frame_override or msg.header.frame_id
        if not src or src == self.tool_frame:
            return np.concatenate([f, t])
        try:
            tf = self._tf_buffer.lookup_transform(self.tool_frame, src, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'TF {src} -> {self.tool_frame} unavailable: {e}', throttle_duration_sec=2.0)
            return None
        q, p = tf.transform.rotation, tf.transform.translation
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        f_t = rot @ f
        t_t = rot @ t + np.cross([p.x, p.y, p.z], f_t)  # lever arm to the tool origin
        return np.concatenate([f_t, t_t])

    def _cb(self, row, msg, frame_override):
        w = self._to_tool_frame(msg, frame_override)
        if w is None:
            return
        now = time.monotonic()
        with self.lock:
            buf = self.buffers[row]
            buf.append((now, w))
            while buf and now - buf[0][0] > self.window_s:
                buf.popleft()
            if self._csv_writer is not None:
                self._csv_writer.writerow([f'{time.time():.6f}', SENSOR_NAMES[row], *[f'{v:.6f}' for v in w]])

    def start_recording(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f'ft_compare_{datetime.now():%Y%m%d_%H%M%S}.csv')
        with self.lock:
            self._csv_file = open(path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['time_s', 'sensor', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
        self.get_logger().info(f'Recording to {path}')
        return path

    def stop_recording(self):
        with self.lock:
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self.get_logger().info('Recording stopped')

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            out = []
            for buf in self.buffers:
                if buf:
                    t = np.array([s[0] for s in buf]) - now
                    w = np.array([s[1] for s in buf])
                else:
                    t, w = np.empty(0), np.empty((0, 6))
                out.append((t, w))
        return out


def main():
    rclpy.init(args=sys.argv)
    node = WrenchCollector()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    window_s = node.window_s

    fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharex=True, sharey='col')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17, wspace=0.3, hspace=0.12)
    lines, stats = [], []
    for r in range(2):
        for c in range(6):
            ax = axes[r, c]
            (line,) = ax.plot([], [], color=COLORS[r], lw=1)
            lines.append(line)
            stats.append(ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=8,
                                 bbox=dict(fc='white', alpha=0.7, lw=0)))
            ax.set_xlim(-window_s, 0)
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(LABELS[c])
            else:
                ax.set_xlabel('t [s]')
        axes[r, 0].set_ylabel(ROW_NAMES[r])

    state = {'paused': False}
    btn_pause = Button(fig.add_axes([0.06, 0.03, 0.08, 0.05]), 'Pause')
    btn_clear = Button(fig.add_axes([0.15, 0.03, 0.08, 0.05]), 'Clear')
    btn_rec = Button(fig.add_axes([0.24, 0.03, 0.10, 0.05]), 'Record')
    rec_text = fig.text(0.36, 0.055, '', va='center', fontsize=9)

    def toggle_pause(_):
        state['paused'] = not state['paused']
        btn_pause.label.set_text('Resume' if state['paused'] else 'Pause')

    def clear(_):
        with node.lock:
            for b in node.buffers:
                b.clear()

    def toggle_record(_):
        if node._csv_writer is None:
            rec_text.set_text(f'recording: {node.start_recording()}')
            btn_rec.label.set_text('Stop')
        else:
            node.stop_recording()
            rec_text.set_text('saved')
            btn_rec.label.set_text('Record')
        fig.canvas.draw_idle()

    btn_pause.on_clicked(toggle_pause)
    btn_clear.on_clicked(clear)
    btn_rec.on_clicked(toggle_record)

    def update(_):
        if state['paused']:
            return []
        for r, (t, w) in enumerate(node.snapshot()):
            for c in range(6):
                i = r * 6 + c
                if len(t) == 0:
                    lines[i].set_data([], [])
                    stats[i].set_text('no data')
                    continue
                lines[i].set_data(t, w[:, c])
                stats[i].set_text(f'mean {w[:, c].mean():+.2f}\nstd {w[:, c].std():.2f}')
        for c in range(6):
            axes[0, c].relim()
            axes[0, c].autoscale_view(scalex=False)
            axes[1, c].relim()
            axes[1, c].autoscale_view(scalex=False)
        return lines + stats

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        del anim
        node.stop_recording()
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
