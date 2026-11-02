"""Zero the wrench UR force_mode sees before starting it (robot still, hands off).

With the HEX-E streamed via RTDE (ft_force_mode_via_rtde:=true), io_and_status_controller/
zero_ftsensor has no effect on the external input, so this calls the streamer's ~/tare, which
checks for a live HEX-E, cancels the offset of the compensated UR wrench and verifies it. Without
the streamer, it falls back to zero_ftsensor (built-in sensor) and does the same checks here:
wait for a live, varying wrench, zero, then verify the mean is close to zero.

The caller's node must not be spun elsewhere (this spins it like spin_until_future_complete).
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger


def _sample_wrench(node, topic: str, duration: float) -> np.ndarray:
    samples = []

    def cb(m):
        w = m.wrench
        samples.append([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])

    sub = node.create_subscription(WrenchStamped, topic, cb, qos_profile_sensor_data)
    try:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        node.destroy_subscription(sub)
    return np.array(samples).reshape(-1, 6)


def _is_live(samples: np.ndarray) -> bool:
    # the net_ft driver repeats its last value when packets stop; real data is never constant
    return len(samples) >= 10 and bool(np.any(np.ptp(samples, axis=0) > 0.0))


def _call(node, cli, timeout: float):
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    return fut.result()


def zero_wrench(node, robot_ns: str = '', wrench_topic: str = 'force_torque_sensor_broadcaster/wrench',
                streamer_node: str = 'external_ft_rtde_streamer', live_timeout: float = 5.0,
                force_tolerance: float = 0.5, torque_tolerance: float = 0.1) -> bool:
    """Tare via the RTDE streamer if it runs, else zero_ftsensor with live/verify checks.

    Returns False (after logging an error) if the wrench is stale/frozen or not ~0 afterwards."""
    log = node.get_logger()
    pre = f'/{robot_ns}' if robot_ns else ''
    topic = wrench_topic if wrench_topic.startswith('/') else f'{pre}/{wrench_topic}'

    tare_cli = node.create_client(Trigger, f'{pre}/{streamer_node}/tare')
    zero_cli = node.create_client(Trigger, f'{pre}/io_and_status_controller/zero_ftsensor')
    try:
        if tare_cli.wait_for_service(timeout_sec=2.0):
            log.info('Taring the external F/T input (robot still, hands off) ...')
            res = _call(node, tare_cli, timeout=30.0)
            if res is None or not res.success:
                log.error(f"F/T tare FAILED: {res.message if res else 'no response'}")
                return False
            log.info(f'F/T tare OK: {res.message}')
            return True

        log.info(f'{tare_cli.srv_name} unavailable (ft_force_mode_via_rtde:=false?); using zero_ftsensor')
        deadline = time.monotonic() + live_timeout
        samples = _sample_wrench(node, topic, 0.5)
        while not _is_live(samples) and time.monotonic() < deadline:
            samples = _sample_wrench(node, topic, 0.5)
        if not _is_live(samples):
            log.error(f'F/T zeroing FAILED: {topic} is not live ({len(samples)} samples in 0.5 s, '
                      'or all identical = frozen)')
            return False
        before = samples.mean(axis=0)

        if not zero_cli.wait_for_service(timeout_sec=5.0):
            log.error(f'F/T zeroing FAILED: {zero_cli.srv_name} unavailable')
            return False
        res = _call(node, zero_cli, timeout=10.0)
        if res is None or not res.success:
            log.error(f"F/T zeroing FAILED: zero_ftsensor: {res.message if res else 'no response'}")
            return False

        _sample_wrench(node, topic, 0.3)  # let the zeroed value reach the topic
        after = _sample_wrench(node, topic, 0.5)
        if not _is_live(after):
            log.error(f'F/T zeroing FAILED: {topic} stopped being live after zero_ftsensor')
            return False
        mean = after.mean(axis=0)
        f, t = np.linalg.norm(mean[:3]), np.linalg.norm(mean[3:])
        np.set_printoptions(precision=3, suppress=True)
        if f > force_tolerance or t > torque_tolerance:
            log.error(f'F/T zeroing FAILED: wrench not ~0 after zero_ftsensor: F={mean[:3]} N, T={mean[3:]} Nm '
                      f'(before: F={before[:3]} N); limits {force_tolerance} N / {torque_tolerance} Nm')
            return False
        log.info(f'F/T zeroed: |F| {f:.3f} N, |T| {t:.3f} Nm (before: F={before[:3]} N)')
        return True
    finally:
        node.destroy_client(tare_cli)
        node.destroy_client(zero_cli)
