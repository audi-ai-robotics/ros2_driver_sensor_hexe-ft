"""Switch the active payload profile (e.g. gripper -> gripper with grasped part) at runtime.

With the HEX-E streamed into UR force_mode via RTDE, gravity compensation is split: the
external_ft_rtde_streamer subtracts sensor_bias and the UR controller subtracts gravity of
(installation payload - ft_rtde_input_enable sensor part). A profile switch therefore sets the
robot payload (io_and_status_controller/set_payload) and the streamer's sensor_bias. The sensor
part stays as configured at launch from the default profile. Only switch while force_mode is
inactive, otherwise the compensation steps under an active controller.

It also sets payload_profile on the ROS-side HEX-E compensation node (hex_e_compensation, started
by start_robot.launch.py with use_ft_sensor:=true), whose output is what gets recorded.
"""

from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from ur_msgs.srv import SetPayload

from .payload_identification import load_payload_params


class PayloadProfileSwitcher:
    def __init__(self, node, robot_ns: str = '', params_file: str = '',
                 streamer_node: str = 'external_ft_rtde_streamer',
                 compensation_node: str = 'hex_e_compensation') -> None:
        self.node = node
        self.params_file = params_file or str(
            Path(get_package_share_directory('net_ft_calibration')) / 'config' / 'payload_params.yaml'
        )
        pre = f'/{robot_ns}' if robot_ns else ''
        self.payload_cli = node.create_client(SetPayload, f'{pre}/io_and_status_controller/set_payload')
        self.streamer_cli = node.create_client(SetParameters, f'{pre}/{streamer_node}/set_parameters')
        self.compensation_cli = node.create_client(SetParameters, f'{pre}/{compensation_node}/set_parameters')

    def _call(self, cli, req, timeout: float = 5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        return fut.result()

    def apply(self, profile: str, wait: float = 5.0) -> bool:
        log = self.node.get_logger()
        try:
            params = load_payload_params(self.params_file, profile)
        except (OSError, KeyError) as e:
            log.error(f'Cannot load payload profile: {e}')
            return False
        payload = params['robot_payload']
        if payload is None:
            log.error(f"Profile '{profile}' has no robot_payload (needs robot_payload and "
                      f'measuring_offset in {self.params_file})')
            return False

        if not self.payload_cli.wait_for_service(timeout_sec=wait):
            log.error(f'Service {self.payload_cli.srv_name} unavailable')
            return False
        req = SetPayload.Request()
        req.mass = float(payload['mass'])
        req.center_of_gravity.x, req.center_of_gravity.y, req.center_of_gravity.z = (float(c) for c in payload['cog'])
        res = self._call(self.payload_cli, req)
        if res is None or not res.success:
            log.error('set_payload failed (is the external_control program running?)')
            return False

        bias = [float(b) for b in params['bias']]
        if self.streamer_cli.wait_for_service(timeout_sec=wait):
            req = SetParameters.Request()
            req.parameters = [Parameter('sensor_bias', Parameter.Type.DOUBLE_ARRAY, bias).to_parameter_msg()]
            res = self._call(self.streamer_cli, req)
            if res is None or not all(r.successful for r in res.results):
                log.error('Setting sensor_bias on the RTDE streamer failed')
                return False
        else:
            log.warn(f'{self.streamer_cli.srv_name} unavailable (ft_force_mode_via_rtde:=false?); '
                     'only the robot payload was set')

        if self.compensation_cli.wait_for_service(timeout_sec=wait):
            req = SetParameters.Request()
            req.parameters = [Parameter('payload_profile', Parameter.Type.STRING, profile).to_parameter_msg()]
            res = self._call(self.compensation_cli, req)
            if res is None or not all(r.successful for r in res.results):
                reason = '; '.join(r.reason for r in res.results) if res else 'no response'
                log.error(f'Setting payload_profile on the HEX-E compensation failed: {reason}')
                return False
        else:
            log.warn(f'{self.compensation_cli.srv_name} unavailable (use_ft_sensor:=false?); '
                     'recorded HEX-E wrench keeps its old profile')

        cog = ', '.join(f'{c:.4f}' for c in payload['cog'])
        log.info(f"Payload profile '{profile}': robot payload {payload['mass']:.3f} kg, CoG [{cog}] m, "
                 f"sensor_bias [{', '.join(f'{b:.4f}' for b in bias)}]")
        return True
