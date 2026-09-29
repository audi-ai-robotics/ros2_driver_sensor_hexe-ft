"""ROS 2 / MoveIt 2 robot interface for hand-eye calibration."""

import os
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from scipy.spatial.transform import Rotation as R

import tf2_ros


class RobotController:
    """
    ROS 2 robot controller using MoveIt 2 and TF2.

    Parameters
    ----------
    node : rclpy.node.Node
        ROS 2 node to create clients/subscriptions on.
    base_frame : str
        Robot base frame (e.g. 'ur16e_base_link').
    ee_frame : str
        End-effector frame (e.g. 'ur16e_tool0').
    move_group_name : str
        MoveIt planning group name.
    velocity_scaling : float
        Max velocity scaling factor (0.0 to 1.0).
    acceleration_scaling : float
        Max acceleration scaling factor (0.0 to 1.0).
    moveit_config_package : str
        MoveIt config package name (for go_home SRDF lookup).
    """

    def __init__(
        self,
        node: Node,
        base_frame: str = 'ur16e_base_link',
        ee_frame: str = 'ur16e_tool0',
        move_group_name: str = 'ur_arm',
        velocity_scaling: float = 0.1,
        acceleration_scaling: float = 0.1,
        robot_ns: str = '',
        moveit_config_package: str = 'ur16e_cell_moveit_config',
    ):
        self._node = node
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.move_group_name = move_group_name
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling
        self._moveit_config_package = moveit_config_package

        ns_prefix = f'/{robot_ns}' if robot_ns else ''
        self._move_client = ActionClient(node, MoveGroup, f'{ns_prefix}/move_action')

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        self._joint_names = None
        self._joint_positions = None
        self._joint_lock = threading.Lock()
        self._joint_sub = node.create_subscription(
            JointState, f'{ns_prefix}/joint_states', self._joint_state_cb, 1
        )

    def _joint_state_cb(self, msg: JointState):
        with self._joint_lock:
            self._joint_names = list(msg.name)
            self._joint_positions = list(msg.position)

    def get_state(self) -> dict:
        """
        Get current robot state via TF2 and joint states.

        Returns
        -------
        dict
            Dictionary with:
            - 'q': list of joint positions
            - 'O_T_EE': 4x4 end-effector pose in base frame
            - 'position': [x, y, z] end-effector position
            - 'joint_names': list of joint names
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            raise RuntimeError(f"TF lookup failed ({self.base_frame} -> {self.ee_frame}): {e}")

        t = tf.transform.translation
        r = tf.transform.rotation

        rot = R.from_quat([r.x, r.y, r.z, r.w])
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]

        with self._joint_lock:
            q = list(self._joint_positions) if self._joint_positions else []
            names = list(self._joint_names) if self._joint_names else []

        return {
            'q': q,
            'O_T_EE': T,
            'position': [t.x, t.y, t.z],
            'joint_names': names,
        }

    def get_ee_pose_flat(self) -> list[float]:
        state = self.get_state()
        return state['O_T_EE'].flatten().tolist()

    def move_joints(self, joint_positions: list[float], joint_names: list[str] | None = None):
        """
        Move to joint positions via MoveIt.

        Parameters
        ----------
        joint_positions : list
            Joint angles in radians (6 for UR).
        joint_names : list, optional
            Joint names. If None, uses the names from the last joint_states message.
        """
        if joint_names is None:
            with self._joint_lock:
                joint_names = list(self._joint_names) if self._joint_names else None

        if joint_names is None:
            raise RuntimeError("No joint names available; is /joint_states publishing?")

        _UR_SUFFIXES = (
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        )
        arm_joints = [n for n in joint_names if any(n.endswith(s) for s in _UR_SUFFIXES)]
        if len(arm_joints) != len(joint_positions):
            raise ValueError(
                f"Expected {len(arm_joints)} joint values, got {len(joint_positions)}"
            )

        constraints = Constraints(
            joint_constraints=[
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.001,
                    tolerance_below=0.001,
                    weight=1.0,
                )
                for name, pos in zip(arm_joints, joint_positions)
            ]
        )
        self._send_move_goal(constraints, f"joint target", cartesian=False)

    def move_cartesian(self, position: list[float], quaternion_xyzw: list[float]):
        """
        Move to a Cartesian pose via MoveIt.

        Parameters
        ----------
        position : list
            [x, y, z] in meters, in base_frame.
        quaternion_xyzw : list
            [x, y, z, w] quaternion.
        """
        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.pose.position.x = position[0]
        target.pose.position.y = position[1]
        target.pose.position.z = position[2]
        target.pose.orientation.x = quaternion_xyzw[0]
        target.pose.orientation.y = quaternion_xyzw[1]
        target.pose.orientation.z = quaternion_xyzw[2]
        target.pose.orientation.w = quaternion_xyzw[3]

        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [0.001]

        position_constraint = PositionConstraint()
        position_constraint.header = target.header
        position_constraint.link_name = self.ee_frame
        position_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        position_constraint.constraint_region = BoundingVolume(
            primitives=[region],
            primitive_poses=[Pose(position=target.pose.position)],
        )
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = target.header
        orientation_constraint.link_name = self.ee_frame
        orientation_constraint.orientation = target.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        orientation_constraint.weight = 1.0

        constraints = Constraints(
            position_constraints=[position_constraint],
            orientation_constraints=[orientation_constraint],
        )
        self._send_move_goal(constraints, "cartesian target", cartesian=True)

    def go_home(self):
        """Move to the 'home' named state defined in the MoveIt SRDF."""
        try:
            pkg = get_package_share_directory(self._moveit_config_package)
            config_dir = os.path.join(pkg, 'config')
            srdf_files = [f for f in os.listdir(config_dir) if f.endswith('.srdf')]
            if not srdf_files:
                raise FileNotFoundError(f"No .srdf files in {config_dir}")

            home_values = None
            for srdf_name in srdf_files:
                srdf_path = os.path.join(config_dir, srdf_name)
                root = ET.parse(srdf_path).getroot()
                for gs in root.findall('group_state'):
                    if gs.get('name') == 'home' and gs.get('group') == self.move_group_name:
                        home_values = {
                            j.get('name'): float(j.get('value'))
                            for j in gs.findall('joint')
                        }
                        break
                if home_values:
                    break

            if not home_values:
                raise ValueError(
                    f"'home' state for group '{self.move_group_name}' "
                    f"not found in any SRDF in {config_dir}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load home pose from SRDF: {e}")

        constraints = Constraints(
            joint_constraints=[
                JointConstraint(
                    joint_name=name,
                    position=value,
                    tolerance_above=0.001,
                    tolerance_below=0.001,
                    weight=1.0,
                )
                for name, value in home_values.items()
            ]
        )
        self._send_move_goal(constraints, "'home' pose", cartesian=False)

    def _send_move_goal(
        self, goal_constraints: Constraints, description: str,
        *, cartesian: bool = False,
    ):
        if cartesian:
            strategies = [
                ("pilz_industrial_motion_planner", "LIN"),
                ("pilz_industrial_motion_planner", "PTP"),
                ("ompl", "RRTConnect"),
            ]
        else:
            strategies = [
                ("pilz_industrial_motion_planner", "PTP"),
                ("ompl", "RRTConnect"),
            ]

        last_err = None
        for pipeline, planner in strategies:
            try:
                self._plan_and_execute(
                    goal_constraints, description,
                    pipeline_id=pipeline, planner_id=planner,
                )
                return
            except RuntimeError as e:
                last_err = e
                self._node.get_logger().warn(
                    f"{pipeline}/{planner} failed for {description} ({e}); trying next"
                )
                time.sleep(0.5)
        raise RuntimeError(f"All planners failed for {description}: {last_err}")

    def _plan_and_execute(
        self, goal_constraints: Constraints, description: str,
        pipeline_id: str = "", planner_id: str = "",
    ):
        goal = MoveGroup.Goal()
        goal.request.pipeline_id = pipeline_id
        goal.request.planner_id = planner_id
        goal.request.group_name = self.move_group_name
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
        goal.request.goal_constraints = [goal_constraints]
        goal.planning_options.plan_only = False

        self._node.get_logger().info(f"Moving to {description} (pipeline={pipeline_id or 'default'})...")
        self._move_client.wait_for_server(timeout_sec=10.0)

        send_future = self._move_client.send_goal_async(goal)
        self._wait_for_future(send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            raise RuntimeError(f"MoveIt rejected goal: {description}")

        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future)
        result = result_future.result().result

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"MoveIt failed for {description} (error {result.error_code.val})"
            )

        self._node.get_logger().info(f"Reached {description}")

    @staticmethod
    def _wait_for_future(future, timeout_sec: float = 60.0):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for MoveIt action result")
            time.sleep(0.02)

    def wait_for_ready(self, timeout_sec: float = 10.0) -> bool:
        """Wait for TF and joint states to become available."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            try:
                self._tf_buffer.lookup_transform(
                    self.base_frame, self.ee_frame, rclpy.time.Time()
                )
                with self._joint_lock:
                    if self._joint_positions is not None:
                        return True
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass
        return False
