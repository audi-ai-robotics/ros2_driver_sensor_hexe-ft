from .payload_identification import (
    build_gravity_regressor,
    identify_payload,
    compute_gravity_wrench,
    split_payload,
    combine_payload,
    estimate_measuring_offset,
    rtde_sensor_params,
    save_payload_params,
    load_payload_params,
    list_payload_profiles,
    DEFAULT_PROFILE,
    rotate_sensor_frame,
    fit_sensor_rotation,
    pose_drift,
    save_captures,
    load_captures,
)

__all__ = [
    'build_gravity_regressor',
    'identify_payload',
    'compute_gravity_wrench',
    'split_payload',
    'combine_payload',
    'estimate_measuring_offset',
    'rtde_sensor_params',
    'save_payload_params',
    'load_payload_params',
    'list_payload_profiles',
    'DEFAULT_PROFILE',
    'rotate_sensor_frame',
    'fit_sensor_rotation',
    'pose_drift',
    'save_captures',
    'load_captures',
    'RobotController',
    'PayloadProfileSwitcher',
    'zero_wrench',
]


def __getattr__(name):
    # Lazy so launch files and tests can use the math without pulling in rclpy/MoveIt.
    if name == 'RobotController':
        from .robot import RobotController
        return RobotController
    if name == 'PayloadProfileSwitcher':
        from .profile_switcher import PayloadProfileSwitcher
        return PayloadProfileSwitcher
    if name == 'zero_wrench':
        from .wrench_zeroing import zero_wrench
        return zero_wrench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
