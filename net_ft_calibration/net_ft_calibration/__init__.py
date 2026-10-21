from .payload_identification import (
    build_gravity_regressor,
    identify_payload,
    compute_gravity_wrench,
    split_payload,
    estimate_measuring_offset,
    rtde_sensor_params,
    save_payload_params,
    load_payload_params,
)

__all__ = [
    'build_gravity_regressor',
    'identify_payload',
    'compute_gravity_wrench',
    'split_payload',
    'estimate_measuring_offset',
    'rtde_sensor_params',
    'save_payload_params',
    'load_payload_params',
    'RobotController',
]


def __getattr__(name):
    # Lazy so launch files and tests can use the math without pulling in rclpy/MoveIt.
    if name == 'RobotController':
        from .robot import RobotController
        return RobotController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
