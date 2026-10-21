"""Least-squares payload identification and gravity compensation math."""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def build_gravity_regressor(R_sensor_world: np.ndarray, g: float = 9.81) -> np.ndarray:
    """
    Build the 6x10 gravity-wrench regressor for one pose.

    The 10 unknowns are [m, m*cx, m*cy, m*cz, bfx, bfy, bfz, btx, bty, btz].

    Parameters
    ----------
    R_sensor_world : (3, 3)
        Rotation from world frame to sensor frame.
    g : float
        Gravitational acceleration magnitude.

    Returns
    -------
    G : (6, 10) regressor matrix.
    """
    g_sensor = R_sensor_world @ np.array([0.0, 0.0, -g])
    gx, gy, gz = g_sensor

    G = np.zeros((6, 10))
    # Force rows: F = m * g_sensor + bias_f
    G[0, 0] = gx;  G[0, 4] = 1.0
    G[1, 0] = gy;  G[1, 5] = 1.0
    G[2, 0] = gz;  G[2, 6] = 1.0
    # Torque rows: T = cog x (m * g_sensor) + bias_t
    #   tx = m*cy*gz - m*cz*gy
    #   ty = m*cz*gx - m*cx*gz
    #   tz = m*cx*gy - m*cy*gx
    G[3, 2] = gz;   G[3, 3] = -gy;  G[3, 7] = 1.0
    G[4, 3] = gx;   G[4, 1] = -gz;  G[4, 8] = 1.0
    G[5, 1] = gy;   G[5, 2] = -gx;  G[5, 9] = 1.0
    return G


def identify_payload(
    wrenches: np.ndarray,
    rotations: list[np.ndarray],
    g: float = 9.81,
) -> dict:
    """
    Identify payload mass, center of gravity, and sensor bias.

    Parameters
    ----------
    wrenches : (N, 6)
        Measured wrench at each pose [fx, fy, fz, tx, ty, tz].
    rotations : list of (3, 3)
        R_sensor_world at each pose (rotation from world to sensor frame).
    g : float
        Gravity magnitude.

    Returns
    -------
    dict with keys: mass, cog, bias, residual_norm, condition_number, num_poses,
    pose_force_residuals, pose_torque_residuals (per-pose residual norms, N and Nm).
    """
    n = len(rotations)
    A = np.zeros((6 * n, 10))
    b = np.zeros(6 * n)

    for i, R_sw in enumerate(rotations):
        A[6 * i:6 * (i + 1), :] = build_gravity_regressor(R_sw, g)
        b[6 * i:6 * (i + 1)] = wrenches[i]

    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float('inf')

    mass = x[0]
    if abs(mass) < 1e-6:
        cog = np.zeros(3)
    else:
        cog = x[1:4] / mass

    residual = (A @ x - b).reshape(n, 6)
    residual_norm = float(np.linalg.norm(residual))

    return {
        'mass': float(mass),
        'cog': cog.tolist(),
        'bias': x[4:10].tolist(),
        'residual_norm': residual_norm,
        'condition_number': float(cond),
        'num_poses': n,
        'pose_force_residuals': np.linalg.norm(residual[:, :3], axis=1).tolist(),
        'pose_torque_residuals': np.linalg.norm(residual[:, 3:], axis=1).tolist(),
    }


def compute_gravity_wrench(
    R_sensor_world: np.ndarray,
    mass: float,
    cog: np.ndarray,
    g: float = 9.81,
) -> np.ndarray:
    """
    Compute the gravitational wrench in sensor frame for a given orientation.

    Returns
    -------
    wrench : (6,) [fx, fy, fz, tx, ty, tz] due to gravity on the payload.
    """
    g_sensor = R_sensor_world @ np.array([0.0, 0.0, -g])
    f_gravity = mass * g_sensor
    t_gravity = np.cross(cog, f_gravity)
    return np.concatenate([f_gravity, t_gravity])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def split_payload(
    total_mass: float,
    total_cog_flange: np.ndarray,
    payload_mass: float,
    payload_cog_sensor: np.ndarray,
    measuring_offset: np.ndarray,
    R_flange_sensor: np.ndarray = np.eye(3),
) -> tuple[float, np.ndarray]:
    """
    Split the total flange payload into the part the external sensor carries and the rest.

    UR's ft_rtde_input_enable expects the installation payload to be the total load on the
    flange, and sensor_mass/sensor_cog to describe what sits between flange and measuring point
    (sensor body, adapters, camera), i.e. what the external sensor cannot see.

    Parameters
    ----------
    total_mass, total_cog_flange :
        Total payload on the flange (e.g. the teach pendant's payload estimate), CoG in flange frame.
    payload_mass, payload_cog_sensor :
        Payload identified from the external sensor, CoG in sensor frame relative to its
        measuring point.
    measuring_offset : (3,)
        Sensor measuring point in flange frame.
    R_flange_sensor : (3, 3)
        Rotation from sensor frame to flange frame.

    Returns
    -------
    (sensor_mass, sensor_cog_flange)
    """
    sensor_mass = total_mass - payload_mass
    if sensor_mass <= 0.0:
        raise ValueError(
            f'Total payload {total_mass:.3f} kg must exceed the sensor-side payload '
            f'{payload_mass:.3f} kg'
        )
    payload_cog_flange = np.asarray(measuring_offset) + R_flange_sensor @ np.asarray(payload_cog_sensor)
    sensor_cog = (total_mass * np.asarray(total_cog_flange) - payload_mass * payload_cog_flange) / sensor_mass
    return float(sensor_mass), sensor_cog


def estimate_measuring_offset(
    dF_sensor: np.ndarray,
    dT_sensor: np.ndarray,
    dF_flange: np.ndarray,
    dT_flange: np.ndarray,
    R_flange_sensor: np.ndarray = np.eye(3),
) -> dict:
    """
    Estimate the external sensor's measuring point in the flange frame from push tests.

    Uses wrench changes (push minus unloaded baseline at a fixed pose, which cancels gravity and
    bias) measured simultaneously by the external sensor and a flange-referenced sensor (UR
    built-in with TCP at the flange): dT_flange = R dT_sensor + r x (R dF_sensor).

    Parameters
    ----------
    dF_sensor, dT_sensor : (N, 3)
        External sensor wrench changes in sensor frame.
    dF_flange, dT_flange : (N, 3)
        Flange-referenced wrench changes in flange frame. Only used to detect a sign convention
        mismatch between the two sensors (forces must agree up to sign).

    Returns
    -------
    dict with keys: offset, sign, residual_norm, force_agreement, condition_number.
    """
    F = np.asarray(dF_sensor) @ R_flange_sensor.T
    T = np.asarray(dT_sensor) @ R_flange_sensor.T
    dF_flange = np.asarray(dF_flange)
    sign = 1.0 if np.sum(F * dF_flange) >= 0.0 else -1.0
    T_flange = sign * np.asarray(dT_flange)

    # r x F = -F x r = -skew(F) r
    A = np.vstack([-_skew(f) for f in F])
    b = (T_flange - T).reshape(-1)
    r, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float('inf')

    return {
        'offset': r,
        'sign': sign,
        'residual_norm': float(np.linalg.norm(A @ r - b)),
        'force_agreement': float(np.linalg.norm(sign * dF_flange - F) / max(np.linalg.norm(F), 1e-9)),
        'condition_number': float(cond),
    }


def rtde_sensor_params(params: dict, R_flange_sensor: np.ndarray = np.eye(3)) -> tuple:
    """
    Derive ft_rtde_input_enable(sensor_mass, sensor_measuring_offset, sensor_cog) from loaded
    payload parameters (see load_payload_params). Requires robot_payload and measuring_offset.
    """
    if params.get('robot_payload') is None or params.get('measuring_offset') is None:
        raise ValueError('payload params need robot_payload and measuring_offset for RTDE use')
    total = params['robot_payload']
    sensor_mass, sensor_cog = split_payload(
        total['mass'], total['cog'], params['mass'], params['cog'],
        params['measuring_offset'], R_flange_sensor,
    )
    return sensor_mass, np.asarray(params['measuring_offset']), sensor_cog


def _vec3_dict(v, digits: int = 6) -> dict:
    return {k: round(float(x), digits) for k, x in zip('xyz', v)}


def _dict_vec3(d: dict) -> np.ndarray:
    return np.array([d['x'], d['y'], d['z']], dtype=float)


def save_payload_params(
    result: dict,
    output_path: str | Path,
    sensor_frame: str = 'ur16e_tool0',
    world_frame: str = 'ur16e_base_link',
    robot_payload: dict | None = None,
    measuring_offset: np.ndarray | None = None,
) -> Path:
    """
    Save identified payload parameters to YAML.

    robot_payload ({'mass', 'cog'}, total on the flange, flange frame) and measuring_offset
    (sensor measuring point in flange frame) are optional; they are needed to derive the
    ft_rtde_input_enable sensor parameters.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'mass': round(result['mass'], 6),
        'center_of_gravity': {
            'x': round(result['cog'][0], 6),
            'y': round(result['cog'][1], 6),
            'z': round(result['cog'][2], 6),
        },
        'sensor_bias': {
            'fx': round(result['bias'][0], 4),
            'fy': round(result['bias'][1], 4),
            'fz': round(result['bias'][2], 4),
            'tx': round(result['bias'][3], 6),
            'ty': round(result['bias'][4], 6),
            'tz': round(result['bias'][5], 6),
        },
        'metadata': {
            'num_poses': result['num_poses'],
            'residual_norm': round(result['residual_norm'], 4),
            'condition_number': round(result['condition_number'], 2),
            'calibrated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'sensor_frame': sensor_frame,
            'world_frame': world_frame,
        },
    }
    if robot_payload is not None:
        data['robot_payload'] = {
            'mass': round(float(robot_payload['mass']), 6),
            'center_of_gravity': _vec3_dict(robot_payload['cog']),
        }
    if measuring_offset is not None:
        data['measuring_offset'] = _vec3_dict(measuring_offset)

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return output_path


def load_payload_params(yaml_path: str | Path) -> dict:
    """
    Load payload parameters from YAML.

    Returns
    -------
    dict with keys: mass, cog (np.ndarray), bias (np.ndarray), robot_payload ({'mass', 'cog'}
    or None), measuring_offset (np.ndarray or None).
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    bias_d = data['sensor_bias']
    robot_payload = data.get('robot_payload')
    if robot_payload is not None:
        robot_payload = {
            'mass': float(robot_payload['mass']),
            'cog': _dict_vec3(robot_payload['center_of_gravity']),
        }
    offset = data.get('measuring_offset')
    return {
        'mass': float(data['mass']),
        'cog': _dict_vec3(data['center_of_gravity']),
        'bias': np.array([
            bias_d['fx'], bias_d['fy'], bias_d['fz'],
            bias_d['tx'], bias_d['ty'], bias_d['tz'],
        ]),
        'robot_payload': robot_payload,
        'measuring_offset': None if offset is None else _dict_vec3(offset),
    }
