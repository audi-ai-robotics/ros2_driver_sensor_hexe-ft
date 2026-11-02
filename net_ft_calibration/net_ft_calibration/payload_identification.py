"""Least-squares payload identification and gravity compensation math."""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


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
    bias: np.ndarray | None = None,
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
    bias : (6,), optional
        Known sensor bias (e.g. from another calibration run); only mass and CoG are fitted.

    Returns
    -------
    dict with keys: mass, cog, bias, residual_norm, condition_number, num_poses,
    pose_force_residuals, pose_torque_residuals (per-pose residual norms, N and Nm),
    force_rms, torque_rms (RMS of the per-pose residual norms, N and Nm; unlike residual_norm
    these don't mix units or grow with the number of poses), axis_rms (per-axis RMS, 6 values)
    and residuals (N x 6).
    """
    n = len(rotations)
    A = np.zeros((6 * n, 10))
    b = np.zeros(6 * n)

    for i, R_sw in enumerate(rotations):
        A[6 * i:6 * (i + 1), :] = build_gravity_regressor(R_sw, g)
        b[6 * i:6 * (i + 1)] = wrenches[i]

    if bias is None:
        x, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    else:
        b_fixed = b - np.tile(bias, n)
        x4, _, _, sv = np.linalg.lstsq(A[:, :4], b_fixed, rcond=None)
        x = np.concatenate([x4, bias])
    cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float('inf')

    mass = x[0]
    if abs(mass) < 1e-6:
        cog = np.zeros(3)
    else:
        cog = x[1:4] / mass

    residual = (A @ x - b).reshape(n, 6)
    residual_norm = float(np.linalg.norm(residual))
    pose_f = np.linalg.norm(residual[:, :3], axis=1)
    pose_t = np.linalg.norm(residual[:, 3:], axis=1)

    return {
        'mass': float(mass),
        'cog': cog.tolist(),
        'bias': x[4:10].tolist(),
        'residual_norm': residual_norm,
        'condition_number': float(cond),
        'num_poses': n,
        'pose_force_residuals': pose_f.tolist(),
        'pose_torque_residuals': pose_t.tolist(),
        'force_rms': float(np.sqrt(np.mean(pose_f ** 2))),
        'torque_rms': float(np.sqrt(np.mean(pose_t ** 2))),
        'axis_rms': np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
        'residuals': residual.tolist(),
    }


def rotate_sensor_frame(rotations: list[np.ndarray], rpy) -> list[np.ndarray]:
    """
    Re-express R_sensor_world for a measurement frame rotated by rpy (fixed-axis xyz, rad)
    relative to the frame the rotations were recorded in, like an <origin rpy> on the sensor
    joint in the URDF.
    """
    R_c = Rotation.from_euler('xyz', rpy).as_matrix()
    return [R_c.T @ R_sw for R_sw in rotations]


def fit_sensor_rotation(
    wrenches: np.ndarray,
    rotations: list[np.ndarray],
    g: float = 9.81,
    rpy0=(0.0, 0.0, 0.0),
) -> dict:
    """
    Estimate a small measurement-frame rotation (rpy) together with the payload by minimizing
    the identification residual. Returns the identify_payload result plus 'rpy'.
    """
    def residual(rpy):
        return np.ravel(identify_payload(wrenches, rotate_sensor_frame(rotations, rpy), g)['residuals'])

    sol = least_squares(residual, np.asarray(rpy0, dtype=float), x_scale=0.01)
    result = identify_payload(wrenches, rotate_sensor_frame(rotations, sol.x), g)
    result['rpy'] = sol.x.tolist()
    return result


def pose_drift(wrenches: np.ndarray, joint_positions: np.ndarray, tol: float = 1e-3) -> np.ndarray | None:
    """
    Wrench change between the first and last capture if both were taken at the same joint pose
    (max joint difference below tol rad), else None. Model-free: shows drift over the run.
    """
    joint_positions = np.asarray(joint_positions)
    if len(wrenches) < 2 or joint_positions.shape[0] != len(wrenches):
        return None
    if np.max(np.abs(joint_positions[-1] - joint_positions[0])) > tol:
        return None
    return np.asarray(wrenches[-1]) - np.asarray(wrenches[0])


def save_captures(
    path: str | Path,
    wrenches: np.ndarray,
    wrench_stds: np.ndarray,
    rotations: list[np.ndarray],
    joint_positions: np.ndarray,
    passes: np.ndarray,
    **meta,
) -> Path:
    """
    Save raw calibration captures (per pose: mean/std wrench, R_sensor_world, arm joint positions
    in joint_states order, auto-capture pass index) to .npz for offline analysis. String metadata
    (label, frames, ...) goes in as extra keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        wrenches=np.asarray(wrenches, dtype=float).reshape(-1, 6),
        wrench_stds=np.asarray(wrench_stds, dtype=float).reshape(-1, 6),
        rotations=np.asarray(rotations, dtype=float).reshape(-1, 3, 3),
        joint_positions=np.asarray(joint_positions, dtype=float).reshape(len(rotations), -1),
        passes=np.asarray(passes, dtype=int),
        **{k: np.asarray(str(v)) for k, v in meta.items()},
    )
    return path


def load_captures(path: str | Path) -> dict:
    """Load captures saved by save_captures; arrays as np.ndarray, metadata as str."""
    with np.load(path) as data:
        return {k: (data[k] if data[k].dtype.kind != 'U' else str(data[k])) for k in data.files}


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


def combine_payload(
    sensor_mass: float,
    sensor_cog_flange: np.ndarray,
    payload_mass: float,
    payload_cog_sensor: np.ndarray,
    measuring_offset: np.ndarray,
    R_flange_sensor: np.ndarray = np.eye(3),
) -> tuple[float, np.ndarray]:
    """
    Inverse of split_payload: total flange payload from the sensor-side part and the payload the
    external sensor carries (e.g. gripper plus a grasped part).

    Returns
    -------
    (total_mass, total_cog_flange)
    """
    total_mass = sensor_mass + payload_mass
    payload_cog_flange = np.asarray(measuring_offset) + R_flange_sensor @ np.asarray(payload_cog_sensor)
    total_cog = (sensor_mass * np.asarray(sensor_cog_flange) + payload_mass * payload_cog_flange) / total_mass
    return float(total_mass), total_cog


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


DEFAULT_PROFILE = 'default'


def save_payload_params(
    result: dict,
    output_path: str | Path,
    sensor_frame: str = 'ur16e_tool0',
    world_frame: str = 'ur16e_base_link',
    robot_payload: dict | None = None,
    measuring_offset: np.ndarray | None = None,
    profile: str | None = None,
) -> Path:
    """
    Save identified payload parameters to YAML.

    robot_payload ({'mass', 'cog'}, total on the flange, flange frame) and measuring_offset
    (sensor measuring point in flange frame) are optional; they are needed to derive the
    ft_rtde_input_enable sensor parameters.

    With a profile name other than 'default', the parameters are merged into profiles.<profile>
    of the existing file (which must exist) instead of overwriting it. robot_payload is not
    stored there: load_payload_params derives it from the default profile's sensor-side part.
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
            **{k: round(result[k], 4) for k in ('force_rms', 'torque_rms') if k in result},
            'calibrated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'sensor_frame': sensor_frame,
            'world_frame': world_frame,
        },
    }
    to_profile = bool(profile) and profile != DEFAULT_PROFILE
    if robot_payload is not None and not to_profile:
        data['robot_payload'] = {
            'mass': round(float(robot_payload['mass']), 6),
            'center_of_gravity': _vec3_dict(robot_payload['cog']),
        }
    if measuring_offset is not None:
        data['measuring_offset'] = _vec3_dict(measuring_offset)

    if to_profile:
        with open(output_path, 'r') as f:
            base = yaml.safe_load(f)
        base.setdefault('profiles', {})[profile] = data
        data = base

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return output_path


def _parse_payload_params(data: dict) -> dict:
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
        ], dtype=float),
        'robot_payload': robot_payload,
        'measuring_offset': None if offset is None else _dict_vec3(offset),
    }


def list_payload_profiles(yaml_path: str | Path) -> list[str]:
    """Profile names in a payload_params.yaml; 'default' is the top level."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return [DEFAULT_PROFILE, *(data.get('profiles') or {})]


def load_payload_params(yaml_path: str | Path, profile: str | None = None) -> dict:
    """
    Load payload parameters from YAML.

    The top-level keys are the 'default' profile. A named profile from the optional `profiles`
    block overrides top-level keys and inherits the rest (e.g. measuring_offset). If it has no
    robot_payload, that is derived as the default profile's sensor-side part (flange to measuring
    point, see rtde_sensor_params) plus the profile's payload, since ft_rtde_input_enable is
    configured once from the default profile.

    Returns
    -------
    dict with keys: mass, cog (np.ndarray), bias (np.ndarray), robot_payload ({'mass', 'cog'}
    or None), measuring_offset (np.ndarray or None), profile (str).
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    profiles = data.pop('profiles', None) or {}
    base = _parse_payload_params(data)
    if not profile or profile == DEFAULT_PROFILE:
        return {**base, 'profile': DEFAULT_PROFILE}
    if profile not in profiles:
        raise KeyError(
            f"Unknown payload profile '{profile}' in {yaml_path}; "
            f"available: {[DEFAULT_PROFILE, *profiles]}"
        )

    params = _parse_payload_params({**data, **profiles[profile]})
    if 'robot_payload' not in profiles[profile]:
        if base['robot_payload'] is None or params['measuring_offset'] is None:
            params['robot_payload'] = None
        else:
            sensor_mass, _, sensor_cog = rtde_sensor_params(base)
            mass, cog = combine_payload(
                sensor_mass, sensor_cog, params['mass'], params['cog'], params['measuring_offset'],
            )
            params['robot_payload'] = {'mass': mass, 'cog': cog}
    return {**params, 'profile': profile}
