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
    dict with keys: mass, cog, bias, residual_norm, condition_number, num_poses.
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

    residual_norm = float(np.linalg.norm(A @ x - b))

    return {
        'mass': float(mass),
        'cog': cog.tolist(),
        'bias': x[4:10].tolist(),
        'residual_norm': residual_norm,
        'condition_number': float(cond),
        'num_poses': n,
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


def save_payload_params(
    result: dict,
    output_path: str | Path,
    sensor_frame: str = 'ur16e_tool0',
    world_frame: str = 'ur16e_base_link',
) -> Path:
    """Save identified payload parameters to YAML."""
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

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return output_path


def load_payload_params(yaml_path: str | Path) -> dict:
    """
    Load payload parameters from YAML.

    Returns
    -------
    dict with keys: mass, cog (np.ndarray), bias (np.ndarray).
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    cog_d = data['center_of_gravity']
    bias_d = data['sensor_bias']
    return {
        'mass': float(data['mass']),
        'cog': np.array([cog_d['x'], cog_d['y'], cog_d['z']]),
        'bias': np.array([
            bias_d['fx'], bias_d['fy'], bias_d['fz'],
            bias_d['tx'], bias_d['ty'], bias_d['tz'],
        ]),
    }
