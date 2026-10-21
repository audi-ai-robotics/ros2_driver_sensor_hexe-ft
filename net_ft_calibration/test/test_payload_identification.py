import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from net_ft_calibration.payload_identification import (
    compute_gravity_wrench,
    estimate_measuring_offset,
    identify_payload,
    load_payload_params,
    rtde_sensor_params,
    save_payload_params,
    split_payload,
)


def _random_rotations(n, seed=0):
    return list(R.random(n, random_state=seed).as_matrix())


def test_identify_payload_recovers_mass_cog_bias():
    mass, cog = 1.7, np.array([-0.003, -0.003, 0.057])
    bias = np.array([1.6, -1.1, -10.2, -0.44, 0.56, 0.05])
    rotations = _random_rotations(12)
    wrenches = np.array([compute_gravity_wrench(Rw, mass, cog) + bias for Rw in rotations])

    result = identify_payload(wrenches, rotations)

    assert result['mass'] == pytest.approx(mass, abs=1e-9)
    np.testing.assert_allclose(result['cog'], cog, atol=1e-9)
    np.testing.assert_allclose(result['bias'], bias, atol=1e-9)
    assert len(result['pose_force_residuals']) == 12
    assert max(result['pose_force_residuals']) < 1e-9


def test_per_pose_residual_flags_outlier():
    mass, cog = 1.7, np.array([0.0, 0.0, 0.05])
    rotations = _random_rotations(10, seed=1)
    wrenches = np.array([compute_gravity_wrench(Rw, mass, cog) for Rw in rotations])
    wrenches[4, :3] += [0.0, 3.0, 0.0]  # e.g. a cable pulling in one pose

    result = identify_payload(wrenches, rotations)

    assert int(np.argmax(result['pose_force_residuals'])) == 4


def test_split_payload_recovers_sensor_side_mass():
    offset = np.array([0.0, 0.0, 0.0445])
    sensor_mass, sensor_cog = 1.08, np.array([0.005, -0.008, 0.02])
    payload_mass, payload_cog_sensor = 1.70, np.array([-0.003, -0.003, 0.057])
    R_fs = R.from_euler('z', 30, degrees=True).as_matrix()
    total_mass = sensor_mass + payload_mass
    total_cog = (sensor_mass * sensor_cog + payload_mass * (offset + R_fs @ payload_cog_sensor)) / total_mass

    m, c = split_payload(total_mass, total_cog, payload_mass, payload_cog_sensor, offset, R_fs)

    assert m == pytest.approx(sensor_mass)
    np.testing.assert_allclose(c, sensor_cog, atol=1e-12)


def test_split_payload_gravity_independent_of_offset():
    # The controller compensates (total - sensor part); its moment about the measuring point must
    # equal the identified payload's regardless of the assumed offset.
    total_mass, total_cog = 2.78, np.array([0.001, -0.005, 0.071])
    payload_mass, payload_cog = 1.70, np.array([-0.003, -0.003, 0.057])
    for z in (0.007, 0.0445, 0.06):
        offset = np.array([0.0, 0.0, z])
        m_s, c_s = split_payload(total_mass, total_cog, payload_mass, payload_cog, offset)
        compensated_cog = (total_mass * total_cog - m_s * c_s) / (total_mass - m_s) - offset
        np.testing.assert_allclose(compensated_cog, payload_cog, atol=1e-12)


def test_split_payload_rejects_inconsistent_masses():
    with pytest.raises(ValueError):
        split_payload(1.0, np.zeros(3), 1.7, np.zeros(3), np.zeros(3))


@pytest.mark.parametrize('ur_sign', [1.0, -1.0])
def test_estimate_measuring_offset(ur_sign):
    rng = np.random.default_rng(2)
    r_true = np.array([0.002, -0.001, 0.045])
    R_fs = R.from_euler('z', 15, degrees=True).as_matrix()
    dF_s = rng.uniform(-40, 40, size=(8, 3))
    dT_s = rng.uniform(-2, 2, size=(8, 3))
    dF_f = dF_s @ R_fs.T
    dT_f = dT_s @ R_fs.T + np.cross(r_true, dF_f)
    noise = rng.normal(0, 0.02, size=dT_f.shape)

    res = estimate_measuring_offset(dF_s, dT_s, ur_sign * dF_f, ur_sign * (dT_f + noise), R_fs)

    assert res['sign'] == ur_sign
    np.testing.assert_allclose(res['offset'], r_true, atol=2e-3)
    assert res['force_agreement'] < 1e-9


def test_save_load_roundtrip_and_rtde_params(tmp_path):
    result = {
        'mass': 1.70, 'cog': [-0.003, -0.003, 0.057], 'bias': [0.0] * 6,
        'residual_norm': 0.1, 'condition_number': 10.0, 'num_poses': 9,
    }
    path = save_payload_params(
        result, tmp_path / 'p.yaml',
        robot_payload={'mass': 2.78, 'cog': [0.001, -0.005, 0.071]},
        measuring_offset=[0.0, 0.0, 0.0445],
    )
    params = load_payload_params(path)

    mass, offset, cog = rtde_sensor_params(params)

    assert mass == pytest.approx(1.08)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.0445])
    assert cog.shape == (3,)


def test_load_legacy_file_without_rtde_blocks(tmp_path):
    result = {
        'mass': 1.0, 'cog': [0.0, 0.0, 0.0], 'bias': [0.0] * 6,
        'residual_norm': 0.0, 'condition_number': 1.0, 'num_poses': 3,
    }
    params = load_payload_params(save_payload_params(result, tmp_path / 'p.yaml'))

    assert params['robot_payload'] is None
    with pytest.raises(ValueError):
        rtde_sensor_params(params)
