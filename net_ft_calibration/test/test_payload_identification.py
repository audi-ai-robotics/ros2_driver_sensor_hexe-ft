import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from net_ft_calibration.payload_identification import (
    combine_payload,
    compute_gravity_wrench,
    estimate_measuring_offset,
    fit_sensor_rotation,
    identify_payload,
    list_payload_profiles,
    load_captures,
    load_payload_params,
    pose_drift,
    rotate_sensor_frame,
    rtde_sensor_params,
    save_captures,
    save_payload_params,
    split_payload,
    transform_wrench,
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


def test_rms_metrics_are_per_pose_and_unit_separated():
    mass, cog = 1.7, np.array([0.0, 0.0, 0.05])
    rng = np.random.default_rng(3)
    rotations = _random_rotations(40, seed=3)
    wrenches = np.array([compute_gravity_wrench(Rw, mass, cog) for Rw in rotations])
    wrenches[:, :3] += rng.normal(0, 0.3, size=(40, 3))

    result = identify_payload(wrenches, rotations)

    # ~0.3 N per axis -> ~0.52 N per pose, independent of the number of poses
    assert result['force_rms'] == pytest.approx(0.3 * np.sqrt(3), rel=0.2)
    assert result['torque_rms'] < 0.05
    assert len(result['axis_rms']) == 6
    assert np.array(result['residuals']).shape == (40, 6)


def test_fixed_bias_fits_only_mass_and_cog():
    mass, cog = 1.7, np.array([-0.003, -0.003, 0.057])
    bias = np.array([1.0, -0.4, -10.7, -0.3, 0.6, 0.03])
    rotations = _random_rotations(12, seed=4)
    wrenches = np.array([compute_gravity_wrench(Rw, mass, cog) + bias for Rw in rotations])

    result = identify_payload(wrenches, rotations, bias=bias)

    assert result['mass'] == pytest.approx(mass, abs=1e-9)
    np.testing.assert_allclose(result['cog'], cog, atol=1e-9)
    np.testing.assert_allclose(result['bias'], bias)


def test_fit_sensor_rotation_recovers_mount_tilt():
    mass, cog = 1.7, np.array([-0.003, -0.003, 0.057])
    bias = np.array([1.0, -0.4, -10.7, -0.3, 0.6, 0.03])
    tilt = np.array([0.006760, 0.035168, -0.080202])
    rotations = _random_rotations(13, seed=5)
    # recorded with the untilted frame, measured in the tilted one
    wrenches = np.array([
        compute_gravity_wrench(Rw, mass, cog) + bias
        for Rw in rotate_sensor_frame(rotations, tilt)
    ])

    assert identify_payload(wrenches, rotations)['force_rms'] > 0.1
    result = fit_sensor_rotation(wrenches, rotations)

    np.testing.assert_allclose(result['rpy'], tilt, atol=1e-6)
    assert result['force_rms'] < 1e-6


def test_pose_drift_only_for_matching_first_and_last_pose():
    wrenches = np.array([[0.0] * 6, [5.0] * 6, [0.2, 0, 0, 0, 0, 0.01]])
    q = np.array([[0.0] * 6, [1.0] * 6, [0.0] * 6])

    np.testing.assert_allclose(pose_drift(wrenches, q), [0.2, 0, 0, 0, 0, 0.01])
    q[-1, 5] = 0.1
    assert pose_drift(wrenches, q) is None


def test_save_load_captures_roundtrip(tmp_path):
    rotations = _random_rotations(4, seed=6)
    wrenches = np.arange(24, dtype=float).reshape(4, 6)
    joints = np.arange(24, dtype=float).reshape(4, 6) * 0.1

    path = save_captures(tmp_path / 'run' / 'captures.npz', list(wrenches), np.ones((4, 6)),
                         rotations, list(joints), [1, 1, 2, 2], label='bare')
    data = load_captures(path)

    np.testing.assert_allclose(data['wrenches'], wrenches)
    np.testing.assert_allclose(data['rotations'], np.array(rotations))
    np.testing.assert_allclose(data['joint_positions'], joints)
    np.testing.assert_array_equal(data['passes'], [1, 1, 2, 2])
    assert data['label'] == 'bare'


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


def test_combine_payload_inverts_split_payload():
    offset = np.array([0.0, 0.0, 0.0375])
    R_fs = R.from_euler('z', 30, degrees=True).as_matrix()
    total_mass, total_cog = 2.72, np.array([0.002, -0.013, 0.082])
    payload_mass, payload_cog = 1.817, np.array([-0.0004, -0.0051, 0.0545])
    m_s, c_s = split_payload(total_mass, total_cog, payload_mass, payload_cog, offset, R_fs)

    m, c = combine_payload(m_s, c_s, payload_mass, payload_cog, offset, R_fs)

    assert m == pytest.approx(total_mass)
    np.testing.assert_allclose(c, total_cog, atol=1e-12)


def _profile_file(tmp_path):
    base = {
        'mass': 1.817, 'cog': [-0.0004, -0.0051, 0.0545], 'bias': [1.5, -2.4, -11.8, -0.34, 0.65, 0.0],
        'residual_norm': 2.2, 'condition_number': 30.9, 'num_poses': 9,
    }
    ecu = {
        'mass': 1.921, 'cog': [-0.0013, -0.0056, 0.0607], 'bias': [2.3, -0.4, -10.6, -0.26, 0.70, 0.09],
        'residual_norm': 2.1, 'condition_number': 36.0, 'num_poses': 8,
    }
    path = save_payload_params(
        base, tmp_path / 'p.yaml',
        robot_payload={'mass': 2.72, 'cog': [0.002, -0.013, 0.082]},
        measuring_offset=[0.0, 0.0, 0.0375],
    )
    # The pendant payload passed here is ignored for profiles; it is derived instead.
    save_payload_params(ecu, path, robot_payload={'mass': 2.72, 'cog': [0.0, 0.0, 0.0]}, profile='ecu')
    return path


def test_profile_overrides_and_inherits(tmp_path):
    path = _profile_file(tmp_path)

    default = load_payload_params(path)
    ecu = load_payload_params(path, 'ecu')

    assert list_payload_profiles(path) == ['default', 'ecu']
    assert default['profile'] == 'default' and default['mass'] == pytest.approx(1.817)
    assert ecu['profile'] == 'ecu' and ecu['mass'] == pytest.approx(1.921)
    np.testing.assert_allclose(ecu['bias'], [2.3, -0.4, -10.6, -0.26, 0.70, 0.09])
    np.testing.assert_allclose(ecu['measuring_offset'], [0.0, 0.0, 0.0375])
    # load_payload_params(path, 'default') is the same as without a profile
    np.testing.assert_allclose(load_payload_params(path, 'default')['bias'], default['bias'])


def test_profile_robot_payload_keeps_sensor_side_part(tmp_path):
    path = _profile_file(tmp_path)
    default = load_payload_params(path)
    ecu = load_payload_params(path, 'ecu')

    # Total grows by exactly the extra sensor-side payload (the grasped part) ...
    assert ecu['robot_payload']['mass'] == pytest.approx(2.72 + (1.921 - 1.817))
    # ... and the ft_rtde_input_enable sensor part stays the one configured from the default profile.
    m_d, _, c_d = rtde_sensor_params(default)
    m_e, _, c_e = rtde_sensor_params(ecu)
    assert m_e == pytest.approx(m_d)
    np.testing.assert_allclose(c_e, c_d, atol=1e-12)


def test_unknown_profile_raises(tmp_path):
    with pytest.raises(KeyError, match='ecu'):
        load_payload_params(_profile_file(tmp_path), 'nope')


def test_transform_wrench_identity_shifts_torque_to_tcp():
    # 10 N along x measured at the HEX-E measuring point, 37.5 mm above the flange/TCP
    w = transform_wrench([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.eye(3), [0.0, 0.0, 0.0375])
    np.testing.assert_allclose(w, [10.0, 0.0, 0.0, 0.0, 0.375, 0.0], atol=1e-12)


def test_transform_wrench_matches_rigid_body_moment():
    rng = np.random.default_rng(7)
    R_oi = R.random(random_state=7).as_matrix()
    f_in, lever = rng.normal(size=3), rng.normal(size=3) * 0.05
    # a pure force applied at the input reference point has zero torque there
    w = transform_wrench(np.r_[f_in, 0.0, 0.0, 0.0], R_oi, lever)
    np.testing.assert_allclose(w[:3], R_oi @ f_in)
    np.testing.assert_allclose(w[3:], np.cross(lever, R_oi @ f_in))
    # moving back with the inverse rotation and the opposite lever recovers the input
    back = transform_wrench(w, R_oi.T, -R_oi.T @ lever)
    np.testing.assert_allclose(back, np.r_[f_in, 0.0, 0.0, 0.0], atol=1e-12)
