#!/usr/bin/env python3
"""Compare payload calibration runs recorded by identify_payload_interactive.py.

Usage: python3 compare_calibrations.py RUN [RUN ...] [--bias-from LABEL] [--plot]
RUN is a run directory (verification/logs/<timestamp>_<label>) or its captures.npz.

Per run: identified mass/CoG/bias, per-pose force/torque RMS, drift (first vs. last capture at
the same pose), hysteresis (same pose captured in several auto-capture passes, e.g. 'a' then
'a rev') and an offline tilt check: the fit with the old measured mount tilt applied and with a
freely fitted measurement-frame rotation, without re-running the robot or rebuilding the URDF.
--bias-from refits the other runs with that run's sensor bias fixed (default: a run labelled
'bare', if given); the bias is sensor-intrinsic and should carry over between mounting stages.
Needs the sourced workspace (net_ft_calibration on PYTHONPATH).
"""
import argparse
from pathlib import Path

import numpy as np

from net_ft_calibration import fit_sensor_rotation, identify_payload, load_captures, pose_drift, rotate_sensor_frame

# Measured HEX-E mount tilt that was in ur16e_cell_macro.xacro (robot repo 487d284, hex_e_meas_rpy).
OLD_TILT_RPY = (0.006760, 0.035168, -0.080202)


def load_run(path):
    path = Path(path)
    data = load_captures(path / 'captures.npz' if path.is_dir() else path)
    data['label'] = data.get('label') or path.name
    data['name'] = path.name if path.is_dir() else path.parent.name
    data['rotations'] = list(data['rotations'])
    return data


def hysteresis(run, tol=1e-3):
    """|dF| between captures of the same joint pose taken in different passes."""
    q, w, p = run['joint_positions'], run['wrenches'], run['passes']
    diffs = []
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if p[i] != p[j] and np.max(np.abs(q[i] - q[j])) < tol:
                diffs.append(np.linalg.norm(w[j, :3] - w[i, :3]))
    return np.array(diffs)


def analyze(run, bias=None):
    w, rot = run['wrenches'], run['rotations']
    res = identify_payload(w, rot)
    out = {
        'label': run['label'], 'name': run['name'], 'n': len(w), 'res': res,
        'tilt': identify_payload(w, rotate_sensor_frame(rot, OLD_TILT_RPY)),
        'fit_rot': fit_sensor_rotation(w, rot),
        'drift': pose_drift(w, run['joint_positions']),
        'hyst': hysteresis(run),
        'passes': {},
    }
    for p in np.unique(run['passes']):
        idx = np.flatnonzero(run['passes'] == p)
        if 3 <= len(idx) < len(w):
            out['passes'][int(p)] = identify_payload(w[idx], [rot[i] for i in idx])
    if bias is not None:
        out['fixed_bias'] = identify_payload(w, rot, bias=bias)
    return out


def print_report(results, bias_label):
    np.set_printoptions(precision=4, suppress=True)
    hdr = f"{'run':28s} {'n':>3s} {'mass kg':>8s} {'F rms N':>8s} {'T rms Nm':>9s} {'cond':>6s} " \
          f"{'tilt F':>7s} {'fitR F':>7s} {'drift N':>8s} {'hyst N':>12s}"
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        res = r['res']
        drift = f"{np.linalg.norm(r['drift'][:3]):8.3f}" if r['drift'] is not None else f"{'-':>8s}"
        hyst = f"{r['hyst'].mean():5.3f}/{r['hyst'].max():5.3f}" if len(r['hyst']) else '-'
        print(f"{r['name'][:28]:28s} {r['n']:3d} {res['mass']:8.4f} {res['force_rms']:8.3f} "
              f"{res['torque_rms']:9.4f} {res['condition_number']:6.1f} {r['tilt']['force_rms']:7.3f} "
              f"{r['fit_rot']['force_rms']:7.3f} {drift} {hyst:>12s}")
    print("tilt F = force RMS with the old mount tilt applied; fitR F = with a fitted frame rotation;"
          " hyst = mean/max |dF| at equal poses across passes")

    for r in results:
        res = r['res']
        print(f"\n== {r['name']} (label '{r['label']}')")
        print(f"  CoG [m]            {np.array(res['cog'])}")
        print(f"  bias F [N]         {np.array(res['bias'][:3])}   T [Nm] {np.array(res['bias'][3:])}")
        print(f"  axis RMS F [N]     {np.array(res['axis_rms'][:3])}   T [Nm] {np.array(res['axis_rms'][3:])}")
        print(f"  fitted rotation    rpy {np.degrees(r['fit_rot']['rpy'])} deg "
              f"(old tilt {np.degrees(OLD_TILT_RPY)} deg)")
        worst = np.argsort(res['pose_force_residuals'])[::-1][:3]
        print(f"  worst poses        " + ', '.join(
            f"#{i + 1}: {res['pose_force_residuals'][i]:.3f} N" for i in worst))
        if r['drift'] is not None:
            print(f"  drift first->last  F {r['drift'][:3]} N   T {r['drift'][3:]} Nm")
        for p, pr in r['passes'].items():
            print(f"  pass {p}             mass {pr['mass']:.4f} kg  F rms {pr['force_rms']:.3f} N  "
                  f"bias F {np.array(pr['bias'][:3])}")
        if 'fixed_bias' in r:
            fb = r['fixed_bias']
            print(f"  bias from '{bias_label}'  mass {fb['mass']:.4f} kg  F rms {fb['force_rms']:.3f} N "
                  f"(free bias: {res['force_rms']:.3f} N)")


def plot(results, runs):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for r, run in zip(results, runs):
        f = r['res']['pose_force_residuals']
        ax1.plot(np.arange(1, len(f) + 1), f, marker='o', label=r['name'])
        # angle between the sensor z axis (tool axis) and "down": 0 deg = tool pointing down
        g_dirs = np.array([Rw @ np.array([0.0, 0.0, -1.0]) for Rw in run['rotations']])
        ax2.scatter(np.degrees(np.arccos(np.clip(g_dirs[:, 2], -1, 1))), f, label=r['name'])
    ax1.set(xlabel='capture #', ylabel='force residual [N]', title='Force residual per capture')
    ax2.set(xlabel='angle tool axis vs. gravity [deg]', ylabel='force residual [N]',
            title='Force residual vs. orientation')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--bias-from', default='bare', help="label of the run whose bias is reused")
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    runs = [load_run(p) for p in args.runs]
    ref = next((r for r in runs if r['label'] == args.bias_from), None)
    bias = np.array(identify_payload(ref['wrenches'], ref['rotations'])['bias']) if ref else None
    results = [analyze(r, bias if r is not ref else None) for r in runs]
    print_report(results, args.bias_from)
    if args.plot:
        plot(results, runs)


if __name__ == '__main__':
    main()
