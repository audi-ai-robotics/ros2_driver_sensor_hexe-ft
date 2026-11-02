# net_ft_calibration

Gravity compensation and payload identification for network F/T sensors (OnRobot Hex-E, ATI) on the UR16e cell with Schunk gripper.

## Overview

Raw F/T sensor readings include the gravitational load of the tool/gripper. This package provides:
1. **Payload identification** — multi-pose calibration to estimate tool mass, center of gravity, and sensor bias
2. **Gravity compensation** — real-time subtraction of gravitational wrench from raw readings
3. **Runtime tare** — re-zeroing service for gripper state changes or drift correction

## Prerequisites

The following must be running:

```bash
# 1. Robot driver + MoveIt (for calibration; only driver needed for compensation)
ros2 launch ur16e_cell_control start_robot.launch.py ns:=robot1
ros2 launch ur16e_cell_moveit_config ur_moveit.launch.py  # only for calibration

# F/T sensor should already be running as part of start_robot
# Verify: ros2 topic echo /robot1/force_torque_sensor_broadcaster/wrench
```

## Quick Start

### Step 1: Payload Identification (one-time)

```bash
ros2 run net_ft_calibration identify_payload_interactive.py --ros-args \
  -r __ns:=/robot1 -r /tf:=tf -r /tf_static:=tf_static -p robot_ns:=robot1
```

Use `ros2 run`, not `ros2 launch`: launch does not forward keyboard input to the node, so the
interactive commands below would be ignored.

Interactive commands:
| Key | Action |
|-----|--------|
| `c` | Capture current pose (ensure robot is stationary, no external contact) |
| `s` | Save current joint pose to `joint_poses.yaml` for reuse |
| `a` | Auto-capture all saved joint poses (`a rev` = reverse order, for hysteresis checks) |
| `r` | Run identification (least-squares solve) |
| `w` | Save results to `payload_params.yaml` |
| `h` | Move to home |
| `d` | Delete all captured data |
| `q` | Quit |

Capture at least 6 poses with diverse tool orientations (see Tips below).

Each session writes to `verification/logs/<timestamp>_<run_label>/` in this repo (gitignored;
override with `-p output_dir:=...`, label with `-p run_label:=gripper`): `captures.npz` (raw
captures, updated after every capture) and `payload_params.yaml` (on `w`). Compare runs offline:

```bash
python3 verification/compare_calibrations.py verification/logs/*_bare verification/logs/*_gripper --plot
```

### Step 2: Gravity Compensation (runtime)

```bash
ros2 launch net_ft_calibration gravity_compensation.launch.py \
    ns:=robot1 \
    payload_params_file:=<repo>/verification/logs/<run>/payload_params.yaml
```

This publishes compensated wrench on `/{ns}/ft_compensated`.

### Step 3: Tare (as needed)

After changing gripper state (open/close) or to correct drift:

```bash
ros2 service call /robot1/gravity_compensation/tare std_srvs/srv/Trigger
```

Ensure the robot is stationary and no external forces are applied when taring.

## Calibration Workflow in Detail

1. **Prepare**: mount the gripper, close it (or set to the most common operating state)
2. **Launch**: start the identification tool with the robot namespace
3. **Capture poses**: move the robot to 8-12 diverse orientations:
   - Tool pointing down (nominal)
   - Tool pointing up (180-degree flip)
   - Tool tilted sideways in 4 directions
   - 45-degree tilts
   - Vary wrist orientation
4. **Compute**: press `r` — check the results:
   - Mass should match the known gripper weight (~1-3 kg for Schunk EGU)
   - CoG Z should be positive (below the sensor, toward the tool)
   - Condition number < 50 indicates good pose diversity
   - Judge the fit by the per-pose force RMS (N), not the residual norm: the latter mixes N and Nm
     and grows with the number of poses
   - If first and last pose are the same, the drift between them is logged (thermal drift check)
5. **Save**: press `w` to save the YAML file
6. **Install**: copy the params file to the package config for permanent use:

```bash
cp <ws>/src/ros2_driver_sensor_hexe-ft/verification/logs/<run>/payload_params.yaml \
   <ws>/src/ros2_driver_sensor_hexe-ft/net_ft_calibration/config/
colcon build --packages-select net_ft_calibration
```

## Payload Parameters YAML Format

```yaml
mass: 1.234
center_of_gravity:
  x: 0.001
  y: -0.002
  z: 0.045
sensor_bias:
  fx: 0.12
  fy: -0.34
  fz: 0.56
  tx: 0.001
  ty: -0.002
  tz: 0.003
metadata:
  num_poses: 12
  residual_norm: 0.42
  condition_number: 8.3
  calibrated_at: "2026-09-29 14:30"
  sensor_frame: ur16e_tool0
  world_frame: ur16e_base_link
```

## Topics and Services

| Name | Type | Description |
|------|------|-------------|
| `/{ns}/force_torque_sensor_broadcaster/wrench` | `WrenchStamped` | Input: raw F/T data |
| `/{ns}/ft_compensated` | `WrenchStamped` | Output: gravity-compensated wrench |
| `/{ns}/gravity_compensation/tare` | `std_srvs/Trigger` | Re-zero the compensated output |

## Parameters

### Gravity Compensation Node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `payload_params_file` | `''` | YAML file path; uses package default if empty |
| `sensor_frame` | `ur16e_tool0` | TF frame of the F/T sensor |
| `world_frame` | `ur16e_base_link` | Gravity reference frame (Z-up) |
| `wrench_topic` | `force_torque_sensor_broadcaster/wrench` | Input topic (relative) |
| `compensated_topic` | `ft_compensated` | Output topic (relative) |
| `tare_samples` | `100` | Samples to average for tare |
| `gravity` | `9.81` | Gravitational acceleration (m/s^2) |

### Identification Script

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_ns` | `''` | Robot namespace |
| `sensor_frame` | `ur16e_tool0` | Sensor TF frame |
| `world_frame` | `ur16e_base_link` | World TF frame |
| `wrench_topic` | `force_torque_sensor_broadcaster/wrench` | F/T topic |
| `velocity_scaling` | `0.1` | MoveIt velocity scaling |
| `capture_samples` | `100` | Samples to average per capture |
| `settle_time` | `1.5` | Wait time after motion before capture (s) |

### Payload profiles (grasped parts)

Optional `robot_payload` / `measuring_offset` blocks are used for the RTDE path. A `profiles`
block holds further calibrations, e.g. the gripper with a grasped part. The top level is the
profile `default`. A profile overrides top-level keys and inherits the rest. Its `robot_payload`
is derived as the default profile's sensor-side part plus the profile's payload, so the
`ft_rtde_input_enable` parameters (set once at launch from `default`) stay valid:

```yaml
profiles:
  ecu:
    mass: 1.920771
    center_of_gravity: {x: -0.001308, y: -0.005615, z: 0.060657}
    sensor_bias: {fx: 2.2791, fy: -0.4311, fz: -10.5865, tx: -0.255993, ty: 0.701065, tz: 0.0898}
```

To add one, calibrate with the part grasped, then copy `mass`, `center_of_gravity`,
`sensor_bias` and `metadata` from the run's `payload_params.yaml` into a new profile. Don't copy
its `robot_payload`. Alternatively, call `save_payload_params(..., profile='name')` on the
config file.

Switch at runtime (only while force_mode is inactive). This sets the robot payload
(`io_and_status_controller/set_payload`) and the `external_ft_rtde_streamer` `sensor_bias`:

```bash
ros2 run net_ft_calibration set_payload_profile.py --list
ros2 run net_ft_calibration set_payload_profile.py ecu --ros-args -p robot_ns:=Peach
ros2 run net_ft_calibration set_payload_profile.py default --ros-args -p robot_ns:=Peach
```

The pglfd executor does the same with `-p ft_profile:=ecu`. The robot starts with `default` (the
pendant payload). A profile stays active until it is switched again; check on the pendant
whether a program restart resets it to the installation payload.

## Gripper State Changes

The calibration estimates mass and CoG for the gripper in one specific state. When the gripper opens/closes, the finger positions shift the CoG slightly. Handle this with the tare service:

1. Change gripper state (open/close)
2. Wait for the robot to be stationary
3. Call `ros2 service call /{ns}/gravity_compensation/tare std_srvs/srv/Trigger`

The gravity compensation handles the bulk of the gravitational wrench across all orientations; the tare corrects the small residual from the CoG shift.

## Theory

At each robot pose, the measured wrench is:

```
w_measured = w_gravity + w_bias
```

where the gravitational wrench in the sensor frame is:

```
g_sensor = R_sensor_world @ [0, 0, -g]^T
F_gravity = m * g_sensor
T_gravity = [cx, cy, cz]^T x F_gravity
```

This can be written as a linear system `A @ x = b` with 10 unknowns: `[m, m*cx, m*cy, m*cz, bfx, bfy, bfz, btx, bty, btz]`. Given N poses (N >= 3, recommended >= 6), the system is solved via least squares.

## File Structure

```
net_ft_calibration/
├── net_ft_calibration/
│   ├── payload_identification.py   # Least-squares math
│   └── robot.py                    # MoveIt robot controller
├── scripts/
│   ├── identify_payload_interactive.py  # Interactive calibration
│   └── gravity_compensation_node.py     # Runtime compensation node
├── config/
│   ├── payload_params.yaml         # Identified payload parameters
│   └── joint_poses.yaml            # Calibration poses
└── launch/
    ├── identify_payload.launch.py
    └── gravity_compensation.launch.py
```

## Tips

- **Pose diversity**: the most important factor. Vary the tool orientation in multiple axes — just changing XY position doesn't help.
- **No external contact**: ensure nothing touches the gripper/tool during capture. No table contact, no cable pull.
- **Stabilization**: the robot must be completely still when capturing. The default 1.5s settle time handles this.
- **Sanity check**: the estimated mass should match the known gripper/tool weight. If it's wildly off, check the wrench topic and TF frames.
- **Condition number**: < 20 is excellent, < 50 is acceptable, > 50 means poor pose diversity.
- **Recalibrate**: whenever the tool/gripper changes physically (new fingers, different payload).
