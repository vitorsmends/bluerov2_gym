# Path Tracking Experiments

This package contains a self-contained organization for BlueROV2 path-tracking
experiments using PID, MPC and PPO controllers.

## Usage

Copy this folder to the root of the `bluerov2_gym` project and run:

```bash
python path_tracking_experiments/run_pid.py
python path_tracking_experiments/run_mpc.py
python path_tracking_experiments/run_ppo.py
```

Or run all experiments:

```bash
python path_tracking_experiments/run_all.py
```

## Outputs

CSV files are saved to:

```text
results/path_tracking/
```

The main metric is:

```text
tracking_error_m
```

which is the Euclidean position tracking error in meters.
