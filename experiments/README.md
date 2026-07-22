# Path Tracking Experiments

This package contains a self-contained organization for BlueROV2 path-tracking
experiments using PID, MPC and PPO controllers.

## Usage

Copy this folder to the root of the `bluerov2_gym` project and run:

```bash
python experiments/run_pid.py
python experiments/run_mpc.py
python experiments/run_ppo.py
```

Or run all experiments:

```bash
python experiments/run_all.py
```

## Outputs

CSV files are saved to:

```text
results/experiments/
```

The main metric is:

```text
tracking_error_m
```

which is the Euclidean position tracking error in meters.
