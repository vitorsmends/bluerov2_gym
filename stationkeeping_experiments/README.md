# BlueROV2 Station-Keeping Experiments

This folder contains a self-contained organization for station-keeping experiments
using the updated BlueROV2 numerical dynamics.

## Files

- `trajectories.py`: reference definitions, including `StationKeepingReference`.
- `env_utils.py`: environment loading, observation conversion, error metrics, CSV helpers.
- `base_controller.py`: shared controller interface.
- `pid_controller.py`: PID controller mapped to direct thruster commands.
- `mpc_controller.py`: MPC controller optimizing direct thruster commands.
- `ppo_controller.py`: trained PPO policy wrapper.
- `experiment_runner.py`: common simulation loop and CSV logging.
- `run_pid.py`: run the PID station-keeping experiment.
- `run_mpc.py`: run the MPC station-keeping experiment.
- `run_ppo.py`: run the PPO station-keeping experiment.
- `run_all.py`: run all experiments sequentially.

## How to install in your project

Copy this whole folder to the root of your project:

```bash
~/Workspaces/master_ws/src/bluerov2_gym/stationkeeping_experiments
```

## How to run

From the project root:

```bash
cd ~/Workspaces/master_ws/src/bluerov2_gym/stationkeeping_experiments
python run_pid.py
python run_mpc.py
python run_ppo.py
```

Or run all:

```bash
python run_all.py
```

No `PYTHONPATH=.` is required.

## Output

The results are saved in:

```text
stationkeeping_experiments/results/stationkeeping_pid.csv
stationkeeping_experiments/results/stationkeeping_mpc.csv
stationkeeping_experiments/results/stationkeeping_ppo.csv
```

Each CSV includes the Euclidean position error in meters under the column:

```text
position_error_m
```
