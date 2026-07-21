# BlueROV2 Gymnasium

A Gymnasium-compatible simulation environment for the **BlueROV2**
autonomous underwater vehicle (AUV). The project provides a realistic
six-degree-of-freedom dynamic model, configurable ocean disturbances,
reinforcement learning support, classical control benchmarks, and an
experiment framework for reproducible controller evaluation.

## Features

-   Six-DoF BlueROV2 dynamic model
-   Gymnasium-compatible API
-   Stable-Baselines3 integration
-   MeshCat visualization
-   Ocean current and JONSWAP disturbances
-   PID, PPO, SMC and NMPC controllers
-   Experiment framework with YAML configuration
-   Automatic logging, CSV export and plotting

## Installation

``` bash
git clone https://github.com/gokulp01/bluerov2_gym.git
cd bluerov2_gym
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

``` python
import gymnasium as gym
import bluerov2_gym

env = gym.make('BlueRov-v0')
obs, info = env.reset()
```

## Experiment Framework

Run:

``` bash
python experiments/run_experiment.py
```

Implemented controllers:

-   PID
-   PPO
-   Sliding Mode Control (SMC)
-   Nonlinear Model Predictive Control (NMPC)

Configuration files:

-   experiments/config/experiment.yaml
-   experiments/config/ocean_environment.yaml

The framework automatically executes experiments, records metrics,
exports CSV files and generates plots.

## Environment

State:

    [x,y,z,roll,pitch,yaw,u,v,w,p,q,r]

Action:

    [t1,t2,t3,t4,t5,t6]

## Repository Structure

    bluerov2_gym/
    ├── bluerov2_gym/
    ├── experiments/
    ├── examples/
    ├── tests/
    └── README.md

## License

MIT License.
