"""Run the PPO path-tracking experiment."""

from experiment_runner import run_path_tracking_experiment
from ppo_controller import PPOController
from trajectories import FigureEightTrajectory
from load_jonswap_config import load_jonswap_config

import yaml
from pathlib import Path

yaml_path = Path("path_tracking_experiments/jonswap_config.yaml")
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)
    config_default = config.get("default_scenario")


def main():
    jonswap_params = load_jonswap_config()

    trajectory = FigureEightTrajectory()

    controller = PPOController(
        model_path="ppo_trajectory_final",
        vecnormalize_path="vec_normalize.pkl",
    )

    run_path_tracking_experiment(
        controller=controller,
        trajectory=trajectory,
        output_csv="results-{}/path_tracking/ppo.csv".format(config_default),
        steps=1000,
        repetitions=10,
        dt=0.1,
        jonswap_params=jonswap_params,
    )


if __name__ == "__main__":
    main()