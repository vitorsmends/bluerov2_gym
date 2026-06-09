"""Run the LQR path-tracking experiment."""

from experiment_runner import run_path_tracking_experiment
from lqr_controller import LQRController
from trajectories import FigureEightTrajectory
from load_jonswap_config import load_jonswap_config


def main():
    jonswap_params = load_jonswap_config()

    trajectory = FigureEightTrajectory()

    controller = LQRController(dt=0.1)

    run_path_tracking_experiment(
        controller=controller,
        trajectory=trajectory,
        output_csv="results/path_tracking/lqr.csv",
        steps=1000,
        repetitions=10,
        dt=0.1,
        jonswap_params=jonswap_params,
    )


if __name__ == "__main__":
    main()