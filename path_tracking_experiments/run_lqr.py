"""Run the LQR path-tracking experiment."""

from env_utils import make_env
from experiment_runner import run_path_tracking_experiment
from lqr_controller import LQRController
from trajectories import FigureEightTrajectory


def main():
    env = make_env(render_mode=None)
    env.close()

    trajectory = FigureEightTrajectory()

    controller = LQRController(dt=0.1)

    run_path_tracking_experiment(
        controller=controller,
        trajectory=trajectory,
        output_csv="results/path_tracking/lqr.csv",
        steps=1000,
        repetitions=10,
        dt=0.1,
    )


if __name__ == "__main__":
    main()