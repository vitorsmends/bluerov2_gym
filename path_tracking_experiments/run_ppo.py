"""Run the PPO path-tracking experiment."""

from experiment_runner import run_path_tracking_experiment
from ppo_controller import PPOController
from trajectories import FigureEightTrajectory


def main():
    trajectory = FigureEightTrajectory()
    controller = PPOController(
        model_path="ppo_trajectory_final",
        vecnormalize_path="vec_normalize.pkl",
    )

    run_path_tracking_experiment(
        controller=controller,
        trajectory=trajectory,
        output_csv="results/path_tracking/ppo.csv",
        steps=1000,
        repetitions=10,
        dt=0.1,
    )


if __name__ == "__main__":
    main()
