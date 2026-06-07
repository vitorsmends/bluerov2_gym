"""Run the SMC path-tracking experiment."""

from env_utils import make_env
from experiment_runner import run_path_tracking_experiment
from smc_controller import SMCController
from trajectories import FigureEightTrajectory


def main():
    env = make_env(render_mode=None)
    dynamics = env.unwrapped.dynamics
    env.close()

    trajectory = FigureEightTrajectory()

    controller = SMCController(
        dynamics=dynamics,
        dt=0.1,
    )

    run_path_tracking_experiment(
        controller=controller,
        trajectory=trajectory,
        output_csv="results/path_tracking/smc.csv",
        steps=1000,
        repetitions=10,
        dt=0.1,
    )


if __name__ == "__main__":
    main()