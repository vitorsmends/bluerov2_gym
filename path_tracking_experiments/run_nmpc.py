"""Run the NMPC path-tracking experiment."""

from env_utils import make_env
from experiment_runner import run_path_tracking_experiment
from nmpc_controller import NMPCController
from trajectories import FigureEightTrajectory


def main():
    env = make_env(render_mode=None)
    dynamics = env.unwrapped.dynamics
    env.close()

    trajectory = FigureEightTrajectory()

    controller = NMPCController(
        trajectory=trajectory,
        dynamics=dynamics,
        dt=0.1,
        horizon=10,
        control_blocks=5,
    )

    run_path_tracking_experiment(
        controller=controller,
        trajectory=trajectory,
        output_csv="results/path_tracking/nmpc.csv",
        steps=1000,
        dt=0.1,
    )


if __name__ == "__main__":
    main()