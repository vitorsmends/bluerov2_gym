"""Run the NMPC path-tracking experiment."""

from env_utils import make_env
from experiment_runner import run_path_tracking_experiment
from nmpc_controller import NMPCController
from trajectories import FigureEightTrajectory
from load_jonswap_config import load_jonswap_config


def main():
    jonswap_params = load_jonswap_config()

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
        repetitions=10,
        dt=0.1,
        jonswap_params=jonswap_params,
    )


if __name__ == "__main__":
    main()