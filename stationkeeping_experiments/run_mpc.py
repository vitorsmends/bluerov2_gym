"""Run station-keeping experiment with MPC control."""

from env_utils import make_env
from experiment_runner import run_stationkeeping_experiment
from mpc_controller import MPCController
from trajectories import StationKeepingReference


def main():
    # Temporary environment only to reuse the same Dynamics model inside MPC.
    env = make_env(render_mode=None)
    dynamics = env.unwrapped.dynamics

    reference = StationKeepingReference(x=0.0, y=0.0, z=-0.5, yaw=0.0)
    controller = MPCController(dynamics=dynamics, dt=0.1, horizon=8)

    env.close()

    run_stationkeeping_experiment(
        controller=controller,
        reference=reference,
        output_csv="results/stationkeeping_mpc.csv",
        steps=800,
        dt=0.1,
    )


if __name__ == "__main__":
    main()
