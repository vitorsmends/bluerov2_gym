"""Run station-keeping experiment with PID control."""

from env_utils import make_env
from experiment_runner import run_stationkeeping_experiment
from pid_controller import PIDController
from trajectories import StationKeepingReference


def main():
    # Temporary environment only to read the allocation matrix used by Dynamics.
    env = make_env(render_mode=None)
    allocation_matrix = env.unwrapped.dynamics.allocation_matrix.copy()
    env.close()

    reference = StationKeepingReference(x=0.0, y=0.0, z=-0.5, yaw=0.0)
    controller = PIDController(allocation_matrix=allocation_matrix)

    run_stationkeeping_experiment(
        controller=controller,
        reference=reference,
        output_csv="results/stationkeeping_pid.csv",
        steps=800,
        dt=0.1,
    )


if __name__ == "__main__":
    main()
