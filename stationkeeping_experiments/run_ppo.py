"""Run station-keeping experiment with a trained PPO controller."""

from experiment_runner import run_stationkeeping_experiment
from ppo_controller import PPOController
from trajectories import StationKeepingReference


def main():
    reference = StationKeepingReference(x=0.0, y=0.0, z=-0.5, yaw=0.0)

    controller = PPOController(
        model_path="bluerov_ppo",
        vecnormalize_path="bluerov_vec_normalize.pkl",
        deterministic=True,
    )

    run_stationkeeping_experiment(
        controller=controller,
        reference=reference,
        output_csv="results/stationkeeping_ppo.csv",
        steps=800,
        dt=0.1,
    )


if __name__ == "__main__":
    main()
