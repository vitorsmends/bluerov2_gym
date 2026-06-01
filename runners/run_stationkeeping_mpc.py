from experiments_common.experiment import run_stationkeeping_experiment
from experiments_common.trajectories import StationKeepingReference
from experiments_common.controllers.mpc_controller import MPCController


def main():
    reference = StationKeepingReference(
        x=0.0,
        y=0.0,
        z=-0.5,
        yaw=0.0,
    )

    controller = MPCController(reference=reference)

    run_stationkeeping_experiment(
        controller=controller,
        reference=reference,
        output_csv="results/stationkeeping_mpc.csv",
        steps=800,
        dt=0.1,
    )


if __name__ == "__main__":
    main()
