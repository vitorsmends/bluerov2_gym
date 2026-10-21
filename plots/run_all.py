from plot_tracking_error import main as plot_tracking_error
from plot_rmse import main as plot_rmse
from plot_cpu_time import main as plot_cpu_time
from plot_energy import main as plot_energy
from plot_pareto import main as plot_pareto


def main():
    from config import SCENARIOS, CONTROLLERS
    import pandas as pd

    # Validate all inputs before writing figures: avoid silently mixing partial
    # campaigns or producing CPU plots from legacy CSVs without timings.
    for folder in SCENARIOS.values():
        for info in CONTROLLERS.values():
            for trajectory in ("circle", "square", "figure_eight_depth", "letter_b", "star"):
                path = folder / f"{info['prefix']}{trajectory}.csv"
                frame = pd.read_csv(path)
                required = {"time", "tracking_error_m", "controller_cpu_time_s", "seed", "repetition_id", *[f"T{i}" for i in range(1, 7)]}
                if frame.empty or required - set(frame.columns):
                    raise ValueError(f"Incomplete benchmark input: {path}")
    plot_tracking_error()
    plot_rmse()
    plot_cpu_time()
    plot_energy()
    plot_pareto()
    print("[OK] All plots were generated.")


if __name__ == "__main__":
    main()
