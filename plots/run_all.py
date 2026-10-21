from plot_tracking_error import main as plot_tracking_error
from plot_rmse import main as plot_rmse
from plot_cpu_time import main as plot_cpu_time
from plot_energy import main as plot_energy
from plot_pareto import main as plot_pareto


def main():
    plot_tracking_error()
    plot_rmse()
    plot_cpu_time()
    plot_energy()
    plot_pareto()
    print("[OK] All plots were generated.")


if __name__ == "__main__":
    main()
