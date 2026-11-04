from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(settings=None):
    from ..paths import ROOT
    settings = settings or {}
    output = Path(settings.get('output_dir', ROOT))
    output.mkdir(parents=True, exist_ok=True)
    nominal = pd.read_csv(
        settings.get("nominal", ROOT / "results/nominal/ppo_figure_eight_depth.csv")
    )
    
    ftc = pd.read_csv(
        settings.get("ftc", ROOT / "results/ftc/ppo_figure_eight_depth.csv")
    )
    
    
    plt.figure(figsize=(7,6))
    
    plt.plot(
        nominal["x_ref"],
        nominal["y_ref"],
        "k--",
        label="Reference",
    )
    
    plt.plot(
        nominal["x"],
        nominal["y"],
        label="PPO",
    )
    
    plt.plot(
        ftc["x"],
        ftc["y"],
        label="PPO-FTC",
    )
    
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(
        output / "comparison_trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    
    if settings.get("show", False):
        plt.show()
    
    
    rmse_nominal = nominal["tracking_error_m"].mean()
    rmse_ftc = ftc["tracking_error_m"].mean()
    
    print()
    print("Tracking RMSE")
    print("---------------------------")
    print(f"PPO      : {rmse_nominal:.3f}")
    print(f"PPO FTC  : {rmse_ftc:.3f}")


if __name__ == "__main__":
    main({"show": True})
