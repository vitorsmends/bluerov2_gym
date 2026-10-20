
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("experiments/ocean_environment.yaml", "r") as f:
    config = yaml.safe_load(f)
    config_default = config["default_scenario"]

INPUT_DIR = f"results-{config_default}/experiments"
OUTPUT_DIR = f"results-{config_default}/plots_path_tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "PID":  {"file": "pid.csv",  "color": "#4E79A7"},
    "SMC":  {"file": "smc.csv",  "color": "#B07AA1"},
    "NMPC": {"file": "nmpc.csv", "color": "#E15759"},
    "PPO":  {"file": "ppo.csv",  "color": "#59A14F"},
}

def save_plot(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"),
                dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {name}")

def load_rmse():
    rmse = {}
    for ctrl, cfg in CONFIG.items():
        path = os.path.join(INPUT_DIR, cfg["file"])
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue

        df = pd.read_csv(path)

        if "repetition_id" not in df.columns:
            df["repetition_id"] = 0

        if "tracking_error_m" not in df.columns:
            df["tracking_error_m"] = np.linalg.norm(
                df[["x","y","z"]].values -
                df[["x_ref","y_ref","z_ref"]].values,
                axis=1,
            )

        values = []

        for _, g in df.groupby("repetition_id"):
            err = g["tracking_error_m"].to_numpy(dtype=float)
            rmse_value = np.sqrt(np.mean(err**2))
            values.append(max(rmse_value, 1e-12))

        rmse[ctrl] = np.asarray(values)

    return rmse

def plot_raincloud(rmse):

    labels = list(rmse.keys())
    colors = [CONFIG[c]["color"] for c in labels]

    # log10 transform
    data = [np.log10(rmse[c]) for c in labels]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(6.5,4.2))

    vp = ax.violinplot(
        data,
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.35,
    )

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)
        body.set_linewidth(0.8)

    bp = ax.boxplot(
        data,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        boxprops=dict(facecolor="white",
                      edgecolor="black",
                      linewidth=1.0),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):

        x = rng.normal(i, 0.035, len(values))

        ax.scatter(
            x,
            values,
            s=30,
            c="black",
            alpha=0.7,
            zorder=10,
        )

    ymin = np.floor(min(np.min(v) for v in data))
    ymax = np.ceil(max(np.max(v) for v in data))

    ticks = np.arange(ymin, ymax+1)

    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{10**t:.3g}" for t in ticks])

    ax.set_xticks(np.arange(1,len(labels)+1))
    ax.set_xticklabels(labels)

    ax.set_ylabel("RMSE Tracking Error [m]")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_plot("raincloud_rmse")

    plt.show()

def main():
    rmse = load_rmse()
    plot_raincloud(rmse)

if __name__ == "__main__":
    main()
