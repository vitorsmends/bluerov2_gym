import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    SCENARIOS, CONTROLLERS, OUTPUT_DIR,
    THRUSTER_COLUMNS, DPI, LINE_WIDTH, BAND_ALPHA,
)
from run_data import iter_csv_runs

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

t200_df = pd.read_csv("plots/t200_performance_16v.csv")

t200_thrust_n = t200_df["force_n"].to_numpy(dtype=float)
t200_power_w = t200_df["power_w"].to_numpy(dtype=float)

sort_idx = np.argsort(t200_thrust_n)
t200_thrust_n = t200_thrust_n[sort_idx]
t200_power_w = t200_power_w[sort_idx]


def estimate_total_power_w(df):
    thrust = df[THRUSTER_COLUMNS].to_numpy(dtype=float)
    shape = thrust.shape
    
    power_flat = np.interp(thrust.flatten(), t200_thrust_n, t200_power_w)
    thruster_power = power_flat.reshape(shape)
    
    return np.sum(thruster_power, axis=1)

def load_energy_runs(folder, prefix):
    runs = []

    for path, repetition_id, df in iter_csv_runs(
        folder, prefix, {"time", *THRUSTER_COLUMNS}
    ):
        time = df["time"].to_numpy(dtype=float)
        power = estimate_total_power_w(df)

        if len(time) < 2:
            continue
        step_duration = np.diff(time, append=time[-1] + np.median(np.diff(time)))
        step_duration[step_duration < 0.0] = 0.0
        cumulative_energy = np.cumsum(power * step_duration)

        runs.append(
            pd.Series(
                np.concatenate(([0.0], cumulative_energy)),
                index=np.concatenate(([time[0]], time + step_duration)),
                name=f"{path.stem}_rep{repetition_id}",
            )
        )

    return runs


def aggregate_runs(runs):
    if not runs:
        return None

    aligned = pd.concat(runs, axis=1).sort_index()

    time = aligned.index.to_numpy(dtype=float)
    mean = aligned.mean(axis=1).to_numpy(dtype=float)
    std = aligned.std(axis=1, ddof=0).fillna(0.0).to_numpy(dtype=float)

    return time, mean, std


def main():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for controller, info in CONTROLLERS.items():
        all_runs = []
        
        for scenario, folder in SCENARIOS.items():
            runs = load_energy_runs(folder, info["prefix"])
            all_runs.extend(runs)
            
        result = aggregate_runs(all_runs)

        if result is None:
            print(f"[WARN] No energy runs for {controller}")
            continue

        time, mean, std = result

        ax.plot(
            time,
            mean,
            color=info["color"],
            linewidth=LINE_WIDTH,
            label=controller,
        )
        ax.fill_between(
            time,
            mean - std,
            mean + std,
            color=info["color"],
            alpha=BAND_ALPHA,
            linewidth=0,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative energy [J]")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(
        loc="upper center",
        ncol=len(CONTROLLERS),
        frameon=True,
        bbox_to_anchor=(0.5, 1.15),
    )

    fig.tight_layout()

    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"cumulative_energy_mean_std.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()