import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from config import SCENARIOS, CONTROLLERS, OUTPUT_DIR, TABLE_DIR, DPI, FIGSIZE
from run_data import iter_csv_runs

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})


def load_cpu_runs(folder, prefix):
    runs = []
    
    for path, repetition_id, df in iter_csv_runs(
        folder, prefix, {"time", "controller_cpu_time_s"}
    ):
        time = df["time"].to_numpy(dtype=float)
        cpu_ms = (
            df["controller_cpu_time_s"]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float) * 1e3
        )
        
        runs.append(
            pd.Series(
                cpu_ms,
                index=time,
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
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    cpu_series = {}

    for controller, info in CONTROLLERS.items():
        all_runs = []
        
        for scenario, folder in SCENARIOS.items():
            runs = load_cpu_runs(folder, info["prefix"])
            all_runs.extend(runs)
            
        result = aggregate_runs(all_runs)
        
        if result is None:
            print(f"[WARN] No CPU time runs found for {controller}")
            continue
            
        time, mean_v, std_v = result
        cpu_series[controller] = (time, mean_v, std_v)

        ax.plot(
            time,
            mean_v,
            color=info["color"],
            linewidth=1.8,
            label=controller,
        )
        ax.fill_between(
            time,
            np.maximum(mean_v - std_v, 0.0),
            mean_v + std_v,
            color=info["color"],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CPU Time [ms]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    x1, x2 = 10.0, 15.0
    zoom_controllers = [name for name in ["PID", "SMC", "PPO"] if name in cpu_series]
    zoom_values = []

    for name in zoom_controllers:
        t, mean_v, std_v = cpu_series[name]
        mask = (t >= x1) & (t <= x2)
        
        if np.any(mask):
            zoom_values.extend((mean_v[mask] - std_v[mask]).tolist())
            zoom_values.extend((mean_v[mask] + std_v[mask]).tolist())

    if zoom_values:
        y1 = max(0.0, np.nanmin(zoom_values) * 0.95)
        y2 = np.nanmax(zoom_values) * 1.05

        axins = inset_axes(
            ax,
            width="32%",
            height="32%",
            loc="upper center",
            borderpad=1.2,
        )

        for controller in cpu_series:
            t, mean_v, std_v = cpu_series[controller]
            info = CONTROLLERS[controller]
            
            axins.plot(
                t,
                mean_v,
                color=info["color"],
                linewidth=1.3,
            )
            axins.fill_between(
                t,
                np.maximum(mean_v - std_v, 0.0),
                mean_v + std_v,
                color=info["color"],
                alpha=0.18,
                linewidth=0,
            )

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=8)

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0)

    fig.tight_layout()

    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"fig_path_tracking_cpu_time_mean_std.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()