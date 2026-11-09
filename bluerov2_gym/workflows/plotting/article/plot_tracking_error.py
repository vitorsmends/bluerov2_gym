from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    SCENARIOS, CONTROLLERS, OUTPUT_DIR,
    DPI, LINE_WIDTH, BAND_ALPHA, FIGSIZE,
)
from .run_data import iter_csv_runs



TRAJECTORIES = None

def configure(settings=None):
    global OUTPUT_DIR, TABLE_DIR, SCENARIOS, CONTROLLERS, TRAJECTORIES
    from . import config as defaults
    settings = settings or {}
    TRAJECTORIES = settings.get("trajectory_names")
    OUTPUT_DIR = Path(settings.get('output_dir', defaults.OUTPUT_DIR))
    TABLE_DIR = Path(settings.get('table_dir', defaults.TABLE_DIR))
    SCENARIOS = {k: Path(v) for k,v in settings.get('scenarios', defaults.SCENARIOS).items()}
    CONTROLLERS = defaults.CONTROLLERS
    if settings.get('controllers') is not None:
        selected = {v.lower() for v in settings['controllers']}
        CONTROLLERS = {k:v for k,v in CONTROLLERS.items() if k.lower() in selected}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })


def load_controller_runs(folder, prefix):
    runs = []

    for path, repetition_id, df in iter_csv_runs(
        folder, prefix, {"time", "tracking_error_m"}, trajectories=TRAJECTORIES
    ):
        run = df[["time", "tracking_error_m"]].copy()
        run.attrs["run_id"] = f"{path.stem}_rep{repetition_id}"
        runs.append(run)

    return runs


def aggregate_runs(runs, column='tracking_error_m'):
    if not runs:
        return None

    series = []

    for index, df in enumerate(runs):
        current = (
            df.groupby("time", as_index=True)[column]
            .mean()
            .rename(df.attrs.get("run_id", f"run_{index}"))
        )
        series.append(current)

    aligned = pd.concat(series, axis=1).sort_index()

    time = aligned.index.to_numpy(dtype=float)
    mean = aligned.mean(axis=1).to_numpy(dtype=float)
    std = aligned.std(axis=1, ddof=0).fillna(0.0).to_numpy(dtype=float)

    return time, mean, std


def main(settings=None):
    configure(settings)
    fig, axes = plt.subplots(
        len(SCENARIOS), 1,
        figsize=(FIGSIZE[0], FIGSIZE[1] * len(SCENARIOS)),
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes)
    for ax, (scenario, folder) in zip(axes, SCENARIOS.items()):
        for controller, info in CONTROLLERS.items():
            runs = load_controller_runs(folder, info["prefix"])
            result = aggregate_runs(runs)

            if result is None:
                print(f"[WARN] No runs for {scenario}/{controller}")
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

        ax.set_ylabel("Tracking error [m]")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time [s]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90 if len(SCENARIOS) == 1 else 0.95))

    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"path_tracking_error_mean_std.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
