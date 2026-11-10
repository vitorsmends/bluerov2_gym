from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .config import SCENARIOS, CONTROLLERS, OUTPUT_DIR, TABLE_DIR, DPI, FIGSIZE
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
    os.makedirs(TABLE_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })

SCENARIO_OFFSETS = {"Moderate": -0.18, "Severe": 0.18}
SCENARIO_HATCH = {"Moderate": "", "Severe": "///"}


def compute_rmse_table():
    rows = []

    for scenario, folder in SCENARIOS.items():
        for controller, info in CONTROLLERS.items():
            for path, repetition_id, df in iter_csv_runs(
                folder, info["prefix"], {"time", "tracking_error_m"}, trajectories=TRAJECTORIES
            ):
                error = df["tracking_error_m"].to_numpy(dtype=float)

                rows.append({
                    "scenario": scenario,
                    "controller": controller,
                    "trajectory": path.stem.replace(info["prefix"], "", 1),
                    "repetition_id": repetition_id,
                    "rmse_error_m": float(np.sqrt(np.nanmean(error**2))),
                })

    return pd.DataFrame(rows)


def main(settings=None):
    configure(settings)
    table = compute_rmse_table()

    if table.empty:
        raise RuntimeError("No valid CSV files were found.")

    table.to_csv(TABLE_DIR / "rmse_per_trajectory.csv", index=False)

    controllers = [
        controller
        for controller in CONTROLLERS
        if controller in table["controller"].unique()
    ]
    positions = np.arange(len(controllers), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    scenarios = list(SCENARIOS)
    offsets = [0.] if len(scenarios) == 1 else np.linspace(-.18, .18, len(scenarios))
    widths = .30 if len(scenarios) <= 2 else .6 / len(scenarios)
    if len(scenarios) > 2:
        offsets = np.linspace(-.24, .24, len(scenarios))
        widths = .8 * (offsets[1] - offsets[0])
    hatches = {name: SCENARIO_HATCH.get(name, ['', '///', '...', 'xx'][i % 4])
               for i, name in enumerate(scenarios)}
    if len(scenarios) > 2:
        hatches = {name: ['', '///', 'xx', '...'][i % 4]
                   for i, name in enumerate(scenarios)}
    for scenario, offset in zip(scenarios, offsets):
        data = [
            table.loc[
                (table["controller"] == controller)
                & (table["scenario"] == scenario),
                "rmse_error_m",
            ].dropna().to_numpy()
            for controller in controllers
        ]

        boxes = ax.boxplot(
            data,
            positions=positions + offset,
            widths=widths,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            medianprops={"color": "black", "linewidth": 1.2},
        )

        for patch, controller in zip(boxes["boxes"], controllers):
            patch.set_facecolor(CONTROLLERS[controller]["color"])
            patch.set_alpha(0.55)
            patch.set_hatch(hatches[scenario])
            patch.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(controllers)
    ax.set_ylabel("RMSE [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    legend = [Patch(facecolor="white", edgecolor="black", hatch=hatches[name], label=name)
              for name in scenarios]
    ax.legend(handles=legend, frameon=True, loc="best")

    fig.tight_layout()

    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"boxplot_rmse_error.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
