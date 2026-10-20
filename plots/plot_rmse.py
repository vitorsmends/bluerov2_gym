import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import SCENARIOS, CONTROLLERS, OUTPUT_DIR, TABLE_DIR, DPI, FIGSIZE
from run_data import iter_csv_runs

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
                folder, info["prefix"], {"time", "tracking_error_m"}
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


def main():
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

    for scenario in ["Moderate", "Severe"]:
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
            positions=positions + SCENARIO_OFFSETS[scenario],
            widths=0.30,
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
            patch.set_hatch(SCENARIO_HATCH[scenario])
            patch.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(controllers)
    ax.set_ylabel("RMSE [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    legend = [
        Patch(facecolor="white", edgecolor="black", label="Moderate"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label="Severe",
        ),
    ]
    ax.legend(handles=legend, frameon=True, loc="best")

    fig.tight_layout()

    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"boxplot_rmse_error.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
