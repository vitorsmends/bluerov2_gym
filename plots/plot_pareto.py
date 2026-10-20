import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import SCENARIOS, CONTROLLERS, OUTPUT_DIR, TABLE_DIR, DPI, FIGSIZE

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

MARKERS = {"Moderate": "o", "Severe": "s"}


def compute_table():
    rows = []

    for scenario, folder in SCENARIOS.items():
        for controller, info in CONTROLLERS.items():
            for path in sorted(folder.glob(f"{info['prefix']}*.csv")):
                source = pd.read_csv(path)
                if "repetition_id" not in source.columns:
                    source["repetition_id"] = 0

                required = {"time", "tracking_error_m", "controller_cpu_time_s"}
                missing = required - set(source.columns)
                if missing:
                    print(f"[WARN] Ignored {path}: missing {sorted(missing)}")
                    continue

                for repetition_id, df in source.groupby("repetition_id", sort=True):
                    error = df["tracking_error_m"].to_numpy(dtype=float)
                    cpu_ms = (
                        df["controller_cpu_time_s"]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .to_numpy(dtype=float)
                        * 1e3
                    )

                    if len(cpu_ms) == 0:
                        continue

                    rows.append({
                        "scenario": scenario,
                        "controller": controller,
                        "trajectory": path.stem.replace(info["prefix"], "", 1),
                        "repetition_id": repetition_id,
                        "rmse_error_m": float(np.sqrt(np.nanmean(error**2))),
                        "cpu_mean_ms": float(np.mean(cpu_ms)),
                    })

    return pd.DataFrame(rows)


def main():
    table = compute_table()

    if table.empty:
        raise RuntimeError("No valid Pareto data were found.")

    table.to_csv(TABLE_DIR / "pareto_per_trajectory.csv", index=False)

    summary = (
        table.groupby(["scenario", "controller"], as_index=False)
        .agg(
            rmse_mean=("rmse_error_m", "mean"),
            rmse_std=("rmse_error_m", lambda x: np.std(x, ddof=0)),
            cpu_mean=("cpu_mean_ms", "mean"),
            cpu_std=("cpu_mean_ms", lambda x: np.std(x, ddof=0)),
        )
    )
    summary.to_csv(TABLE_DIR / "pareto_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for controller, info in CONTROLLERS.items():
        current = (
            summary[summary["controller"] == controller]
            .set_index("scenario")
            .reindex(["Moderate", "Severe"])
            .dropna()
        )

        if current.empty:
            continue

        if len(current) == 2:
            ax.plot(
                current["cpu_mean"],
                current["rmse_mean"],
                color=info["color"],
                linewidth=1.2,
                alpha=0.8,
            )

        for scenario, row in current.iterrows():
            ax.errorbar(
                row["cpu_mean"],
                row["rmse_mean"],
                xerr=row["cpu_std"],
                yerr=row["rmse_std"],
                fmt=MARKERS[scenario],
                color=info["color"],
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=7,
                capsize=3,
                label=f"{controller} ({scenario})",
            )

    ax.set_xlabel("Mean CPU time per step [ms]")
    ax.set_ylabel("RMSE [m]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(
        frameon=True,
        fontsize=8,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"pareto_rmse_vs_cpu_time_mean_std.{extension}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
