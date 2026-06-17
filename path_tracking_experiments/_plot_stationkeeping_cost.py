import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml
from pathlib import Path

with open("path_tracking_experiments/jonswap_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    config_default = config.get("default_scenario")

INPUT_DIR = f"results-{config_default}/stationkeeping"
OUTPUT_DIR = f"results-{config_default}/plots_stationkeeping_cost_tradeoff"
TABLE_DIR = f"results-{config_default}/tables_stationkeeping_cost_tradeoff"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


CONFIG = {
    "PID":  {"file": "pid_stationkeeping.csv",  "color": "#4E79A7"},
    "SMC":  {"file": "smc_stationkeeping.csv",  "color": "#B07AA1"},
    "NMPC": {"file": "nmpc_stationkeeping.csv", "color": "#E15759"},
    "PPO":  {"file": "ppo_stationkeeping.csv",  "color": "#59A14F"},
}


def save_plot(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {name}")


def save_table(df, name):
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Saved table: {path}")


def load_data():
    dfs = {}

    required = {
        "controller",
        "sea_scenario",
        "target_id",
        "repetition_id",
        "position_error_m",
        "yaw_error",
        "control_effort",
        "total_power_W",
        "step_energy_J",
        "controller_wall_time_s",
        "controller_cpu_time_s",
        "controller_frequency_hz",
    }

    for controller, cfg in CONFIG.items():
        path = os.path.join(INPUT_DIR, cfg["file"])

        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] {controller} skipped. Missing columns: {sorted(missing)}")
            continue

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[
            "position_error_m",
            "controller_wall_time_s",
            "controller_cpu_time_s",
        ]).copy()

        if df.empty:
            print(f"[WARN] {controller} skipped. No valid metric rows.")
            continue

        df["controller"] = controller
        df["controller_wall_time_ms"] = df["controller_wall_time_s"] * 1e3
        df["controller_cpu_time_ms"] = df["controller_cpu_time_s"] * 1e3
        df["abs_yaw_error_rad"] = np.abs(df["yaw_error"])

        dfs[controller] = df

    if not dfs:
        raise RuntimeError("No valid station-keeping data found.")

    return dfs


def compute_case_metrics(dfs):
    rows = []

    group_cols = [
        "sea_scenario",
        "target_id",
        "repetition_id",
    ]

    for controller, df in dfs.items():
        for keys, g in df.groupby(group_cols):
            sea_scenario, target_id, repetition_id = keys

            err = g["position_error_m"].to_numpy(dtype=float)
            yaw = g["abs_yaw_error_rad"].to_numpy(dtype=float)
            effort = g["control_effort"].to_numpy(dtype=float)
            power = g["total_power_W"].to_numpy(dtype=float)
            energy = g["step_energy_J"].to_numpy(dtype=float)
            wall = g["controller_wall_time_ms"].to_numpy(dtype=float)
            cpu = g["controller_cpu_time_ms"].to_numpy(dtype=float)

            rmse = float(np.sqrt(np.nanmean(err**2)))
            mean_error = float(np.nanmean(err))
            p95_error = float(np.nanpercentile(err, 95))
            max_error = float(np.nanmax(err))

            mean_yaw = float(np.nanmean(yaw))
            p95_yaw = float(np.nanpercentile(yaw, 95))

            mean_wall_ms = float(np.nanmean(wall))
            p95_wall_ms = float(np.nanpercentile(wall, 95))
            total_wall_ms = float(np.nansum(wall))

            mean_cpu_ms = float(np.nanmean(cpu))
            p95_cpu_ms = float(np.nanpercentile(cpu, 95))
            total_cpu_ms = float(np.nansum(cpu))

            total_energy = float(np.nansum(energy))
            mean_power = float(np.nanmean(power))
            mean_effort = float(np.nanmean(effort))

            rmse_cpu_product = rmse * mean_cpu_ms
            rmse_wall_product = rmse * mean_wall_ms
            rmse_energy_product = rmse * total_energy

            rows.append({
                "controller": controller,
                "sea_scenario": sea_scenario,
                "target_id": target_id,
                "repetition_id": repetition_id,
                "n_steps": int(len(g)),

                "rmse_position_error_m": rmse,
                "mean_position_error_m": mean_error,
                "p95_position_error_m": p95_error,
                "max_position_error_m": max_error,

                "mean_abs_yaw_error_rad": mean_yaw,
                "p95_abs_yaw_error_rad": p95_yaw,

                "mean_control_effort": mean_effort,
                "mean_power_W": mean_power,
                "total_energy_J": total_energy,

                "mean_wall_time_ms": mean_wall_ms,
                "p95_wall_time_ms": p95_wall_ms,
                "total_wall_time_ms": total_wall_ms,

                "mean_cpu_time_ms": mean_cpu_ms,
                "p95_cpu_time_ms": p95_cpu_ms,
                "total_cpu_time_ms": total_cpu_ms,

                "rmse_x_cpu_ms": rmse_cpu_product,
                "rmse_x_wall_ms": rmse_wall_product,
                "rmse_x_energy_J": rmse_energy_product,

                "performance_per_cpu_cost": (
                    1.0 / rmse_cpu_product if rmse_cpu_product > 0 else np.nan
                ),
                "performance_per_wall_cost": (
                    1.0 / rmse_wall_product if rmse_wall_product > 0 else np.nan
                ),
                "performance_per_energy_cost": (
                    1.0 / rmse_energy_product if rmse_energy_product > 0 else np.nan
                ),
            })

    case_df = pd.DataFrame(rows)
    save_table(case_df, "stationkeeping_cost_tradeoff_per_case")

    return case_df


def normalize_minmax(series, lower_is_better=True):
    values = series.to_numpy(dtype=float)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        return np.ones_like(values)

    if lower_is_better:
        return (vmax - values) / (vmax - vmin)

    return (values - vmin) / (vmax - vmin)


def add_composite_score(case_df):
    case_df = case_df.copy()

    case_df["score_position"] = normalize_minmax(
        case_df["rmse_position_error_m"],
        lower_is_better=True,
    )

    case_df["score_yaw"] = normalize_minmax(
        case_df["mean_abs_yaw_error_rad"],
        lower_is_better=True,
    )

    case_df["score_cpu"] = normalize_minmax(
        case_df["mean_cpu_time_ms"],
        lower_is_better=True,
    )

    case_df["score_wall"] = normalize_minmax(
        case_df["mean_wall_time_ms"],
        lower_is_better=True,
    )

    case_df["score_energy"] = normalize_minmax(
        case_df["total_energy_J"],
        lower_is_better=True,
    )

    case_df["stationkeeping_cost_score"] = (
        0.45 * case_df["score_position"]
        + 0.15 * case_df["score_yaw"]
        + 0.15 * case_df["score_cpu"]
        + 0.10 * case_df["score_wall"]
        + 0.15 * case_df["score_energy"]
    )

    return case_df


def build_summary(case_df):
    metric_cols = [
        "rmse_position_error_m",
        "mean_position_error_m",
        "p95_position_error_m",
        "max_position_error_m",
        "mean_abs_yaw_error_rad",
        "p95_abs_yaw_error_rad",
        "mean_control_effort",
        "mean_power_W",
        "total_energy_J",
        "mean_wall_time_ms",
        "p95_wall_time_ms",
        "mean_cpu_time_ms",
        "p95_cpu_time_ms",
        "rmse_x_cpu_ms",
        "rmse_x_wall_ms",
        "rmse_x_energy_J",
        "performance_per_cpu_cost",
        "performance_per_wall_cost",
        "performance_per_energy_cost",
        "stationkeeping_cost_score",
    ]

    rows = []

    for controller, g in case_df.groupby("controller"):
        row = {
            "controller": controller,
            "n_cases": int(len(g)),
            "n_sea_scenarios": int(g["sea_scenario"].nunique()),
            "n_targets": int(g["target_id"].nunique()),
            "n_repetitions": int(g["repetition_id"].nunique()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary["rank_position_rmse"] = (
        summary["rmse_position_error_m_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_cpu"] = (
        summary["mean_cpu_time_ms_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_energy"] = (
        summary["total_energy_J_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_tradeoff"] = (
        summary["stationkeeping_cost_score_mean"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    save_table(summary, "stationkeeping_cost_tradeoff_summary")

    return summary


def build_summary_by_sea(case_df):
    rows = []

    metric_cols = [
        "rmse_position_error_m",
        "mean_abs_yaw_error_rad",
        "mean_cpu_time_ms",
        "mean_wall_time_ms",
        "total_energy_J",
        "rmse_x_cpu_ms",
        "stationkeeping_cost_score",
    ]

    for (controller, sea), g in case_df.groupby(["controller", "sea_scenario"]):
        row = {
            "controller": controller,
            "sea_scenario": sea,
            "n_cases": int(len(g)),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))

        rows.append(row)

    sea_summary = pd.DataFrame(rows)
    save_table(sea_summary, "stationkeeping_cost_tradeoff_by_sea")

    return sea_summary


def save_latex_table(summary):
    cols = [
        "controller",
        "rmse_position_error_m_mean",
        "rmse_position_error_m_std",
        "mean_abs_yaw_error_rad_mean",
        "mean_cpu_time_ms_mean",
        "total_energy_J_mean",
        "rmse_x_cpu_ms_mean",
        "stationkeeping_cost_score_mean",
        "rank_tradeoff",
    ]

    latex_df = summary[cols].copy().round(4)

    print("\n=== LATEX TABLE ===\n")
    print(latex_df.to_latex(index=False))


def plot_pareto_cpu(summary):
    plt.figure(figsize=(6.5, 4.2))

    for _, row in summary.iterrows():
        ctrl = row["controller"]
        color = CONFIG[ctrl]["color"]

        plt.errorbar(
            row["mean_cpu_time_ms_mean"],
            row["rmse_position_error_m_mean"],
            xerr=row["mean_cpu_time_ms_std"],
            yerr=row["rmse_position_error_m_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=ctrl,
        )

        plt.text(
            row["mean_cpu_time_ms_mean"] * 1.04,
            row["rmse_position_error_m_mean"],
            ctrl,
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Mean CPU Time [ms/step]")
    plt.ylabel("RMSE Position Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("stationkeeping_pareto_rmse_vs_cpu_time")


def plot_pareto_wall(summary):
    plt.figure(figsize=(6.5, 4.2))

    for _, row in summary.iterrows():
        ctrl = row["controller"]
        color = CONFIG[ctrl]["color"]

        plt.errorbar(
            row["mean_wall_time_ms_mean"],
            row["rmse_position_error_m_mean"],
            xerr=row["mean_wall_time_ms_std"],
            yerr=row["rmse_position_error_m_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=ctrl,
        )

        plt.text(
            row["mean_wall_time_ms_mean"] * 1.04,
            row["rmse_position_error_m_mean"],
            ctrl,
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Mean Wall Time [ms/step]")
    plt.ylabel("RMSE Position Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("stationkeeping_pareto_rmse_vs_wall_time")


def plot_pareto_energy(summary):
    plt.figure(figsize=(6.5, 4.2))

    for _, row in summary.iterrows():
        ctrl = row["controller"]
        color = CONFIG[ctrl]["color"]

        plt.errorbar(
            row["total_energy_J_mean"],
            row["rmse_position_error_m_mean"],
            xerr=row["total_energy_J_std"],
            yerr=row["rmse_position_error_m_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=ctrl,
        )

        plt.text(
            row["total_energy_J_mean"] * 1.04,
            row["rmse_position_error_m_mean"],
            ctrl,
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Total Energy [J]")
    plt.ylabel("RMSE Position Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("stationkeeping_pareto_rmse_vs_energy")


def plot_tradeoff_score(summary):
    ordered = summary.sort_values(
        "stationkeeping_cost_score_mean",
        ascending=False,
    )

    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))

    plt.bar(
        ordered["controller"],
        ordered["stationkeeping_cost_score_mean"],
        yerr=ordered["stationkeeping_cost_score_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel("Station-Keeping Performance-Cost Score [-]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("stationkeeping_cost_score")


def plot_rmse_cpu_product(summary):
    ordered = summary.sort_values("rmse_x_cpu_ms_mean")
    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))

    plt.bar(
        ordered["controller"],
        ordered["rmse_x_cpu_ms_mean"],
        yerr=ordered["rmse_x_cpu_ms_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel(r"RMSE $\times$ CPU Time [m ms]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("stationkeeping_rmse_cpu_product")


def plot_rmse_energy_product(summary):
    ordered = summary.sort_values("rmse_x_energy_J_mean")
    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))

    plt.bar(
        ordered["controller"],
        ordered["rmse_x_energy_J_mean"],
        yerr=ordered["rmse_x_energy_J_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel(r"RMSE $\times$ Energy [m J]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("stationkeeping_rmse_energy_product")


def plot_boxplot(case_df, metric, ylabel, filename):
    data = []
    labels = []
    colors = []

    for ctrl in CONFIG:
        g = case_df[case_df["controller"] == ctrl]

        if g.empty:
            continue

        data.append(g[metric].dropna().to_numpy(dtype=float))
        labels.append(ctrl)
        colors.append(CONFIG[ctrl]["color"])

    plt.figure(figsize=(6.5, 4.2))

    bp = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot(filename)


def print_summary(summary):
    cols = [
        "controller",
        "rmse_position_error_m_mean",
        "mean_abs_yaw_error_rad_mean",
        "mean_cpu_time_ms_mean",
        "total_energy_J_mean",
        "rmse_x_cpu_ms_mean",
        "stationkeeping_cost_score_mean",
        "rank_tradeoff",
    ]

    print("\n=== STATION-KEEPING PERFORMANCE-COST TRADE-OFF SUMMARY ===")
    print(
        summary[cols]
        .sort_values("rank_tradeoff")
        .round(5)
        .to_string(index=False)
    )


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    dfs = load_data()

    case_df = compute_case_metrics(dfs)
    case_df = add_composite_score(case_df)

    summary = build_summary(case_df)
    build_summary_by_sea(case_df)

    save_latex_table(summary)

    plot_pareto_cpu(summary)
    plot_pareto_wall(summary)
    plot_pareto_energy(summary)

    plot_tradeoff_score(summary)
    plot_rmse_cpu_product(summary)
    plot_rmse_energy_product(summary)

    plot_boxplot(
        case_df,
        metric="rmse_position_error_m",
        ylabel="RMSE Position Error [m]",
        filename="boxplot_stationkeeping_rmse_position",
    )

    plot_boxplot(
        case_df,
        metric="mean_abs_yaw_error_rad",
        ylabel="Mean Absolute Yaw Error [rad]",
        filename="boxplot_stationkeeping_yaw_error",
    )

    plot_boxplot(
        case_df,
        metric="rmse_x_cpu_ms",
        ylabel=r"RMSE $\times$ CPU Time [m ms]",
        filename="boxplot_stationkeeping_rmse_cpu_product",
    )

    plot_boxplot(
        case_df,
        metric="rmse_x_energy_J",
        ylabel=r"RMSE $\times$ Energy [m J]",
        filename="boxplot_stationkeeping_rmse_energy_product",
    )

    plot_boxplot(
        case_df,
        metric="stationkeeping_cost_score",
        ylabel="Station-Keeping Performance-Cost Score [-]",
        filename="boxplot_stationkeeping_cost_score",
    )

    print_summary(summary)

    print("[OK] Station-keeping performance-cost analysis finished.")


if __name__ == "__main__":
    main()