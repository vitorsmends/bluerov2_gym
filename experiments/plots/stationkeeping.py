import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml
from pathlib import Path

with open("experiments/ocean_environment.yaml", "r") as f:
    config = yaml.safe_load(f)
    config_default = config.get("default_scenario")

INPUT_DIR = Path(f"results-{config_default}/stationkeeping")
OUTPUT_DIR = Path(f"results-{config_default}/plots_stationkeeping")
TABLE_DIR = Path(f"results-{config_default}/tables_stationkeeping")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "PID":  {"file": "pid_stationkeeping.csv",  "color": "#4E79A7", "ls": "-"},
    "SMC":  {"file": "smc_stationkeeping.csv",  "color": "#B07AA1", "ls": "-"},
    "NMPC": {"file": "nmpc_stationkeeping.csv", "color": "#E15759", "ls": "-"},
    "PPO":  {"file": "ppo_stationkeeping.csv",  "color": "#59A14F", "ls": "-"},
}


REQUIRED_COLUMNS = {
    "controller",
    "sea_scenario",
    "target_id",
    "repetition_id",
    "time",

    "x", "y", "z",
    "roll", "pitch", "yaw",

    "x_ref", "y_ref", "z_ref",
    "yaw_ref",

    "position_error_m",
    "tracking_error_m",
    "velocity_error",
    "yaw_error",
    "reward",

    "T1", "T2", "T3", "T4", "T5", "T6",

    "control_effort",
    "control_effort_normalized",

    "total_power_W",
    "step_energy_J",
    "cumulative_energy_J",

    "controller_wall_time_s",
    "controller_cpu_time_s",
    "controller_frequency_hz",
}


def save_plot(name):
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    png_path = OUTPUT_DIR / f"{name}.png"

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


def save_table(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[OK] Saved table: {path}")


def load_data():
    dfs = []

    for ctrl, cfg in CONFIG.items():
        path = INPUT_DIR / cfg["file"]

        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"[WARN] {ctrl} skipped. Missing columns: {sorted(missing)}")
            continue

        df = df.replace([np.inf, -np.inf], np.nan)
        df["controller"] = ctrl

        df["controller_wall_time_ms"] = df["controller_wall_time_s"] * 1e3
        df["controller_cpu_time_ms"] = df["controller_cpu_time_s"] * 1e3

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid station-keeping CSV files found.")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(
        ["controller", "sea_scenario", "target_id", "repetition_id", "time"]
    ).reset_index(drop=True)

    return data


def aggregate_time_series(df, value_col, group_cols=None):
    if group_cols is None:
        group_cols = ["repetition_id", "target_id"]

    grouped = (
        df.groupby(group_cols + ["time"])[value_col]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index="time",
        columns=group_cols,
        values=value_col,
        aggfunc="mean",
    )

    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1).fillna(0.0)

    return mean.index.to_numpy(), mean.to_numpy(), std.to_numpy()


def compute_case_metrics(data):
    rows = []

    group_cols = [
        "controller",
        "sea_scenario",
        "target_id",
        "repetition_id",
    ]

    for keys, g in data.groupby(group_cols):
        controller, sea_scenario, target_id, repetition_id = keys

        g = g.sort_values("time")

        pos_error = g["position_error_m"].to_numpy(dtype=float)
        track_error = g["tracking_error_m"].to_numpy(dtype=float)
        yaw_error = np.abs(g["yaw_error"].to_numpy(dtype=float))
        effort = g["control_effort"].to_numpy(dtype=float)
        norm_effort = g["control_effort_normalized"].to_numpy(dtype=float)
        power = g["total_power_W"].to_numpy(dtype=float)
        energy = g["step_energy_J"].to_numpy(dtype=float)
        wall_ms = g["controller_wall_time_ms"].to_numpy(dtype=float)
        cpu_ms = g["controller_cpu_time_ms"].to_numpy(dtype=float)

        rows.append({
            "controller": controller,
            "sea_scenario": sea_scenario,
            "target_id": target_id,
            "repetition_id": repetition_id,
            "n_steps": int(len(g)),
            "duration_s": float(g["time"].max() - g["time"].min()),

            "mean_position_error_m": float(np.nanmean(pos_error)),
            "rmse_position_error_m": float(np.sqrt(np.nanmean(pos_error ** 2))),
            "p95_position_error_m": float(np.nanpercentile(pos_error, 95)),
            "max_position_error_m": float(np.nanmax(pos_error)),
            "final_position_error_m": float(pos_error[-1]),

            "mean_tracking_error_m": float(np.nanmean(track_error)),
            "rmse_tracking_error_m": float(np.sqrt(np.nanmean(track_error ** 2))),

            "mean_abs_yaw_error_rad": float(np.nanmean(yaw_error)),
            "p95_abs_yaw_error_rad": float(np.nanpercentile(yaw_error, 95)),
            "max_abs_yaw_error_rad": float(np.nanmax(yaw_error)),

            "mean_control_effort": float(np.nanmean(effort)),
            "total_control_effort": float(np.nansum(effort)),
            "mean_normalized_effort": float(np.nanmean(norm_effort)),

            "mean_power_W": float(np.nanmean(power)),
            "p95_power_W": float(np.nanpercentile(power, 95)),
            "total_energy_J": float(np.nansum(energy)),

            "mean_wall_time_ms": float(np.nanmean(wall_ms)),
            "p95_wall_time_ms": float(np.nanpercentile(wall_ms, 95)),
            "total_wall_time_ms": float(np.nansum(wall_ms)),

            "mean_cpu_time_ms": float(np.nanmean(cpu_ms)),
            "p95_cpu_time_ms": float(np.nanpercentile(cpu_ms, 95)),
            "total_cpu_time_ms": float(np.nansum(cpu_ms)),
        })

    case_df = pd.DataFrame(rows)
    save_table(case_df, "stationkeeping_metrics_per_case")

    return case_df


def build_summary(case_df):
    metric_cols = [
        "mean_position_error_m",
        "rmse_position_error_m",
        "p95_position_error_m",
        "max_position_error_m",
        "final_position_error_m",
        "mean_abs_yaw_error_rad",
        "p95_abs_yaw_error_rad",
        "mean_control_effort",
        "total_control_effort",
        "mean_power_W",
        "total_energy_J",
        "mean_wall_time_ms",
        "mean_cpu_time_ms",
    ]

    rows = []

    for controller, g in case_df.groupby("controller"):
        row = {
            "controller": controller,
            "n_cases": int(len(g)),
            "n_targets": int(g["target_id"].nunique()),
            "n_sea_scenarios": int(g["sea_scenario"].nunique()),
            "n_repetitions": int(g["repetition_id"].nunique()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary["rank_rmse_position"] = (
        summary["rmse_position_error_m_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_total_energy"] = (
        summary["total_energy_J_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_cpu_time"] = (
        summary["mean_cpu_time_ms_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    save_table(summary, "stationkeeping_summary_by_controller")

    print("\n=== STATION-KEEPING SUMMARY BY CONTROLLER ===")
    cols = [
        "controller",
        "rmse_position_error_m_mean",
        "rmse_position_error_m_std",
        "mean_abs_yaw_error_rad_mean",
        "total_energy_J_mean",
        "mean_cpu_time_ms_mean",
        "rank_rmse_position",
    ]
    print(summary[cols].round(4).to_string(index=False))

    return summary


def build_summary_by_sea(case_df):
    rows = []

    metric_cols = [
        "rmse_position_error_m",
        "mean_abs_yaw_error_rad",
        "mean_control_effort",
        "mean_power_W",
        "total_energy_J",
        "mean_cpu_time_ms",
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

    df = pd.DataFrame(rows)
    save_table(df, "stationkeeping_summary_by_controller_and_sea")

    return df


def plot_error_time_by_sea(data):
    for sea, sea_df in data.groupby("sea_scenario"):
        plt.figure(figsize=(6.5, 4.2))

        for ctrl in CONFIG:
            g = sea_df[sea_df["controller"] == ctrl]

            if g.empty:
                continue

            t, mean_v, std_v = aggregate_time_series(g, "position_error_m")
            color = CONFIG[ctrl]["color"]

            plt.plot(
                t,
                mean_v,
                color=color,
                linestyle=CONFIG[ctrl]["ls"],
                linewidth=1.8,
                label=ctrl,
            )

            plt.fill_between(
                t,
                np.maximum(mean_v - std_v, 0.0),
                mean_v + std_v,
                color=color,
                alpha=0.18,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        # plt.title(f"Station Keeping Error - {sea}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        save_plot(f"stationkeeping_position_error_{sea}")


def plot_yaw_error_time_by_sea(data):
    data = data.copy()
    data["abs_yaw_error"] = np.abs(data["yaw_error"])

    for sea, sea_df in data.groupby("sea_scenario"):
        plt.figure(figsize=(6.5, 4.2))

        for ctrl in CONFIG:
            g = sea_df[sea_df["controller"] == ctrl]

            if g.empty:
                continue

            t, mean_v, std_v = aggregate_time_series(g, "abs_yaw_error")
            color = CONFIG[ctrl]["color"]

            plt.plot(
                t,
                mean_v,
                color=color,
                linestyle=CONFIG[ctrl]["ls"],
                linewidth=1.8,
                label=ctrl,
            )

            plt.fill_between(
                t,
                np.maximum(mean_v - std_v, 0.0),
                mean_v + std_v,
                color=color,
                alpha=0.18,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Absolute Yaw Error [rad]")
        # plt.title(f"Yaw Holding Error - {sea}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        save_plot(f"stationkeeping_yaw_error_{sea}")


def plot_effort_time_by_sea(data):
    for sea, sea_df in data.groupby("sea_scenario"):
        plt.figure(figsize=(6.5, 4.2))

        for ctrl in CONFIG:
            g = sea_df[sea_df["controller"] == ctrl]

            if g.empty:
                continue

            t, mean_v, std_v = aggregate_time_series(g, "control_effort")
            color = CONFIG[ctrl]["color"]

            plt.plot(
                t,
                mean_v,
                color=color,
                linestyle=CONFIG[ctrl]["ls"],
                linewidth=1.8,
                label=ctrl,
            )

            plt.fill_between(
                t,
                np.maximum(mean_v - std_v, 0.0),
                mean_v + std_v,
                color=color,
                alpha=0.18,
            )

        plt.xlabel("Time [s]")
        plt.ylabel(r"Control Effort $\sum_i T_i^2$ [N$^2$]")
        # plt.title(f"Control Effort - {sea}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        save_plot(f"stationkeeping_control_effort_{sea}")


def plot_power_time_by_sea(data):
    for sea, sea_df in data.groupby("sea_scenario"):
        plt.figure(figsize=(6.5, 4.2))

        for ctrl in CONFIG:
            g = sea_df[sea_df["controller"] == ctrl]

            if g.empty:
                continue

            t, mean_v, std_v = aggregate_time_series(g, "total_power_W")
            color = CONFIG[ctrl]["color"]

            plt.plot(
                t,
                mean_v,
                color=color,
                linestyle=CONFIG[ctrl]["ls"],
                linewidth=1.8,
                label=ctrl,
            )

            plt.fill_between(
                t,
                np.maximum(mean_v - std_v, 0.0),
                mean_v + std_v,
                color=color,
                alpha=0.18,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Total Power [W]")
        # plt.title(f"Estimated Thruster Power - {sea}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        save_plot(f"stationkeeping_power_{sea}")


def plot_xy_targets(data):
    plt.figure(figsize=(6.5, 5.0))

    target_df = (
        data[["target_id", "x_ref", "y_ref", "z_ref", "yaw_ref"]]
        .drop_duplicates()
        .sort_values("target_id")
    )

    plt.scatter(
        target_df["x_ref"],
        target_df["y_ref"],
        s=80,
        marker="x",
        color="black",
        label="Targets",
    )

    for _, row in target_df.iterrows():
        plt.text(
            row["x_ref"] + 0.05,
            row["y_ref"] + 0.05,
            str(int(row["target_id"])),
            fontsize=9,
        )

        yaw = row["yaw_ref"]
        dx = 0.25 * np.cos(yaw)
        dy = 0.25 * np.sin(yaw)

        plt.arrow(
            row["x_ref"],
            row["y_ref"],
            dx,
            dy,
            head_width=0.06,
            length_includes_head=True,
            color="black",
            alpha=0.7,
        )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    # plt.title("Station-Keeping Target Positions and Desired Headings")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("stationkeeping_targets_xy")


def plot_boxplot(case_df, metric, ylabel, filename, by="controller"):
    plt.figure(figsize=(6.5, 4.2))

    if by == "controller":
        labels = []
        data = []
        colors = []

        for ctrl in CONFIG:
            g = case_df[case_df["controller"] == ctrl]

            if g.empty:
                continue

            labels.append(ctrl)
            data.append(g[metric].dropna().to_numpy(dtype=float))
            colors.append(CONFIG[ctrl]["color"])

        bp = plt.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

    elif by == "sea":
        labels = []
        data = []

        for sea in sorted(case_df["sea_scenario"].unique()):
            g = case_df[case_df["sea_scenario"] == sea]
            labels.append(sea)
            data.append(g[metric].dropna().to_numpy(dtype=float))

        plt.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
        )

    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot(filename)


def plot_bar_summary(summary, metric_mean, metric_std, ylabel, filename):
    ordered = summary.sort_values(metric_mean)
    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))

    plt.bar(
        ordered["controller"],
        ordered[metric_mean],
        yerr=ordered[metric_std],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot(filename)


def plot_heatmap_controller_target(case_df, sea_scenario=None):
    df = case_df.copy()

    if sea_scenario is not None:
        df = df[df["sea_scenario"] == sea_scenario]

    metric = "rmse_position_error_m"

    pivot = df.pivot_table(
        index="controller",
        columns="target_id",
        values=metric,
        aggfunc="mean",
    )

    if pivot.empty:
        return

    plt.figure(figsize=(8.0, 3.8))

    im = plt.imshow(pivot.values, aspect="auto")

    plt.colorbar(im, label="RMSE Position Error [m]")

    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=[str(c) for c in pivot.columns],
    )

    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index,
    )

    title = "RMSE by Controller and Target"
    filename = "heatmap_rmse_controller_target"

    if sea_scenario is not None:
        title += f" - {sea_scenario}"
        filename += f"_{sea_scenario}"

    # plt.title(title)
    plt.xlabel("Target ID")
    plt.ylabel("Controller")
    plt.tight_layout()
    save_plot(filename)


def plot_3d_targets(data):
    target_df = (
        data[["target_id", "x_ref", "y_ref", "z_ref"]]
        .drop_duplicates()
        .sort_values("target_id")
    )

    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        target_df["x_ref"],
        target_df["y_ref"],
        target_df["z_ref"],
        s=60,
        marker="x",
        color="black",
    )

    for _, row in target_df.iterrows():
        ax.text(
            row["x_ref"],
            row["y_ref"],
            row["z_ref"],
            str(int(row["target_id"])),
            fontsize=8,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    # ax.set_title("Station-Keeping Target Positions")
    ax.invert_zaxis()
    plt.tight_layout()
    save_plot("stationkeeping_targets_3d")


def print_latex_table(summary):
    cols = [
        "controller",
        "rmse_position_error_m_mean",
        "rmse_position_error_m_std",
        "mean_abs_yaw_error_rad_mean",
        "total_energy_J_mean",
        "mean_cpu_time_ms_mean",
        "rank_rmse_position",
    ]

    latex_df = summary[cols].copy().round(4)

    print("\n=== LATEX TABLE ===\n")
    print(latex_df.to_latex(index=False))


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    data = load_data()

    case_df = compute_case_metrics(data)
    summary = build_summary(case_df)
    sea_summary = build_summary_by_sea(case_df)

    print_latex_table(summary)

    plot_xy_targets(data)
    plot_3d_targets(data)

    plot_error_time_by_sea(data)
    plot_yaw_error_time_by_sea(data)
    plot_effort_time_by_sea(data)
    plot_power_time_by_sea(data)

    plot_boxplot(
        case_df,
        metric="rmse_position_error_m",
        ylabel="RMSE Position Error [m]",
        filename="boxplot_rmse_position_error_by_controller",
        by="controller",
    )

    plot_boxplot(
        case_df,
        metric="mean_abs_yaw_error_rad",
        ylabel="Mean Absolute Yaw Error [rad]",
        filename="boxplot_yaw_error_by_controller",
        by="controller",
    )

    plot_boxplot(
        case_df,
        metric="total_energy_J",
        ylabel="Total Energy [J]",
        filename="boxplot_total_energy_by_controller",
        by="controller",
    )

    plot_boxplot(
        case_df,
        metric="mean_cpu_time_ms",
        ylabel="Mean CPU Time [ms]",
        filename="boxplot_cpu_time_by_controller",
        by="controller",
    )

    plot_boxplot(
        case_df,
        metric="rmse_position_error_m",
        ylabel="RMSE Position Error [m]",
        filename="boxplot_rmse_position_error_by_sea",
        by="sea",
    )

    plot_bar_summary(
        summary,
        metric_mean="rmse_position_error_m_mean",
        metric_std="rmse_position_error_m_std",
        ylabel="RMSE Position Error [m]",
        filename="bar_rmse_position_error",
    )

    plot_bar_summary(
        summary,
        metric_mean="total_energy_J_mean",
        metric_std="total_energy_J_std",
        ylabel="Total Energy [J]",
        filename="bar_total_energy",
    )

    plot_bar_summary(
        summary,
        metric_mean="mean_cpu_time_ms_mean",
        metric_std="mean_cpu_time_ms_std",
        ylabel="Mean CPU Time [ms]",
        filename="bar_mean_cpu_time",
    )

    plot_heatmap_controller_target(case_df)

    for sea in sorted(case_df["sea_scenario"].unique()):
        plot_heatmap_controller_target(case_df, sea_scenario=sea)

    print("\n[OK] Station-keeping plots generated.")
    print(f"[OK] Tables saved in: {TABLE_DIR}")
    print(f"[OK] Figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()