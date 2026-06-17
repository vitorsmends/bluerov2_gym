import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml
from pathlib import Path

with open("path_tracking_experiments/jonswap_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    config_default = config.get("default_scenario")

INPUT_DIR = f"results-{config_default}/path_tracking"
OUTPUT_DIR = f"results-{config_default}/plots_path_tracking"
TABLE_DIR = f"results-{config_default}/tables_path_tracking"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


CONFIG = {
    "PID":  {"file": "pid.csv",  "color": "#4E79A7", "ls": "-"},
    "SMC":  {"file": "smc.csv",  "color": "#B07AA1", "ls": "-"},
    "NMPC": {"file": "nmpc.csv", "color": "#E15759", "ls": "-"},
    "PPO":  {"file": "ppo.csv",  "color": "#59A14F", "ls": "-"},
}


def save_plot(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {name}")


def save_table(df, name):
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Saved table: {path}")


def load_results():
    dfs = {}

    for ctrl, cfg in CONFIG.items():
        path = os.path.join(INPUT_DIR, cfg["file"])

        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        required = {"time", "x", "y", "z", "x_ref", "y_ref", "z_ref"}
        missing = required - set(df.columns)

        if missing:
            print(f"[WARN] {ctrl} skipped. Missing columns: {missing}")
            continue

        if "repetition_id" not in df.columns:
            df["repetition_id"] = 0

        df = df.sort_values(["repetition_id", "time"]).reset_index(drop=True)

        if "tracking_error_m" not in df.columns:
            df["tracking_error_m"] = np.linalg.norm(
                df[["x", "y", "z"]].values
                - df[["x_ref", "y_ref", "z_ref"]].values,
                axis=1,
            )

        thruster_cols = [f"T{i}" for i in range(1, 7)]

        if all(c in df.columns for c in thruster_cols):
            u = df[thruster_cols].values
            df["control_effort"] = np.sum(u**2, axis=1)
            df["control_effort_normalized"] = np.mean((u / 40.0) ** 2, axis=1)

            cumulative = []
            for _, g in df.groupby("repetition_id"):
                g = g.sort_values("time")
                dt = np.mean(np.diff(g["time"].values)) if len(g) > 1 else 0.0
                cumulative.extend(np.cumsum(g["control_effort"].values) * dt)

            df["cumulative_effort"] = cumulative
        else:
            df["control_effort"] = np.nan
            df["control_effort_normalized"] = np.nan
            df["cumulative_effort"] = np.nan

        dfs[ctrl] = df

    return dfs


def aggregate_time_series(df, value_col):
    grouped = (
        df.groupby(["repetition_id", "time"])[value_col]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="time",
        columns="repetition_id",
        values=value_col,
    )

    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1).fillna(0.0)

    return mean.index.to_numpy(), mean.to_numpy(), std.to_numpy()


def get_first_repetition(df):
    first_rep = sorted(df["repetition_id"].unique())[0]
    return df[df["repetition_id"] == first_rep].copy()


def compute_repetition_metrics(ctrl, df):
    rows = []

    for rep, g in df.groupby("repetition_id"):
        g = g.sort_values("time")

        err = g["tracking_error_m"].to_numpy(dtype=float)
        effort = g["control_effort"].to_numpy(dtype=float)
        norm_effort = g["control_effort_normalized"].to_numpy(dtype=float)

        rows.append({
            "controller": ctrl,
            "repetition_id": rep,
            "duration_s": float(g["time"].max() - g["time"].min()),
            "n_steps": int(len(g)),

            "mean_error_m": float(np.nanmean(err)),
            "rmse_error_m": float(np.sqrt(np.nanmean(err**2))),
            "max_error_m": float(np.nanmax(err)),
            "p95_error_m": float(np.nanpercentile(err, 95)),
            "final_error_m": float(err[-1]),

            "mean_control_effort": float(np.nanmean(effort)),
            "total_control_effort": float(np.nansum(effort)),
            "mean_normalized_effort": float(np.nanmean(norm_effort)),
        })

    return pd.DataFrame(rows)


def build_summary(dfs):
    rep_tables = []

    for ctrl, df in dfs.items():
        rep_tables.append(compute_repetition_metrics(ctrl, df))

    rep_df = pd.concat(rep_tables, ignore_index=True)
    save_table(rep_df, "summary_per_repetition_path_tracking")

    summary_rows = []

    metric_cols = [
        "mean_error_m",
        "rmse_error_m",
        "max_error_m",
        "p95_error_m",
        "final_error_m",
        "mean_control_effort",
        "total_control_effort",
        "mean_normalized_effort",
    ]

    for ctrl, g in rep_df.groupby("controller"):
        row = {
            "controller": ctrl,
            "n_repetitions": int(g["repetition_id"].nunique()),
            "mean_steps": float(g["n_steps"].mean()),
            "mean_duration_s": float(g["duration_s"].mean()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    save_table(summary, "summary_path_tracking")

    print("\n=== PATH TRACKING SUMMARY ===")
    print(summary.round(4).to_string(index=False))

    return rep_df, summary


def plot_xy(dfs):
    plt.figure(figsize=(6.5, 4.2))

    first = get_first_repetition(next(iter(dfs.values())))
    plt.plot(
        first["x_ref"],
        first["y_ref"],
        "k--",
        linewidth=2.0,
        label="Reference",
    )

    for ctrl, df in dfs.items():
        g = get_first_repetition(df)

        plt.plot(
            g["x"],
            g["y"],
            color=CONFIG[ctrl]["color"],
            linestyle=CONFIG[ctrl]["ls"],
            linewidth=1.8,
            label=ctrl,
        )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("path_tracking_xy_first_repetition")


def plot_3d(dfs):
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    first = get_first_repetition(next(iter(dfs.values())))
    ax.plot(
        first["x_ref"],
        first["y_ref"],
        first["z_ref"],
        "k--",
        linewidth=2.0,
        label="Reference",
    )

    for ctrl, df in dfs.items():
        g = get_first_repetition(df)

        ax.plot(
            g["x"],
            g["y"],
            g["z"],
            color=CONFIG[ctrl]["color"],
            linestyle=CONFIG[ctrl]["ls"],
            linewidth=1.6,
            label=ctrl,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.invert_zaxis()
    ax.legend(frameon=True)
    plt.tight_layout()
    save_plot("path_tracking_3d_first_repetition")


def plot_error(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for ctrl, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "tracking_error_m")
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
    plt.ylabel("Euclidean Tracking Error [m]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("path_tracking_error_mean_std")


def plot_control_effort(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for ctrl, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "control_effort")
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
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("control_effort_mean_std")


def plot_cumulative_effort(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for ctrl, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "cumulative_effort")
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
    plt.ylabel(r"Cumulative Effort $\int \sum_i T_i^2 dt$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("cumulative_control_effort_mean_std")


def plot_boxplot_metric(rep_df, metric, ylabel, filename):
    controllers = list(CONFIG.keys())
    data = []
    labels = []
    colors = []

    for ctrl in controllers:
        g = rep_df[rep_df["controller"] == ctrl]
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


def plot_thrusters(dfs):
    thruster_cols = [f"T{i}" for i in range(1, 7)]

    for ctrl, df in dfs.items():
        if not all(c in df.columns for c in thruster_cols):
            continue

        g = get_first_repetition(df)

        plt.figure(figsize=(6.5, 4.2))

        for col in thruster_cols:
            plt.plot(
                g["time"],
                g[col],
                linewidth=1.3,
                label=col,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Thruster Command [N]")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, ncol=3)
        plt.tight_layout()
        save_plot(f"thrusters_{ctrl.lower()}_first_repetition")


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    dfs = load_results()

    if not dfs:
        print("[ERROR] No valid CSV files found.")
        return

    rep_df, summary = build_summary(dfs)

    plot_xy(dfs)
    plot_3d(dfs)

    plot_error(dfs)
    plot_control_effort(dfs)
    plot_cumulative_effort(dfs)

    plot_boxplot_metric(
        rep_df,
        metric="rmse_error_m",
        ylabel="RMSE Tracking Error [m]",
        filename="boxplot_rmse_error",
    )

    plot_boxplot_metric(
        rep_df,
        metric="mean_error_m",
        ylabel="Mean Tracking Error [m]",
        filename="boxplot_mean_error",
    )

    plot_boxplot_metric(
        rep_df,
        metric="final_error_m",
        ylabel="Final Tracking Error [m]",
        filename="boxplot_final_error",
    )

    plot_boxplot_metric(
        rep_df,
        metric="mean_control_effort",
        ylabel=r"Mean Control Effort $\sum_i T_i^2$ [N$^2$]",
        filename="boxplot_mean_control_effort",
    )

    plot_thrusters(dfs)

    print("[OK] All repeated-experiment plots generated.")


if __name__ == "__main__":
    main()