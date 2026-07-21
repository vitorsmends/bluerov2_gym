import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml

with open("experiments/ocean_environment.yaml", "r") as f:
    config = yaml.safe_load(f)
    config_default = config.get("default_scenario")

INPUT_DIR = f"results-{config_default}/experiments"
OUTPUT_DIR = f"results-{config_default}/plots_computational_cost"
TABLE_DIR = f"results-{config_default}/tables_computational_cost"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


CONFIG = {
    "PID":  {"file": "pid.csv",  "color": "#4E79A7"},
    "SMC":  {"file": "smc.csv",  "color": "#B07AA1"},
    "NMPC": {"file": "nmpc.csv", "color": "#E15759"},
    "PPO":  {"file": "ppo.csv",  "color": "#59A14F"},
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
        "tracking_error_m",
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

        if "repetition_id" not in df.columns:
            df["repetition_id"] = 0

        metric_cols = list(required)

        df[metric_cols] = df[metric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=metric_cols).copy()

        if df.empty:
            print(f"[WARN] {controller} skipped. No valid metric rows.")
            continue

        df["controller"] = controller
        df = df.sort_values(["repetition_id", "time"]).reset_index(drop=True)

        dfs[controller] = df

    if not dfs:
        raise RuntimeError("No valid controller data found.")

    return dfs


def compute_repetition_metrics(dfs):
    rows = []

    for controller, df in dfs.items():
        for rep, g in df.groupby("repetition_id"):
            error = g["tracking_error_m"].to_numpy(dtype=float)
            wall = g["controller_wall_time_s"].to_numpy(dtype=float)
            cpu = g["controller_cpu_time_s"].to_numpy(dtype=float)

            rmse = float(np.sqrt(np.nanmean(error**2)))
            mean_error = float(np.nanmean(error))
            p95_error = float(np.nanpercentile(error, 95))
            max_error = float(np.nanmax(error))

            mean_wall_ms = float(np.nanmean(wall) * 1e3)
            p95_wall_ms = float(np.nanpercentile(wall, 95) * 1e3)
            total_wall_ms = float(np.nansum(wall) * 1e3)

            mean_cpu_ms = float(np.nanmean(cpu) * 1e3)
            p95_cpu_ms = float(np.nanpercentile(cpu, 95) * 1e3)
            total_cpu_ms = float(np.nansum(cpu) * 1e3)

            rmse_cpu_product = rmse * mean_cpu_ms
            rmse_wall_product = rmse * mean_wall_ms

            rows.append({
                "controller": controller,
                "repetition_id": rep,
                "n_steps": int(len(g)),

                "rmse_error_m": rmse,
                "mean_error_m": mean_error,
                "p95_error_m": p95_error,
                "max_error_m": max_error,

                "mean_wall_time_ms": mean_wall_ms,
                "p95_wall_time_ms": p95_wall_ms,
                "total_wall_time_ms": total_wall_ms,

                "mean_cpu_time_ms": mean_cpu_ms,
                "p95_cpu_time_ms": p95_cpu_ms,
                "total_cpu_time_ms": total_cpu_ms,

                "rmse_x_cpu_ms": rmse_cpu_product,
                "rmse_x_wall_ms": rmse_wall_product,

                "performance_per_cpu_cost": (
                    1.0 / rmse_cpu_product if rmse_cpu_product > 0 else np.nan
                ),
                "performance_per_wall_cost": (
                    1.0 / rmse_wall_product if rmse_wall_product > 0 else np.nan
                ),
            })

    rep_df = pd.DataFrame(rows)
    save_table(rep_df, "performance_cost_tradeoff_per_repetition")

    return rep_df


def normalize_minmax(series, lower_is_better=True):
    values = series.to_numpy(dtype=float)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        return np.ones_like(values)

    if lower_is_better:
        return (vmax - values) / (vmax - vmin)

    return (values - vmin) / (vmax - vmin)


def add_composite_score(rep_df):
    rep_df = rep_df.copy()

    rep_df["score_tracking"] = normalize_minmax(
        rep_df["rmse_error_m"],
        lower_is_better=True,
    )

    rep_df["score_cpu"] = normalize_minmax(
        rep_df["mean_cpu_time_ms"],
        lower_is_better=True,
    )

    rep_df["score_wall"] = normalize_minmax(
        rep_df["mean_wall_time_ms"],
        lower_is_better=True,
    )

    rep_df["performance_cost_score"] = (
        0.60 * rep_df["score_tracking"]
        + 0.20 * rep_df["score_cpu"]
        + 0.20 * rep_df["score_wall"]
    )

    return rep_df


def build_summary(rep_df):
    metric_cols = [
        "rmse_error_m",
        "mean_error_m",
        "p95_error_m",
        "max_error_m",
        "mean_wall_time_ms",
        "p95_wall_time_ms",
        "total_wall_time_ms",
        "mean_cpu_time_ms",
        "p95_cpu_time_ms",
        "total_cpu_time_ms",
        "rmse_x_cpu_ms",
        "rmse_x_wall_ms",
        "performance_per_cpu_cost",
        "performance_per_wall_cost",
        "performance_cost_score",
    ]

    rows = []

    for controller, g in rep_df.groupby("controller"):
        row = {
            "controller": controller,
            "n_repetitions": int(g["repetition_id"].nunique()),
            "mean_steps": float(g["n_steps"].mean()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary["rank_rmse"] = (
        summary["rmse_error_m_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_cpu"] = (
        summary["mean_cpu_time_ms_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_tradeoff"] = (
        summary["performance_cost_score_mean"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    save_table(summary, "performance_cost_tradeoff_summary")

    return summary


def save_latex_table(summary):
    cols = [
        "controller",
        "rmse_error_m_mean",
        "rmse_error_m_std",
        "mean_cpu_time_ms_mean",
        "mean_cpu_time_ms_std",
        "mean_wall_time_ms_mean",
        "rmse_x_cpu_ms_mean",
        "performance_cost_score_mean",
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
            row["rmse_error_m_mean"],
            xerr=row["mean_cpu_time_ms_std"],
            yerr=row["rmse_error_m_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=ctrl,
        )

        plt.text(
            row["mean_cpu_time_ms_mean"] * 1.04,
            row["rmse_error_m_mean"],
            ctrl,
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Mean CPU Time [ms/step]")
    plt.ylabel("RMSE Tracking Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("pareto_rmse_vs_cpu_time_mean_std")


def plot_pareto_wall(summary):
    plt.figure(figsize=(6.5, 4.2))

    for _, row in summary.iterrows():
        ctrl = row["controller"]
        color = CONFIG[ctrl]["color"]

        plt.errorbar(
            row["mean_wall_time_ms_mean"],
            row["rmse_error_m_mean"],
            xerr=row["mean_wall_time_ms_std"],
            yerr=row["rmse_error_m_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=color,
            label=ctrl,
        )

        plt.text(
            row["mean_wall_time_ms_mean"] * 1.04,
            row["rmse_error_m_mean"],
            ctrl,
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Mean Wall Time [ms/step]")
    plt.ylabel("RMSE Tracking Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("pareto_rmse_vs_wall_time_mean_std")


def plot_tradeoff_score(summary):
    ordered = summary.sort_values("performance_cost_score_mean", ascending=False)
    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))

    plt.bar(
        ordered["controller"],
        ordered["performance_cost_score_mean"],
        yerr=ordered["performance_cost_score_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel("Performance-Cost Score [-]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("performance_cost_score_mean_std")


def plot_rmse_cpu_product(summary):
    ordered = summary.sort_values("rmse_x_cpu_ms_mean", ascending=True)
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
    save_plot("rmse_cpu_product_mean_std")


def plot_boxplot(rep_df, metric, ylabel, filename):
    data = []
    labels = []
    colors = []

    for ctrl in CONFIG:
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


def print_summary(summary):
    cols = [
        "controller",
        "rmse_error_m_mean",
        "mean_cpu_time_ms_mean",
        "mean_wall_time_ms_mean",
        "rmse_x_cpu_ms_mean",
        "performance_cost_score_mean",
        "rank_tradeoff",
    ]

    print("\n=== PERFORMANCE-COST TRADE-OFF SUMMARY ===")
    print(summary[cols].sort_values("rank_tradeoff").round(5).to_string(index=False))


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    dfs = load_data()

    rep_df = compute_repetition_metrics(dfs)
    rep_df = add_composite_score(rep_df)

    summary = build_summary(rep_df)

    save_latex_table(summary)

    plot_pareto_cpu(summary)
    plot_pareto_wall(summary)
    plot_tradeoff_score(summary)
    plot_rmse_cpu_product(summary)

    plot_boxplot(
        rep_df,
        metric="rmse_x_cpu_ms",
        ylabel=r"RMSE $\times$ CPU Time [m ms]",
        filename="boxplot_rmse_cpu_product",
    )

    plot_boxplot(
        rep_df,
        metric="performance_cost_score",
        ylabel="Performance-Cost Score [-]",
        filename="boxplot_performance_cost_score",
    )

    plot_boxplot(
        rep_df,
        metric="rmse_error_m",
        ylabel="RMSE Tracking Error [m]",
        filename="boxplot_tradeoff_rmse",
    )

    print_summary(summary)

    print("[OK] Performance-cost trade-off analysis finished.")


if __name__ == "__main__":
    main()