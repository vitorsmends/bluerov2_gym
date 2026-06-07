import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = "results/path_tracking"
OUTPUT_DIR = "results/plots_performance_cost_tradeoff"
TABLE_DIR = "results/tables_performance_cost_tradeoff"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


CONFIG = {
    "PID": {"file": "pid.csv"},
    "SMC": {"file": "smc.csv"},
    # "MPC": {"file": "mpc.csv"},
    "NMPC": {"file": "nmpc.csv"},
    "PPO": {"file": "ppo.csv"},
}


def save_plot(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {name}")


def load_data():
    rows = []

    for controller, cfg in CONFIG.items():
        path = os.path.join(INPUT_DIR, cfg["file"])

        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        required = {
            "tracking_error_m",
            "controller_wall_time_s",
            "controller_cpu_time_s",
            "controller_frequency_hz",
        }

        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] {controller} skipped. Missing columns: {sorted(missing)}")
            continue

        metric_cols = [
            "tracking_error_m",
            "controller_wall_time_s",
            "controller_cpu_time_s",
            "controller_frequency_hz",
        ]

        df[metric_cols] = df[metric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=metric_cols).copy()

        if df.empty:
            print(f"[WARN] {controller} skipped. No valid metric rows.")
            continue

        error = df["tracking_error_m"].to_numpy(dtype=float)
        wall = df["controller_wall_time_s"].to_numpy(dtype=float)
        cpu = df["controller_cpu_time_s"].to_numpy(dtype=float)

        rmse = float(np.sqrt(np.mean(error**2)))
        mean_error = float(np.mean(error))
        p95_error = float(np.percentile(error, 95))
        max_error = float(np.max(error))

        mean_wall_ms = float(np.mean(wall) * 1e3)
        p95_wall_ms = float(np.percentile(wall, 95) * 1e3)
        total_wall_ms = float(np.sum(wall) * 1e3)

        mean_cpu_ms = float(np.mean(cpu) * 1e3)
        p95_cpu_ms = float(np.percentile(cpu, 95) * 1e3)
        total_cpu_ms = float(np.sum(cpu) * 1e3)

        # Trade-off metrics
        rmse_cpu_product = rmse * mean_cpu_ms
        rmse_wall_product = rmse * mean_wall_ms

        performance_per_cpu_cost = 1.0 / rmse_cpu_product if rmse_cpu_product > 0 else np.nan
        performance_per_wall_cost = 1.0 / rmse_wall_product if rmse_wall_product > 0 else np.nan

        rows.append({
            "controller": controller,
            "n_steps": int(len(df)),

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

            "performance_per_cpu_cost": performance_per_cpu_cost,
            "performance_per_wall_cost": performance_per_wall_cost,
        })

    if not rows:
        raise RuntimeError("No valid controller data found.")

    return pd.DataFrame(rows)


def normalize_minmax(series, lower_is_better=True):
    values = series.to_numpy(dtype=float)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        return np.ones_like(values)

    if lower_is_better:
        return (vmax - values) / (vmax - vmin)

    return (values - vmin) / (vmax - vmin)


def add_composite_score(df):
    df = df.copy()

    df["score_tracking"] = normalize_minmax(df["rmse_error_m"], lower_is_better=True)
    df["score_cpu"] = normalize_minmax(df["mean_cpu_time_ms"], lower_is_better=True)
    df["score_wall"] = normalize_minmax(df["mean_wall_time_ms"], lower_is_better=True)

    # Weighted score:
    # 60% tracking quality, 20% CPU cost, 20% wall-time cost.
    df["performance_cost_score"] = (
        0.60 * df["score_tracking"]
        + 0.20 * df["score_cpu"]
        + 0.20 * df["score_wall"]
    )

    df["rank_rmse"] = df["rmse_error_m"].rank(method="min", ascending=True).astype("Int64")
    df["rank_cpu"] = df["mean_cpu_time_ms"].rank(method="min", ascending=True).astype("Int64")
    df["rank_tradeoff"] = df["performance_cost_score"].rank(method="min", ascending=False).astype("Int64")

    return df


def save_tables(df):
    table_path = os.path.join(TABLE_DIR, "performance_cost_tradeoff_summary.csv")
    df.to_csv(table_path, index=False)
    print(f"[OK] Saved table: {table_path}")

    latex_cols = [
        "controller",
        "rmse_error_m",
        "mean_cpu_time_ms",
        "mean_wall_time_ms",
        "rmse_x_cpu_ms",
        "performance_cost_score",
        "rank_tradeoff",
    ]

    latex_df = df[latex_cols].copy().round(4)

    print("\n=== LATEX TABLE ===\n")
    print(latex_df.to_latex(index=False))


def plot_pareto_cpu(df):
    plt.figure(figsize=(6.5, 4.2))

    for _, row in df.iterrows():
        plt.scatter(row["mean_cpu_time_ms"], row["rmse_error_m"], s=70)
        plt.text(
            row["mean_cpu_time_ms"] * 1.04,
            row["rmse_error_m"],
            row["controller"],
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Mean CPU Time [ms/step]")
    plt.ylabel("RMSE Tracking Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("pareto_rmse_vs_cpu_time")


def plot_pareto_wall(df):
    plt.figure(figsize=(6.5, 4.2))

    for _, row in df.iterrows():
        plt.scatter(row["mean_wall_time_ms"], row["rmse_error_m"], s=70)
        plt.text(
            row["mean_wall_time_ms"] * 1.04,
            row["rmse_error_m"],
            row["controller"],
            fontsize=10,
            va="center",
        )

    plt.xscale("log")
    plt.xlabel("Mean Wall Time [ms/step]")
    plt.ylabel("RMSE Tracking Error [m]")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("pareto_rmse_vs_wall_time")


def plot_tradeoff_score(df):
    ordered = df.sort_values("performance_cost_score", ascending=False)

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(ordered["controller"], ordered["performance_cost_score"])

    plt.xlabel("Controller")
    plt.ylabel("Performance-Cost Score [-]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("performance_cost_score")


def plot_rmse_cpu_product(df):
    ordered = df.sort_values("rmse_x_cpu_ms", ascending=True)

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(ordered["controller"], ordered["rmse_x_cpu_ms"])

    plt.xlabel("Controller")
    plt.ylabel(r"RMSE $\times$ CPU Time [m ms]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("rmse_cpu_product")


def print_summary(df):
    cols = [
        "controller",
        "rmse_error_m",
        "mean_cpu_time_ms",
        "mean_wall_time_ms",
        "rmse_x_cpu_ms",
        "performance_per_cpu_cost",
        "performance_cost_score",
        "rank_tradeoff",
    ]

    print("\n=== PERFORMANCE-COST TRADE-OFF SUMMARY ===")
    print(df[cols].sort_values("rank_tradeoff").round(5).to_string(index=False))


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    df = load_data()
    df = add_composite_score(df)

    save_tables(df)

    plot_pareto_cpu(df)
    plot_pareto_wall(df)
    plot_tradeoff_score(df)
    plot_rmse_cpu_product(df)

    print_summary(df)

    print("[OK] Performance-cost trade-off analysis finished.")


if __name__ == "__main__":
    main()
