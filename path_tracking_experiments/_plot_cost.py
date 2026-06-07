import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


INPUT_DIR = "results/path_tracking"
OUTPUT_DIR = "results/plots_computational_cost"
TABLE_DIR = "results/tables_computational_cost"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


CONFIG = {
    "PID": {"file": "pid.csv", "color": "#1f77b4", "ls": "-"},
    "SMC": {"file": "smc.csv", "color": "#9467bd", "ls": "--"},
    "NMPC": {"file": "nmpc.csv", "color": "#ff7f0e", "ls": "-"},
    "PPO": {"file": "ppo.csv", "color": "#2ca02c", "ls": "-."},
}


def save_plot(name):
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")

    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


def save_table(df, name):
    csv_path = os.path.join(TABLE_DIR, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved table: {csv_path}")


def load_data():
    required_cols = {
        "time",
        "controller_wall_time_s",
        "controller_cpu_time_s",
        "controller_frequency_hz",
    }

    dfs = {}

    for name, info in CONFIG.items():
        path = os.path.join(INPUT_DIR, info["file"])

        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        missing = required_cols - set(df.columns)
        if missing:
            print(f"[WARN] {name} skipped. Missing columns: {sorted(missing)}")
            continue

        df = df.sort_values("time").reset_index(drop=True)

        df["controller"] = name

        df["controller_wall_time_ms"] = df["controller_wall_time_s"] * 1e3
        df["controller_cpu_time_ms"] = df["controller_cpu_time_s"] * 1e3

        df["dt_s"] = df["time"].diff()
        df["dt_s"] = df["dt_s"].bfill().ffill()
        df["dt_s"] = df["dt_s"].replace(0.0, np.nan)

        df["dt_ms"] = df["dt_s"] * 1e3

        df["equivalent_rate_from_wall_hz"] = np.where(
            df["controller_wall_time_s"] > 0.0,
            1.0 / df["controller_wall_time_s"],
            np.nan,
        )

        df["wall_utilization_pct"] = (
            100.0 * df["controller_wall_time_s"] / df["dt_s"]
        )

        df["cpu_utilization_pct"] = (
            100.0 * df["controller_cpu_time_s"] / df["dt_s"]
        )

        df["wall_deadline_miss"] = df["controller_wall_time_s"] > df["dt_s"]
        df["cpu_deadline_miss"] = df["controller_cpu_time_s"] > df["dt_s"]

        dfs[name] = df

    return dfs


def compute_metrics(df):
    wall_ms = df["controller_wall_time_ms"].to_numpy(dtype=float)
    cpu_ms = df["controller_cpu_time_ms"].to_numpy(dtype=float)
    freq_hz = df["controller_frequency_hz"].to_numpy(dtype=float)
    eq_rate_hz = df["equivalent_rate_from_wall_hz"].to_numpy(dtype=float)
    wall_util = df["wall_utilization_pct"].to_numpy(dtype=float)
    cpu_util = df["cpu_utilization_pct"].to_numpy(dtype=float)

    return {
        "n_steps": int(len(df)),
        "sim_time_s": float(df["time"].max() - df["time"].min()),

        "wall_time_mean_ms": float(np.nanmean(wall_ms)),
        "wall_time_std_ms": float(np.nanstd(wall_ms)),
        "wall_time_max_ms": float(np.nanmax(wall_ms)),
        "wall_time_p95_ms": float(np.nanpercentile(wall_ms, 95)),
        "wall_time_total_ms": float(np.nansum(wall_ms)),

        "cpu_time_mean_ms": float(np.nanmean(cpu_ms)),
        "cpu_time_std_ms": float(np.nanstd(cpu_ms)),
        "cpu_time_max_ms": float(np.nanmax(cpu_ms)),
        "cpu_time_p95_ms": float(np.nanpercentile(cpu_ms, 95)),
        "cpu_time_total_ms": float(np.nansum(cpu_ms)),

        "cpu_to_wall_ratio": (
            float(np.nansum(cpu_ms) / np.nansum(wall_ms))
            if np.nansum(wall_ms) > 0.0 else np.nan
        ),

        "freq_mean_hz": float(np.nanmean(freq_hz)),
        "freq_std_hz": float(np.nanstd(freq_hz)),
        "freq_min_hz": float(np.nanmin(freq_hz)),

        "eq_rate_mean_hz": float(np.nanmean(eq_rate_hz)),
        "eq_rate_min_hz": float(np.nanmin(eq_rate_hz)),
        "eq_rate_p05_hz": float(np.nanpercentile(eq_rate_hz, 5)),

        "wall_util_mean_pct": float(np.nanmean(wall_util)),
        "wall_util_max_pct": float(np.nanmax(wall_util)),
        "cpu_util_mean_pct": float(np.nanmean(cpu_util)),
        "cpu_util_max_pct": float(np.nanmax(cpu_util)),

        "wall_deadline_miss_count": int(df["wall_deadline_miss"].sum()),
        "cpu_deadline_miss_count": int(df["cpu_deadline_miss"].sum()),
        "wall_deadline_miss_rate_pct": float(100.0 * df["wall_deadline_miss"].mean()),
        "cpu_deadline_miss_rate_pct": float(100.0 * df["cpu_deadline_miss"].mean()),
    }


def build_summary_table(dfs):
    rows = []

    for name, df in dfs.items():
        metrics = compute_metrics(df)
        metrics["controller"] = name
        rows.append(metrics)

    summary = pd.DataFrame(rows)

    summary["rank_lowest_cpu_total"] = (
        summary["cpu_time_total_ms"].rank(method="min", ascending=True).astype("Int64")
    )

    summary["rank_lowest_wall_total"] = (
        summary["wall_time_total_ms"].rank(method="min", ascending=True).astype("Int64")
    )

    summary["rank_fastest_response"] = (
        summary["wall_time_mean_ms"].rank(method="min", ascending=True).astype("Int64")
    )

    summary["rank_best_eq_rate"] = (
        summary["eq_rate_mean_hz"].rank(method="min", ascending=False).astype("Int64")
    )

    summary["rank_lowest_wall_util"] = (
        summary["wall_util_mean_pct"].rank(method="min", ascending=True).astype("Int64")
    )

    cols = ["controller"] + [c for c in summary.columns if c != "controller"]
    summary = summary[cols]

    save_table(summary, "table_computational_summary_path_tracking")

    return summary


def build_latex_table(summary):
    cols = [
        "controller",
        "wall_time_mean_ms",
        "wall_time_p95_ms",
        "cpu_time_mean_ms",
        "cpu_time_p95_ms",
        "eq_rate_mean_hz",
        "wall_util_mean_pct",
        "cpu_time_total_ms",
    ]

    latex_df = summary[cols].copy()
    latex_df = latex_df.round(3)

    print("\n=== TABELA LATEX SUGERIDA ===\n")
    print(latex_df.to_latex(index=False))


def print_rankings(summary):
    print("\n=== RANKING: MENOR CUSTO COMPUTACIONAL TOTAL CPU ===")
    for i, row in enumerate(summary.sort_values("cpu_time_total_ms").itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.cpu_time_total_ms:.3f} ms")

    print("\n=== RANKING: MENOR WALL TIME MÉDIO ===")
    for i, row in enumerate(summary.sort_values("wall_time_mean_ms").itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.wall_time_mean_ms:.6f} ms")

    print("\n=== RANKING: MAIOR TAXA EQUIVALENTE ===")
    for i, row in enumerate(summary.sort_values("eq_rate_mean_hz", ascending=False).itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.eq_rate_mean_hz:.2f} Hz")

    print("\n=== RANKING: MENOR UTILIZAÇÃO WALL ===")
    for i, row in enumerate(summary.sort_values("wall_util_mean_pct").itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.wall_util_mean_pct:.4f} %")


def aggregate_time_series(df, value_col):
    grouped = df.groupby("time")[value_col].mean().reset_index()
    return grouped["time"].to_numpy(), grouped[value_col].to_numpy()


def plot_wall_time(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, v = aggregate_time_series(df, "controller_wall_time_ms")
        plt.plot(
            t,
            v,
            label=name,
            color=CONFIG[name]["color"],
            linestyle=CONFIG[name]["ls"],
            linewidth=1.8,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Wall Time [ms]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_path_tracking_wall_time")


def plot_cpu_time_with_inset(dfs):
    plt.figure(figsize=(6.5, 4.2))
    ax = plt.gca()

    cpu_series = {}

    for name, df in dfs.items():
        t, v = aggregate_time_series(df, "controller_cpu_time_ms")
        cpu_series[name] = (t, v)

        ax.plot(
            t,
            v,
            label=name,
            color=CONFIG[name]["color"],
            linestyle=CONFIG[name]["ls"],
            linewidth=1.8,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CPU Time [ms]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    x1, x2 = 10.0, 15.0
    zoom_controllers = [name for name in ["PID", "SMC", "PPO"] if name in cpu_series]

    zoom_values = []

    for name in zoom_controllers:
        t, v = cpu_series[name]
        mask = (t >= x1) & (t <= x2)

        if np.any(mask):
            zoom_values.extend(v[mask].tolist())

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

        for name, (t, v) in cpu_series.items():
            axins.plot(
                t,
                v,
                color=CONFIG[name]["color"],
                linestyle=CONFIG[name]["ls"],
                linewidth=1.3,
            )

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=8)

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0)

    plt.tight_layout()
    save_plot("fig_path_tracking_cpu_time")


def plot_frequency(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, v = aggregate_time_series(df, "controller_frequency_hz")

        plt.plot(
            t,
            v,
            label=name,
            color=CONFIG[name]["color"],
            linestyle=CONFIG[name]["ls"],
            linewidth=1.8,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Equivalent Frequency [Hz]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_path_tracking_frequency")


def plot_summary_bars(summary):
    ordered = summary.sort_values("wall_time_mean_ms")

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(
        ordered["controller"],
        ordered["wall_time_mean_ms"],
    )
    plt.xlabel("Controller")
    plt.ylabel("Mean Wall Time [ms]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("fig_mean_wall_time_bar")

    ordered = summary.sort_values("cpu_time_total_ms")

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(
        ordered["controller"],
        ordered["cpu_time_total_ms"],
    )
    plt.xlabel("Controller")
    plt.ylabel("Total CPU Time [ms]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("fig_total_cpu_time_bar")


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    dfs = load_data()

    if not dfs:
        print("[ERROR] No valid CSV files found.")
        return

    summary = build_summary_table(dfs)
    build_latex_table(summary)

    detailed = pd.concat(dfs.values(), ignore_index=True)
    save_table(detailed, "table_computational_raw_path_tracking")

    plot_wall_time(dfs)
    plot_cpu_time_with_inset(dfs)
    plot_frequency(dfs)
    plot_summary_bars(summary)

    print("\n=== RESUMO CUSTO COMPUTACIONAL PATH TRACKING ===")
    print(summary.round(4).to_string(index=False))

    print_rankings(summary)

    print("[OK] Computational cost pipeline finished.")


if __name__ == "__main__":
    main()