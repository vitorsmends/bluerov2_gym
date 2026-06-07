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
    "PID":  {"file": "pid.csv",  "color": "#4E79A7", "ls": "-"},
    "SMC":  {"file": "smc.csv",  "color": "#B07AA1", "ls": "-"},
    "NMPC": {"file": "nmpc.csv", "color": "#E15759", "ls": "-"},
    "PPO":  {"file": "ppo.csv",  "color": "#59A14F", "ls": "-"},
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

        if "repetition_id" not in df.columns:
            df["repetition_id"] = 0

        df = df.sort_values(["repetition_id", "time"]).reset_index(drop=True)
        df["controller"] = name

        metric_cols = [
            "controller_wall_time_s",
            "controller_cpu_time_s",
            "controller_frequency_hz",
        ]

        df[metric_cols] = df[metric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=metric_cols).copy()

        if df.empty:
            print(f"[WARN] {name} skipped. No valid computational metrics.")
            continue

        df["controller_wall_time_ms"] = df["controller_wall_time_s"] * 1e3
        df["controller_cpu_time_ms"] = df["controller_cpu_time_s"] * 1e3

        df["dt_s"] = df.groupby("repetition_id")["time"].diff()
        df["dt_s"] = df.groupby("repetition_id")["dt_s"].bfill()
        df["dt_s"] = df.groupby("repetition_id")["dt_s"].ffill()
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


def compute_metrics_for_group(g):
    wall_ms = g["controller_wall_time_ms"].to_numpy(dtype=float)
    cpu_ms = g["controller_cpu_time_ms"].to_numpy(dtype=float)
    freq_hz = g["controller_frequency_hz"].to_numpy(dtype=float)
    eq_rate_hz = g["equivalent_rate_from_wall_hz"].to_numpy(dtype=float)
    wall_util = g["wall_utilization_pct"].to_numpy(dtype=float)
    cpu_util = g["cpu_utilization_pct"].to_numpy(dtype=float)

    return {
        "n_steps": int(len(g)),
        "sim_time_s": float(g["time"].max() - g["time"].min()),

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

        "wall_deadline_miss_count": int(g["wall_deadline_miss"].sum()),
        "cpu_deadline_miss_count": int(g["cpu_deadline_miss"].sum()),
        "wall_deadline_miss_rate_pct": float(100.0 * g["wall_deadline_miss"].mean()),
        "cpu_deadline_miss_rate_pct": float(100.0 * g["cpu_deadline_miss"].mean()),
    }


def build_repetition_metrics(dfs):
    rows = []

    for name, df in dfs.items():
        for rep, g in df.groupby("repetition_id"):
            metrics = compute_metrics_for_group(g)
            metrics["controller"] = name
            metrics["repetition_id"] = rep
            rows.append(metrics)

    rep_df = pd.DataFrame(rows)
    save_table(rep_df, "table_computational_metrics_per_repetition")

    return rep_df


def build_summary_table(rep_df):
    metric_cols = [
        "wall_time_mean_ms",
        "wall_time_p95_ms",
        "wall_time_max_ms",
        "wall_time_total_ms",
        "cpu_time_mean_ms",
        "cpu_time_p95_ms",
        "cpu_time_max_ms",
        "cpu_time_total_ms",
        "freq_mean_hz",
        "freq_min_hz",
        "eq_rate_mean_hz",
        "eq_rate_p05_hz",
        "wall_util_mean_pct",
        "wall_util_max_pct",
        "cpu_util_mean_pct",
        "cpu_util_max_pct",
        "wall_deadline_miss_rate_pct",
        "cpu_deadline_miss_rate_pct",
        "cpu_to_wall_ratio",
    ]

    rows = []

    for controller, g in rep_df.groupby("controller"):
        row = {
            "controller": controller,
            "n_repetitions": int(g["repetition_id"].nunique()),
            "mean_steps": float(g["n_steps"].mean()),
            "mean_sim_time_s": float(g["sim_time_s"].mean()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary["rank_lowest_cpu_total"] = (
        summary["cpu_time_total_ms_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_lowest_wall_total"] = (
        summary["wall_time_total_ms_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_fastest_response"] = (
        summary["wall_time_mean_ms_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    summary["rank_best_eq_rate"] = (
        summary["eq_rate_mean_hz_mean"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    summary["rank_lowest_wall_util"] = (
        summary["wall_util_mean_pct_mean"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    save_table(summary, "table_computational_summary_path_tracking")

    return summary


def build_latex_table(summary):
    cols = [
        "controller",
        "wall_time_mean_ms_mean",
        "wall_time_mean_ms_std",
        "wall_time_p95_ms_mean",
        "cpu_time_mean_ms_mean",
        "cpu_time_mean_ms_std",
        "cpu_time_p95_ms_mean",
        "eq_rate_mean_hz_mean",
        "wall_util_mean_pct_mean",
    ]

    latex_df = summary[cols].copy().round(4)

    print("\n=== TABELA LATEX SUGERIDA ===\n")
    print(latex_df.to_latex(index=False))


def print_rankings(summary):
    print("\n=== RANKING: MENOR CUSTO COMPUTACIONAL TOTAL CPU ===")
    rank_cpu = summary.sort_values("cpu_time_total_ms_mean")
    for i, row in enumerate(rank_cpu.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.cpu_time_total_ms_mean:.3f} ms")

    print("\n=== RANKING: MENOR WALL TIME MÉDIO ===")
    rank_wall = summary.sort_values("wall_time_mean_ms_mean")
    for i, row in enumerate(rank_wall.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.wall_time_mean_ms_mean:.6f} ms")

    print("\n=== RANKING: MAIOR TAXA EQUIVALENTE ===")
    rank_rate = summary.sort_values("eq_rate_mean_hz_mean", ascending=False)
    for i, row in enumerate(rank_rate.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.eq_rate_mean_hz_mean:.2f} Hz")

    print("\n=== RANKING: MENOR UTILIZAÇÃO WALL ===")
    rank_util = summary.sort_values("wall_util_mean_pct_mean")
    for i, row in enumerate(rank_util.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.wall_util_mean_pct_mean:.4f} %")


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


def plot_wall_time(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_wall_time_ms")
        color = CONFIG[name]["color"]

        plt.plot(
            t,
            mean_v,
            label=name,
            color=color,
            linestyle=CONFIG[name]["ls"],
            linewidth=1.8,
        )

        plt.fill_between(
            t,
            np.maximum(mean_v - std_v, 0.0),
            mean_v + std_v,
            color=color,
            alpha=0.18,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Wall Time [ms]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_path_tracking_wall_time_mean_std")


def plot_cpu_time_with_inset(dfs):
    plt.figure(figsize=(6.5, 4.2))
    ax = plt.gca()

    cpu_series = {}

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_cpu_time_ms")
        cpu_series[name] = (t, mean_v, std_v)

        ax.plot(
            t,
            mean_v,
            label=name,
            color=CONFIG[name]["color"],
            linestyle=CONFIG[name]["ls"],
            linewidth=1.8,
        )

        ax.fill_between(
            t,
            np.maximum(mean_v - std_v, 0.0),
            mean_v + std_v,
            color=CONFIG[name]["color"],
            alpha=0.18,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CPU Time [ms]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    x1, x2 = 10.0, 15.0
    zoom_controllers = [name for name in ["PID", "SMC", "PPO"] if name in cpu_series]

    zoom_values = []

    for name in zoom_controllers:
        t, mean_v, std_v = cpu_series[name]
        mask = (t >= x1) & (t <= x2)

        if np.any(mask):
            zoom_values.extend((mean_v[mask] - std_v[mask]).tolist())
            zoom_values.extend((mean_v[mask] + std_v[mask]).tolist())

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

        for name, (t, mean_v, std_v) in cpu_series.items():
            axins.plot(
                t,
                mean_v,
                color=CONFIG[name]["color"],
                linestyle=CONFIG[name]["ls"],
                linewidth=1.3,
            )

            axins.fill_between(
                t,
                np.maximum(mean_v - std_v, 0.0),
                mean_v + std_v,
                color=CONFIG[name]["color"],
                alpha=0.18,
            )

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=8)

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0)

    plt.tight_layout()
    save_plot("fig_path_tracking_cpu_time_mean_std")


def plot_frequency(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_frequency_hz")
        color = CONFIG[name]["color"]

        plt.plot(
            t,
            mean_v,
            label=name,
            color=color,
            linestyle=CONFIG[name]["ls"],
            linewidth=1.8,
        )

        plt.fill_between(
            t,
            np.maximum(mean_v - std_v, 0.0),
            mean_v + std_v,
            color=color,
            alpha=0.18,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Equivalent Frequency [Hz]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_path_tracking_frequency_mean_std")


def plot_summary_bars(summary):
    ordered = summary.sort_values("wall_time_mean_ms_mean")
    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(
        ordered["controller"],
        ordered["wall_time_mean_ms_mean"],
        yerr=ordered["wall_time_mean_ms_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel("Mean Wall Time [ms]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("fig_mean_wall_time_bar")

    ordered = summary.sort_values("cpu_time_total_ms_mean")
    colors = [CONFIG[c]["color"] for c in ordered["controller"]]

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(
        ordered["controller"],
        ordered["cpu_time_total_ms_mean"],
        yerr=ordered["cpu_time_total_ms_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel("Controller")
    plt.ylabel("Total CPU Time [ms]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_plot("fig_total_cpu_time_bar")


def plot_boxplot(rep_df, metric, ylabel, filename):
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


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    dfs = load_data()

    if not dfs:
        print("[ERROR] No valid CSV files found.")
        return

    rep_df = build_repetition_metrics(dfs)
    summary = build_summary_table(rep_df)
    build_latex_table(summary)

    detailed = pd.concat(dfs.values(), ignore_index=True)
    save_table(detailed, "table_computational_raw_path_tracking")

    plot_wall_time(dfs)
    plot_cpu_time_with_inset(dfs)
    plot_frequency(dfs)
    plot_summary_bars(summary)

    plot_boxplot(
        rep_df,
        metric="wall_time_mean_ms",
        ylabel="Mean Wall Time [ms]",
        filename="boxplot_wall_time_mean",
    )

    plot_boxplot(
        rep_df,
        metric="cpu_time_mean_ms",
        ylabel="Mean CPU Time [ms]",
        filename="boxplot_cpu_time_mean",
    )

    plot_boxplot(
        rep_df,
        metric="eq_rate_mean_hz",
        ylabel="Equivalent Rate [Hz]",
        filename="boxplot_equivalent_rate",
    )

    plot_boxplot(
        rep_df,
        metric="wall_util_mean_pct",
        ylabel="Wall Utilization [%]",
        filename="boxplot_wall_utilization",
    )

    print("\n=== RESUMO CUSTO COMPUTACIONAL PATH TRACKING ===")
    print(summary.round(4).to_string(index=False))

    print_rankings(summary)

    print("[OK] Computational cost pipeline finished.")


if __name__ == "__main__":
    main()