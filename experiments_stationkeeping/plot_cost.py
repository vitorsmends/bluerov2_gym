import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ==========================================
# 0. CONFIGURAÇÃO DE SAÍDA
# ==========================================
OUTPUT_DIR = "plots_computational_cost"
TABLE_DIR = "tables_computational_cost"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


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


# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data():
    config = {
        "PID": {"file": "data_pid_stationkeeping.csv", "color": "#1f77b4", "ls": "-"},
        "MPC": {"file": "data_mpc_stationkeeping.csv", "color": "#d62728", "ls": "--"},
        "PPO": {"file": "data_ppo_stationkeeping.csv", "color": "#2ca02c", "ls": "-."},
    }

    required_cols = {
        "controller",
        "scenario_id",
        "time",
        "controller_wall_time_s",
        "controller_cpu_time_s",
        "controller_frequency_hz",
    }

    dfs = {}

    for name, info in config.items():
        path = info["file"]

        if not os.path.exists(path):
            print(f"[AVISO] Arquivo não encontrado: {path}")
            continue

        df = pd.read_csv(path)

        missing = required_cols - set(df.columns)
        if missing:
            print(f"[AVISO] {name} ignorado. Faltam colunas: {sorted(missing)}")
            continue

        df = df.sort_values(["scenario_id", "time"]).reset_index(drop=True)

        # conversões
        df["controller_wall_time_ms"] = df["controller_wall_time_s"] * 1e3
        df["controller_cpu_time_ms"] = df["controller_cpu_time_s"] * 1e3

        # dt por cenário
        df["dt_s"] = df.groupby("scenario_id")["time"].diff()
        df["dt_s"] = df["dt_s"].bfill().ffill()

        # evita divisão por zero / NaN
        df["dt_s"] = df["dt_s"].replace(0.0, np.nan)
        df["dt_ms"] = df["dt_s"] * 1e3

        # taxa equivalente baseada no wall time
        df["equivalent_rate_from_wall_hz"] = np.where(
            df["controller_wall_time_s"] > 0.0,
            1.0 / df["controller_wall_time_s"],
            np.nan,
        )

        # utilização computacional em relação ao passo de simulação
        df["wall_utilization_pct"] = 100.0 * df["controller_wall_time_s"] / df["dt_s"]
        df["cpu_utilization_pct"] = 100.0 * df["controller_cpu_time_s"] / df["dt_s"]

        # deadline miss: quando o tempo computacional excede o passo disponível
        df["wall_deadline_miss"] = df["controller_wall_time_s"] > df["dt_s"]
        df["cpu_deadline_miss"] = df["controller_cpu_time_s"] > df["dt_s"]

        dfs[name] = df

    return dfs, config


# ==========================================
# 2. MÉTRICAS POR CENÁRIO
# ==========================================
def compute_metrics_per_scenario(df):
    metrics = []

    for scenario_id, g in df.groupby("scenario_id"):
        wall_ms = g["controller_wall_time_ms"].to_numpy(dtype=float)
        cpu_ms = g["controller_cpu_time_ms"].to_numpy(dtype=float)
        freq_hz = g["controller_frequency_hz"].to_numpy(dtype=float)
        eq_rate_hz = g["equivalent_rate_from_wall_hz"].to_numpy(dtype=float)
        wall_util = g["wall_utilization_pct"].to_numpy(dtype=float)
        cpu_util = g["cpu_utilization_pct"].to_numpy(dtype=float)

        metrics.append({
            "scenario_id": scenario_id,
            "n_steps": int(len(g)),
            "sim_time_s": float(g["time"].max() - g["time"].min()),

            # wall time
            "wall_time_mean_ms": float(np.nanmean(wall_ms)),
            "wall_time_std_ms": float(np.nanstd(wall_ms)),
            "wall_time_max_ms": float(np.nanmax(wall_ms)),
            "wall_time_p95_ms": float(np.nanpercentile(wall_ms, 95)),
            "wall_time_total_ms": float(np.nansum(wall_ms)),

            # cpu time
            "cpu_time_mean_ms": float(np.nanmean(cpu_ms)),
            "cpu_time_std_ms": float(np.nanstd(cpu_ms)),
            "cpu_time_max_ms": float(np.nanmax(cpu_ms)),
            "cpu_time_p95_ms": float(np.nanpercentile(cpu_ms, 95)),
            "cpu_time_total_ms": float(np.nansum(cpu_ms)),

            # razão cpu/wall
            "cpu_to_wall_ratio": float(np.nansum(cpu_ms) / np.nansum(wall_ms))
            if np.nansum(wall_ms) > 0 else np.nan,

            # frequência reportada
            "freq_mean_hz": float(np.nanmean(freq_hz)),
            "freq_std_hz": float(np.nanstd(freq_hz)),
            "freq_min_hz": float(np.nanmin(freq_hz)),

            # taxa equivalente pelo wall time
            "eq_rate_mean_hz": float(np.nanmean(eq_rate_hz)),
            "eq_rate_min_hz": float(np.nanmin(eq_rate_hz)),
            "eq_rate_p05_hz": float(np.nanpercentile(eq_rate_hz, 5)),

            # utilização
            "wall_util_mean_pct": float(np.nanmean(wall_util)),
            "wall_util_max_pct": float(np.nanmax(wall_util)),
            "cpu_util_mean_pct": float(np.nanmean(cpu_util)),
            "cpu_util_max_pct": float(np.nanmax(cpu_util)),

            # deadline miss
            "wall_deadline_miss_count": int(g["wall_deadline_miss"].sum()),
            "cpu_deadline_miss_count": int(g["cpu_deadline_miss"].sum()),
            "wall_deadline_miss_rate_pct": float(100.0 * g["wall_deadline_miss"].mean()),
            "cpu_deadline_miss_rate_pct": float(100.0 * g["cpu_deadline_miss"].mean()),
        })

    return pd.DataFrame(metrics)


# ==========================================
# 3. AGREGAÇÃO TEMPORAL
# ==========================================
def aggregate_time_series(df, value_col):
    grouped = df.groupby(["scenario_id", "time"])[value_col].mean().reset_index()
    pivot = grouped.pivot(index="time", columns="scenario_id", values=value_col)

    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1).fillna(0.0)

    return mean.index.to_numpy(), mean.to_numpy(), std.to_numpy()


# ==========================================
# 4. TABELAS PARA ARTIGO
# ==========================================
def build_article_summary_table(metrics_dict):
    rows = []

    for name, m in metrics_dict.items():
        rows.append({
            "controller": name,

            "wall_mean_ms": m["wall_time_mean_ms"].mean(),
            "wall_std_ms": m["wall_time_mean_ms"].std(ddof=0),
            "wall_p95_ms": m["wall_time_p95_ms"].mean(),
            "wall_max_ms": m["wall_time_max_ms"].max(),
            "wall_total_ms": m["wall_time_total_ms"].sum(),

            "cpu_mean_ms": m["cpu_time_mean_ms"].mean(),
            "cpu_std_ms": m["cpu_time_mean_ms"].std(ddof=0),
            "cpu_p95_ms": m["cpu_time_p95_ms"].mean(),
            "cpu_max_ms": m["cpu_time_max_ms"].max(),
            "cpu_total_ms": m["cpu_time_total_ms"].sum(),

            "freq_mean_hz": m["freq_mean_hz"].mean(),
            "freq_min_hz": m["freq_min_hz"].min(),
            "eq_rate_mean_hz": m["eq_rate_mean_hz"].mean(),
            "eq_rate_p05_hz": m["eq_rate_p05_hz"].mean(),

            "wall_util_mean_pct": m["wall_util_mean_pct"].mean(),
            "wall_util_max_pct": m["wall_util_max_pct"].max(),
            "cpu_util_mean_pct": m["cpu_util_mean_pct"].mean(),
            "cpu_util_max_pct": m["cpu_util_max_pct"].max(),

            "wall_deadline_miss_rate_pct": m["wall_deadline_miss_rate_pct"].mean(),
            "cpu_deadline_miss_rate_pct": m["cpu_deadline_miss_rate_pct"].mean(),

            "cpu_to_wall_ratio": m["cpu_to_wall_ratio"].mean(),
            "n_scenarios": len(m),
            "mean_steps_per_scenario": m["n_steps"].mean(),
        })

    df = pd.DataFrame(rows)

    # rankings
    df["rank_lowest_cpu_total"] = df["cpu_total_ms"].rank(method="min", ascending=True).astype(int)
    df["rank_lowest_wall_total"] = df["wall_total_ms"].rank(method="min", ascending=True).astype(int)
    df["rank_fastest_response"] = df["wall_mean_ms"].rank(method="min", ascending=True).astype(int)
    df["rank_best_eq_rate"] = df["eq_rate_mean_hz"].rank(method="min", ascending=False).astype(int)
    df["rank_lowest_wall_util"] = df["wall_util_mean_pct"].rank(method="min", ascending=True).astype(int)

    return df.sort_values("controller").reset_index(drop=True)


def build_latex_table(df):
    cols = [
        "controller",
        "wall_mean_ms",
        "wall_p95_ms",
        "cpu_mean_ms",
        "cpu_p95_ms",
        "eq_rate_mean_hz",
        "wall_util_mean_pct",
        "cpu_total_ms",
    ]

    latex_df = df[cols].copy()
    latex_df = latex_df.round(3)

    print("\n=== TABELA LATEX SUGERIDA ===\n")
    print(latex_df.to_latex(index=False))


# ==========================================
# 5. PRINTS DE INSPEÇÃO
# ==========================================
def print_rankings(summary_df):
    print("\n=== RANKING: MENOR CUSTO COMPUTACIONAL (CPU TOTAL) ===")
    rank_cpu = summary_df.sort_values("cpu_total_ms")
    for i, row in enumerate(rank_cpu.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.cpu_total_ms:.3f} ms")

    print("\n=== RANKING: MENOR TEMPO DE RESPOSTA (WALL MEAN) ===")
    rank_wall = summary_df.sort_values("wall_mean_ms")
    for i, row in enumerate(rank_wall.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.wall_mean_ms:.6f} ms")

    print("\n=== RANKING: MAIOR TAXA EQUIVALENTE ===")
    rank_rate = summary_df.sort_values("eq_rate_mean_hz", ascending=False)
    for i, row in enumerate(rank_rate.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.eq_rate_mean_hz:.2f} Hz")

    print("\n=== RANKING: MENOR UTILIZAÇÃO COMPUTACIONAL MÉDIA (WALL) ===")
    rank_util = summary_df.sort_values("wall_util_mean_pct")
    for i, row in enumerate(rank_util.itertuples(index=False), 1):
        print(f"{i}. {row.controller}: {row.wall_util_mean_pct:.4f} %")


def print_interpretation(summary_df):
    cpu_worst = summary_df.loc[summary_df["cpu_total_ms"].idxmax(), "controller"]
    cpu_best = summary_df.loc[summary_df["cpu_total_ms"].idxmin(), "controller"]

    wall_worst = summary_df.loc[summary_df["wall_mean_ms"].idxmax(), "controller"]
    wall_best = summary_df.loc[summary_df["wall_mean_ms"].idxmin(), "controller"]

    rate_best = summary_df.loc[summary_df["eq_rate_mean_hz"].idxmax(), "controller"]
    util_best = summary_df.loc[summary_df["wall_util_mean_pct"].idxmin(), "controller"]

    print("\n=== INTERPRETAÇÃO RÁPIDA ===")
    print(f"Maior custo computacional acumulado (CPU total): {cpu_worst}")
    print(f"Menor custo computacional acumulado (CPU total): {cpu_best}")
    print(f"Maior tempo médio de resposta: {wall_worst}")
    print(f"Menor tempo médio de resposta: {wall_best}")
    print(f"Maior taxa equivalente de processamento: {rate_best}")
    print(f"Menor utilização computacional média: {util_best}")


# ==========================================
# 6. PLOTS PRINCIPAIS
# ==========================================
def plot_computational_cost_results():
    dfs, config = load_data()

    if not dfs:
        print("[ERRO] Nenhum CSV válido encontrado.")
        return

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    metrics_dict = {}

    for name, df in dfs.items():
        metrics = compute_metrics_per_scenario(df)
        metrics.insert(0, "controller", name)
        metrics_dict[name] = metrics
        save_table(metrics, f"table_metrics_per_scenario_{name.lower()}")

    summary_df = build_article_summary_table(metrics_dict)
    save_table(summary_df, "table_computational_summary")
    build_latex_table(summary_df)

    detailed_df = pd.concat(metrics_dict.values(), ignore_index=True)
    save_table(detailed_df, "table_metrics_all_scenarios")

    # ==========================================
    # FIGURA 1: WALL TIME AO LONGO DO TEMPO
    # ==========================================
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_wall_time_ms")

        plt.plot(
            t, mean_v,
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.8,
        )

        plt.fill_between(
            t,
            mean_v - std_v,
            mean_v + std_v,
            color=config[name]["color"],
            alpha=0.15,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Wall Time [ms]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_computational_wall_time")

    # ==========================================
    # FIGURA 2: CPU TIME AO LONGO DO TEMPO + INSET ZOOM
    # ==========================================
    plt.figure(figsize=(6.5, 4.2))
    ax = plt.gca()

    cpu_series = {}

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_cpu_time_ms")
        cpu_series[name] = (t, mean_v, std_v)

        ax.plot(
            t, mean_v,
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.8,
        )

        ax.fill_between(
            t,
            mean_v - std_v,
            mean_v + std_v,
            color=config[name]["color"],
            alpha=0.15,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CPU Time [ms]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    # região do zoom
    x1, x2 = 10.0, 15.0

    zoom_values = []
    for ctrl_name in ["PID", "PPO"]:
        if ctrl_name in cpu_series:
            t, mean_v, std_v = cpu_series[ctrl_name]
            mask = (t >= x1) & (t <= x2)
            if np.any(mask):
                zoom_values.extend((mean_v[mask] - std_v[mask]).tolist())
                zoom_values.extend((mean_v[mask] + std_v[mask]).tolist())

    if len(zoom_values) > 0:
        y1 = max(0.0, np.min(zoom_values) * 0.95)
        y2 = np.max(zoom_values) * 1.05

        axins = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1.2)

        for name, (t, mean_v, std_v) in cpu_series.items():
            axins.plot(
                t, mean_v,
                color=config[name]["color"],
                linestyle=config[name]["ls"],
                linewidth=1.4,
            )
            axins.fill_between(
                t,
                mean_v - std_v,
                mean_v + std_v,
                color=config[name]["color"],
                alpha=0.15,
            )

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=8)

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=1.0)

    plt.tight_layout()
    save_plot("fig_computational_cpu_time")

    # ==========================================
    # FIGURA 3: FREQUÊNCIA AO LONGO DO TEMPO
    # ==========================================
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_frequency_hz")

        plt.plot(
            t, mean_v,
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.8,
        )

        plt.fill_between(
            t,
            np.maximum(mean_v - std_v, 0.0),
            mean_v + std_v,
            color=config[name]["color"],
            alpha=0.15,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Equivalent Frequency [Hz]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_computational_frequency")

    # ==========================================
    # RESUMO NO TERMINAL
    # ==========================================
    print("\n=== RESUMO CUSTO COMPUTACIONAL ===")
    print(summary_df.round(4).to_string(index=False))

    print_rankings(summary_df)
    print_interpretation(summary_df)

    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    plot_computational_cost_results()