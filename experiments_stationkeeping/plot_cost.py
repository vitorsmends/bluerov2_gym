import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 0. CONFIGURAÇÃO DE SAÍDA
# ==========================================
OUTPUT_DIR = "plots_computational_cost"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(name):
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")

    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


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

        # conversões úteis
        df["controller_wall_time_ms"] = df["controller_wall_time_s"] * 1e3
        df["controller_cpu_time_ms"] = df["controller_cpu_time_s"] * 1e3

        dfs[name] = df

    return dfs, config


# ==========================================
# 2. MÉTRICAS
# ==========================================
def compute_metrics(df):
    metrics = []

    for scenario_id, g in df.groupby("scenario_id"):
        wall_ms = g["controller_wall_time_ms"].to_numpy(dtype=float)
        cpu_ms = g["controller_cpu_time_ms"].to_numpy(dtype=float)
        freq_hz = g["controller_frequency_hz"].to_numpy(dtype=float)

        metrics.append({
            "scenario_id": scenario_id,
            "wall_time_mean_ms": float(np.mean(wall_ms)),
            "wall_time_std_ms": float(np.std(wall_ms)),
            "wall_time_max_ms": float(np.max(wall_ms)),
            "cpu_time_mean_ms": float(np.mean(cpu_ms)),
            "cpu_time_std_ms": float(np.std(cpu_ms)),
            "cpu_time_max_ms": float(np.max(cpu_ms)),
            "freq_mean_hz": float(np.mean(freq_hz)),
            "freq_std_hz": float(np.std(freq_hz)),
            "freq_min_hz": float(np.min(freq_hz)),
            "n_steps": int(len(g)),
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
# 4. BOXPLOT
# ==========================================
def boxplot_from_metrics(metrics_dict, key, ylabel, filename):
    labels = []
    series = []

    for name, m in metrics_dict.items():
        labels.append(name)
        series.append(m[key].to_numpy(dtype=float))

    plt.figure(figsize=(6, 4))
    plt.boxplot(series, labels=labels, showmeans=True)

    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_plot(filename)


# ==========================================
# 5. PLOTS PRINCIPAIS
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

    metrics_dict = {name: compute_metrics(df) for name, df in dfs.items()}

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
    # FIGURA 2: CPU TIME AO LONGO DO TEMPO
    # ==========================================
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_v, std_v = aggregate_time_series(df, "controller_cpu_time_ms")

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
    plt.ylabel("CPU Time [ms]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
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
    # FIGURA 4: WALL TIME BOXPLOT
    # ==========================================
    boxplot_from_metrics(
        metrics_dict,
        key="wall_time_mean_ms",
        ylabel="Mean Wall Time per Scenario [ms]",
        filename="fig_computational_wall_time_boxplot"
    )

    # ==========================================
    # FIGURA 5: CPU TIME BOXPLOT
    # ==========================================
    boxplot_from_metrics(
        metrics_dict,
        key="cpu_time_mean_ms",
        ylabel="Mean CPU Time per Scenario [ms]",
        filename="fig_computational_cpu_time_boxplot"
    )

    # ==========================================
    # FIGURA 6: FREQUENCY BOXPLOT
    # ==========================================
    boxplot_from_metrics(
        metrics_dict,
        key="freq_mean_hz",
        ylabel="Mean Equivalent Frequency per Scenario [Hz]",
        filename="fig_computational_frequency_boxplot"
    )

    # ==========================================
    # RESUMO NO TERMINAL
    # ==========================================
    print("\n=== RESUMO CUSTO COMPUTACIONAL ===")
    for name, m in metrics_dict.items():
        print(
            f"{name}: "
            f"Wall time médio={m['wall_time_mean_ms'].mean():.6f} ms | "
            f"CPU time médio={m['cpu_time_mean_ms'].mean():.6f} ms | "
            f"Frequência média={m['freq_mean_hz'].mean():.2f} Hz | "
            f"Steps médios por cenário={m['n_steps'].mean():.1f}"
        )

    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    plot_computational_cost_results()