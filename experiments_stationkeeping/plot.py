import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 0. CONFIGURAÇÃO DE SAÍDA
# ==========================================
OUTPUT_DIR = "plots_stationkeeping"
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
        "controller", "scenario_id", "time",
        "target_x", "target_y", "target_z",
        "x", "y", "z", "error",
        "total_power_W", "total_step_energy_J", "total_cum_energy_J"
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
        dfs[name] = df

    return dfs, config


# ==========================================
# 2. MÉTRICAS
# ==========================================
def compute_metrics(df):
    metrics = []

    for scenario_id, g in df.groupby("scenario_id"):
        err = g["error"].to_numpy(dtype=float)

        rmse = float(np.sqrt(np.mean(err ** 2)))
        final_energy = float(g["total_cum_energy_J"].iloc[-1])
        mean_power = float(g["total_power_W"].mean())

        settle_like = float((err[-50:] if len(err) >= 50 else err).mean())

        metrics.append({
            "scenario_id": scenario_id,
            "rmse": rmse,
            "final_energy": final_energy,
            "mean_power": mean_power,
            "late_error_mean": settle_like,
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
def plot_stationkeeping_results():
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
    # FIGURA 1: ERRO AO LONGO DO TEMPO
    # ==========================================
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_err, std_err = aggregate_time_series(df, "error")

        plt.plot(
            t, mean_err,
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.8,
        )

        plt.fill_between(
            t,
            mean_err - std_err,
            mean_err + std_err,
            color=config[name]["color"],
            alpha=0.15,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot("fig_stationkeeping_error_time")

    # ==========================================
    # FIGURA 2: ENERGIA ACUMULADA
    # ==========================================
    plt.figure(figsize=(6.5, 4.2))

    for name, df in dfs.items():
        t, mean_e, std_e = aggregate_time_series(df, "total_cum_energy_J")

        plt.plot(
            t, mean_e,
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.8,
        )

        plt.fill_between(
            t,
            mean_e - std_e,
            mean_e + std_e,
            color=config[name]["color"],
            alpha=0.15,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative Energy [J]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot("fig_stationkeeping_cumulative_energy")

    # ==========================================
    # FIGURA 3: RMSE BOXPLOT
    # ==========================================
    boxplot_from_metrics(
        metrics_dict,
        key="rmse",
        ylabel="RMSE of Position Error [m]",
        filename="fig_stationkeeping_rmse_boxplot"
    )

    # ==========================================
    # FIGURA 4: ENERGIA FINAL BOXPLOT
    # ==========================================
    boxplot_from_metrics(
        metrics_dict,
        key="final_energy",
        ylabel="Final Cumulative Energy [J]",
        filename="fig_stationkeeping_energy_boxplot"
    )

    # ==========================================
    # FIGURA 5: RMSE POR CENÁRIO
    # ==========================================
    scenario_ids = sorted(
        set().union(*[set(m["scenario_id"].tolist()) for m in metrics_dict.values()])
    )

    x = np.arange(len(scenario_ids))
    width = 0.25

    plt.figure(figsize=(7.2, 4.2))
    offset_map = {"PID": -width, "MPC": 0.0, "PPO": width}

    for name, metrics in metrics_dict.items():
        vals = []

        for sid in scenario_ids:
            row = metrics[metrics["scenario_id"] == sid]
            vals.append(float(row["rmse"].iloc[0]) if not row.empty else np.nan)

        plt.bar(
            x + offset_map.get(name, 0.0),
            vals,
            width=width,
            label=name,
            color=config[name]["color"],
            alpha=0.9,
        )

    plt.xticks(x, [str(s) for s in scenario_ids])
    plt.xlabel("Scenario ID")
    plt.ylabel("RMSE [m]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot("fig_stationkeeping_rmse_by_scenario")

    # ==========================================
    # RESUMO NO TERMINAL
    # ==========================================
    print("\n=== RESUMO MÉTRICAS ===")
    for name, m in metrics_dict.items():
        print(
            f"{name}: "
            f"RMSE médio={m['rmse'].mean():.4f} m | "
            f"Energia final média={m['final_energy'].mean():.2f} J | "
            f"Potência média={m['mean_power'].mean():.2f} W"
        )

    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    plot_stationkeeping_results()