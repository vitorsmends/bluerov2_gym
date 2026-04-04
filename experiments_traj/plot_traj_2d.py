import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_reference_trajectory(t_array):
    radius = 1.0
    speed = 0.15
    z_target = -0.5

    ref_x, ref_y, ref_z = [], [], []

    for t in t_array:
        t_s = t * speed

        ref_x.append(radius * math.sin(t_s))
        ref_y.append(radius * math.sin(t_s) * math.cos(t_s))

        if t < 20.0:
            ref_z.append((z_target / 20.0) * t)
        else:
            ref_z.append(z_target)

    return np.array(ref_x), np.array(ref_y), np.array(ref_z)


def load_valid_data(config):
    required_cols = {"time", "x", "y", "z", "error"}
    dfs = {}

    for name, info in config.items():
        file_path = info["file"]

        if not os.path.exists(file_path):
            print(f"[AVISO] Arquivo não encontrado: {file_path}")
            continue

        df = pd.read_csv(file_path)

        missing = required_cols - set(df.columns)
        if missing:
            print(f"[AVISO] {name} ignorado. Faltam colunas obrigatórias: {sorted(missing)}")
            continue

        df = df.sort_values("time").reset_index(drop=True)
        dfs[name] = df

    return dfs


def plot_for_article():
    config = {
        "PID": {
            "file": "data_pid_traj.csv",
            "color": "#1f77b4",
            "ls": "-"
        },
        "MPC": {
            "file": "data_mpc_traj.csv",
            "color": "#d62728",
            "ls": "--"
        },
        "PPO": {
            "file": "data_ppo_traj.csv",
            "color": "#2ca02c",
            "ls": "-."
        },
        "SMC": {
            "file": "data_smc_traj.csv",
            "color": "#9467bd",
            "ls": "-"
        },
    }

    dfs = load_valid_data(config)

    if not dfs:
        print("[ERRO] Nenhum dado válido encontrado.")
        return

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    max_time = max(df["time"].max() for df in dfs.values())
    t_ref = np.arange(0.0, max_time + 0.1, 0.1)
    ref_x, ref_y, ref_z = get_reference_trajectory(t_ref)

    # FIGURA 1: XY
    plt.figure(figsize=(6, 5))
    plt.plot(ref_x, ref_y, "k:", label="Reference", linewidth=2)

    for name, df in dfs.items():
        plt.plot(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.5,
        )

    plt.xlabel("X position [m]")
    plt.ylabel("Y position [m]")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("fig_trajectory_xy.pdf", format="pdf", dpi=300, bbox_inches="tight")
    print("[OK] Saved: fig_trajectory_xy.pdf")

    # FIGURA 2: Z
    plt.figure(figsize=(6, 4))
    plt.plot(t_ref, ref_z, "k:", label="Reference", linewidth=2)

    for name, df in dfs.items():
        plt.plot(
            df["time"].to_numpy(),
            df["z"].to_numpy(),
            label=name,
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.5,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Depth Z [m]")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("fig_depth_tracking.pdf", format="pdf", dpi=300, bbox_inches="tight")
    print("[OK] Saved: fig_depth_tracking.pdf")

    # FIGURA 3: erro
    plt.figure(figsize=(6, 4))
    for name, df in dfs.items():
        err = df["error"].to_numpy(dtype=float)
        rmse = np.sqrt(np.mean(err ** 2))

        plt.plot(
            df["time"].to_numpy(),
            err,
            label=f"{name} (RMSE: {rmse:.3f} m)",
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.5,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("fig_tracking_error.pdf", format="pdf", dpi=300, bbox_inches="tight")
    print("[OK] Saved: fig_tracking_error.pdf")

    # FIGURA 4: energia acumulada total
    plt.figure(figsize=(6, 4))
    plotted_energy = False

    for name, df in dfs.items():
        if "total_cum_energy_J" not in df.columns:
            print(f"[AVISO] {name} não possui a coluna 'total_cum_energy_J'.")
            continue

        energy = df["total_cum_energy_J"].to_numpy(dtype=float)

        plt.plot(
            df["time"].to_numpy(),
            energy,
            label=f"{name} (Final: {energy[-1]:.2f} J)",
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.5,
        )
        plotted_energy = True

    if plotted_energy:
        plt.xlabel("Time [s]")
        plt.ylabel("Cumulative Energy [J]")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        plt.savefig("fig_cumulative_energy.pdf", format="pdf", dpi=300, bbox_inches="tight")
        print("[OK] Saved: fig_cumulative_energy.pdf")
    else:
        print("[AVISO] Nenhum arquivo possui dados de energia. Figura de energia não foi gerada.")
        plt.close()

    plt.show()


if __name__ == "__main__":
    plot_for_article()