import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. TRAJETÓRIA DE REFERÊNCIA
# ==========================================
def get_reference_trajectory(t_array):
    radius = 1.0
    speed = 0.15
    z_target = -0.5

    ref_x, ref_y, ref_z = [], [], []

    for t in t_array:
        t_s = t * speed

        x_d = radius * math.sin(t_s)
        y_d = radius * math.sin(t_s) * math.cos(t_s)

        if t < 20.0:
            z_d = (z_target / 20.0) * t
        else:
            z_d = z_target

        ref_x.append(x_d)
        ref_y.append(y_d)
        ref_z.append(z_d)

    return np.array(ref_x), np.array(ref_y), np.array(ref_z)


# ==========================================
# 2. LEITURA DOS DADOS
# ==========================================
def load_data():
    config = {
        "PID": {"file": "data_pid_traj.csv", "color": "#1f77b4", "ls": "-"},
        "MPC": {"file": "data_mpc_traj.csv", "color": "#d62728", "ls": "--"},
        "PPO": {"file": "data_ppo_traj.csv", "color": "#2ca02c", "ls": "-."},
    }

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
            print(f"[AVISO] {name} ignorado. Colunas ausentes: {sorted(missing)}")
            continue

        df = df.sort_values("time").reset_index(drop=True)
        dfs[name] = df

    return dfs, config


# ==========================================
# 3. AJUSTE DE ASPECT RATIO 3D
# ==========================================
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# ==========================================
# 4. PLOT 3D ACADÊMICO
# ==========================================
def plot_3d_trajectory():
    dfs, config = load_data()

    if not dfs:
        print("[ERRO] Nenhum CSV válido encontrado.")
        return

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    max_time = max(df["time"].max() for df in dfs.values())
    t_ref = np.arange(0.0, max_time + 0.1, 0.1)
    ref_x, ref_y, ref_z = get_reference_trajectory(t_ref)

    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # referência
    ax.plot(
        ref_x,
        ref_y,
        ref_z,
        color="black",
        linestyle=":",
        linewidth=2.2,
        label="Reference"
    )

    # trajetórias dos controladores
    for name, df in dfs.items():
        ax.plot(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["z"].to_numpy(),
            color=config[name]["color"],
            linestyle=config[name]["ls"],
            linewidth=1.8,
            label=name
        )

    # início da trajetória
    ax.scatter(
        ref_x[0], ref_y[0], ref_z[0],
        color="black",
        marker="o",
        s=35,
        label="Start"
    )

    # fim da trajetória
    ax.scatter(
        ref_x[-1], ref_y[-1], ref_z[-1],
        color="black",
        marker="^",
        s=45,
        label="End"
    )

    ax.set_xlabel("X position [m]", labelpad=10)
    ax.set_ylabel("Y position [m]", labelpad=10)
    ax.set_zlabel("Depth Z [m]", labelpad=10)

    ax.view_init(elev=24, azim=-58)
    set_axes_equal(ax)

    # grade discreta
    ax.grid(True)

    # deixa o fundo mais limpo
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # legenda acadêmica
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black"
    )

    plt.tight_layout()
    plt.savefig("fig_trajectory_3d.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig("fig_trajectory_3d.png", format="png", dpi=300, bbox_inches="tight")
    print("[OK] Saved: fig_trajectory_3d.pdf")
    print("[OK] Saved: fig_trajectory_3d.png")

    plt.show()


if __name__ == "__main__":
    plot_3d_trajectory()