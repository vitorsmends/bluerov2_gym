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
        z_d = (z_target / 20.0) * t if t < 20.0 else z_target
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
    required_cols = {"time", "x", "y", "z"}
    dfs = {}

    for name, info in config.items():
        if os.path.exists(info["file"]):
            df = pd.read_csv(info["file"])
            if required_cols.issubset(df.columns):
                dfs[name] = df.sort_values("time").reset_index(drop=True)
            else:
                print(f"[AVISO] Colunas ausentes em {name}")
        else:
            print(f"[AVISO] Arquivo não encontrado: {info['file']}")
    return dfs, config

# ==========================================
# 3. AJUSTE DE ASPECT RATIO 3D
# ==========================================
def set_axes_equal(ax):
    # Garante que a escala X, Y e Z seja a mesma para não deformar a trajetória
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

# ==========================================
# 4. PLOT 3D ACADÊMICO
# ==========================================
def plot_3d_trajectory():
    dfs, config = load_data()
    
    # --- Configurações de Estilo ---
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": [12, 4] # Sua solicitação de 12x4
    })

    if not dfs:
        print("[ERRO] Nenhum CSV válido. Gerando apenas referência para teste.")
        t_ref = np.arange(0, 50, 0.1)
    else:
        max_time = max(df["time"].max() for df in dfs.values())
        t_ref = np.arange(0.0, max_time + 0.1, 0.1)

    ref_x, ref_y, ref_z = get_reference_trajectory(t_ref)

    # CRIANDO A FIGURA (Removido o figsize daqui para usar o do rcParams)
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection="3d")

    # Plot Referência
    ax.plot(ref_x, ref_y, ref_z, color="black", linestyle=":", linewidth=2, label="Reference")

    # Plot Controladores
    for name, df in dfs.items():
        ax.plot(df["x"], df["y"], df["z"], 
                color=config[name]["color"], linestyle=config[name]["ls"], 
                linewidth=1.5, label=name)

    # Pontos de Início/Fim
    ax.scatter(ref_x[0], ref_y[0], ref_z[0], color="black", marker="o", s=30, label="Start")
    ax.scatter(ref_x[-1], ref_y[-1], ref_z[-1], color="black", marker="^", s=40, label="End")

    # Estética dos Eixos
    ax.set_xlabel("X position [m]", labelpad=5)
    ax.set_ylabel("Y position [m]", labelpad=5)
    ax.set_zlabel("Depth Z [m]", labelpad=5)
    
    # Visão otimizada para formato horizontal
    ax.view_init(elev=20, azim=-60)
    set_axes_equal(ax)

    # Limpeza visual (Panes transparentes)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    
    # Legenda fora do gráfico ou ajustada para não poluir
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=True, edgecolor="black")

    plt.tight_layout()
    
    # Salvamento
    plt.savefig("fig_trajectory_3d.png", dpi=300, bbox_inches="tight")
    print("[OK] Gráfico salvo como fig_trajectory_3d.png")
    plt.show()

if __name__ == "__main__":
    plot_3d_trajectory()