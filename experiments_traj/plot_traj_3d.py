import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mpl_toolkits.mplot3d import Axes3D

# --- REFERENCE RECONSTRUCTION ---
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

def plot_3d_for_article():
    config = {
        'PID': {'file': 'data_pid_traj.csv', 'color': '#1f77b4', 'ls': '-'},
        'MPC': {'file': 'data_mpc_traj.csv', 'color': '#d62728', 'ls': '--'},
        'PPO': {'file': 'data_ppo_traj.csv', 'color': '#2ca02c', 'ls': '-.'}
    }
    
    dfs = {}
    for name, info in config.items():
        if os.path.exists(info['file']):
            df = pd.read_csv(info['file'])
            if 'x' in df.columns:
                dfs[name] = df

    if not dfs:
        print("No data found for 3D plot.")
        return

    # --- TRUNCATION ---
    min_len = min(len(df) for df in dfs.values())
    for name in dfs:
        dfs[name] = dfs[name].iloc[:min_len].reset_index(drop=True)

    t_ref = dfs[list(dfs.keys())[0]]['time'].values
    ref_x, ref_y, ref_z = get_reference_trajectory(t_ref)

    # Global style settings
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

    # --- FIGURE: 3D TRAJECTORY ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Reference
    ax.plot(ref_x, ref_y, ref_z, 'k:', label='Reference', linewidth=2, alpha=0.8)

    # Plot Controllers
    for name, df in dfs.items():
        ax.plot(df['x'], df['y'], df['z'], label=name, 
                color=config[name]['color'], linestyle=config[name]['ls'], linewidth=1.5)

    # Setting labels
    ax.set_xlabel('X position [m]', labelpad=10)
    ax.set_ylabel('Y position [m]', labelpad=10)
    ax.set_zlabel('Depth Z [m]', labelpad=10)

    # Invert Z axis to show depth (NED convention style)
    ax.set_zlim(min(ref_z)-0.2, 0.2)
    ax.invert_zaxis()

    # Academic styling
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.view_init(elev=25, azim=-135) # Good angle for 3D trajectory visualization
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))

    plt.tight_layout()
    plt.savefig('fig_trajectory_3d.pdf', format='pdf', dpi=300)
    print("Saved: fig_trajectory_3d.pdf")
    plt.show()

if __name__ == "__main__":
    plot_3d_for_article()   