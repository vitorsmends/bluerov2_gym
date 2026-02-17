import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

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

def plot_for_article():
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
        print("No data found.")
        return

    # --- TRUNCATION ---
    min_len = min(len(df) for df in dfs.values())
    for name in dfs:
        dfs[name] = dfs[name].iloc[:min_len].reset_index(drop=True)

    t_ref = dfs[list(dfs.keys())[0]]['time'].values
    ref_x, ref_y, ref_z = get_reference_trajectory(t_ref)

    # Global style settings for academic papers
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    # --- FIGURE 1: 2D TRAJECTORY (XY PLANE) ---
    plt.figure(figsize=(6, 5))
    plt.plot(ref_x, ref_y, 'k:', label='Reference', linewidth=2)
    for name, df in dfs.items():
        plt.plot(df['x'], df['y'], label=name, color=config[name]['color'], 
                 linestyle=config[name]['ls'], linewidth=1.5)
    plt.xlabel('X position [m]')
    plt.ylabel('Y position [m]')
    plt.axis('equal')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('fig_trajectory_xy.pdf', format='pdf', dpi=300)
    print("Saved: fig_trajectory_xy.pdf")

    # --- FIGURE 2: DEPTH TRACKING (Z AXIS) ---
    plt.figure(figsize=(6, 4))
    plt.plot(t_ref, ref_z, 'k:', label='Reference', linewidth=2)
    for name, df in dfs.items():
        plt.plot(df['time'], df['z'], label=name, color=config[name]['color'], 
                 linestyle=config[name]['ls'])
    plt.xlabel('Time [s]')
    plt.ylabel('Depth Z [m]')
    plt.gca().invert_yaxis() # Depth convention
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('fig_depth_tracking.pdf', format='pdf', dpi=300)
    print("Saved: fig_depth_tracking.pdf")

    # --- FIGURE 3: TRACKING ERROR ---
    plt.figure(figsize=(6, 4))
    for name, df in dfs.items():
        err = df['error']
        rmse = np.sqrt(np.mean(err**2))
        plt.plot(df['time'], err, label=f'{name} (RMSE: {rmse:.3f})', 
                 color=config[name]['color'], linestyle=config[name]['ls'])
    plt.xlabel('Time [s]')
    plt.ylabel('Position Error [m]')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fig_tracking_error.pdf', format='pdf', dpi=300)
    print("Saved: fig_tracking_error.pdf")

    plt.show()

if __name__ == "__main__":
    plot_for_article()