import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Estilo acadêmico refinado
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": 'tight'
})

def generate_mpc_ppo_comparison():
    # Comparação focada apenas nos dois controladores
    experiments = [
        ("data_mpc.csv", "MPC (Optimization-based)", "#ff7f0e", "-"),
        ("data_ppo.csv", "PPO Agent (RL-based)", "#d62728", "-")
    ]
    
    processed_data = []
    for file, label, color, style in experiments:
        try:
            df = pd.read_csv(file)
            processed_data.append({
                "df": df,
                "label": label,
                "color": color,
                "style": style
            })
        except FileNotFoundError:
            print(f"Warning: {file} not found. Please ensure the experiments were run.")

    if len(processed_data) < 2:
        print("Error: Missing data files for comparison.")
        return

    # --- PLOT: PPO vs MPC Comparison ---
    plt.figure(figsize=(8, 5))
    
    for data in processed_data:
        plt.plot(data["df"]["time"], data["df"]["error"], 
                 label=data["label"], color=data["color"], 
                 linestyle=data["style"], linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend(loc='upper right', frameon=True)
    
    # Adicionando preenchimento entre as curvas para destacar a diferença de performance
    # Assumindo que ambos tenham o mesmo vetor de tempo
    t = processed_data[0]["df"]["time"]
    e_mpc = processed_data[0]["df"]["error"]
    e_ppo = processed_data[1]["df"]["error"]
    
    # Limita o tempo para o menor entre os dois caso haja diferença
    min_len = min(len(e_mpc), len(e_ppo))
    plt.fill_between(t[:min_len], e_mpc[:min_len], e_ppo[:min_len], 
                     color='gray', alpha=0.1, label='Performance Gap')

    plt.savefig("fig_comparison_mpc_ppo.png")
    plt.show()

    print("Success: fig_comparison_mpc_ppo.png generated.")

if __name__ == "__main__":
    generate_mpc_ppo_comparison()