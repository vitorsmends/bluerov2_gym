import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo acadêmico refinado
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "savefig.dpi": 300,
    "savefig.transparent": False,
    "savefig.bbox": 'tight'
})

def generate_scientific_plots():
    # Definição dos experimentos, cores e estilos de linha
    experiments = [
        ("data_pid.csv", "Simple PID", "#7f7f7f", "--"),
        ("data_cascaded_pid.csv", "Cascaded PID", "#1f77b4", "-"),
        ("data_smc.csv", "SMC", "#2ca02c", "-"),
        ("data_ppo.csv", "PPO Agent", "#d62728", "-"),
        ("data_mpc.csv", "MPC", "#ff7f0e", "-")
    ]
    
    processed_data = []
    for file, label, color, style in experiments:
        try:
            df = pd.read_csv(file)
            
            # Cálculo do esforço de controle usando a nova função trapezoid
            # Se não houver coluna de ação, usa a variação do erro como proxy
            if "action" in df.columns:
                effort = np.trapezoid(np.abs(df["action"]), df["time"])
            else:
                effort = np.trapezoid(np.abs(np.diff(df["error"], prepend=df["error"].iloc[0])), df["time"])
            
            processed_data.append({
                "df": df,
                "label": label,
                "color": color,
                "style": style,
                "ss_error": df["error"].tail(50).mean(),
                "std_dev": df["error"].std(),
                "effort": effort
            })
        except FileNotFoundError:
            print(f"Warning: {file} not found. Skipping...")

    if not processed_data:
        print("Error: No data files found. Run experiments first.")
        return

    labels = [d["label"] for d in processed_data]
    colors = [d["color"] for d in processed_data]

    # --- PLOT 1: Position Error Over Time ---
    plt.figure(figsize=(7, 4))
    for data in processed_data:
        plt.plot(data["df"]["time"], data["df"]["error"], 
                 label=data["label"], color=data["color"], 
                 linestyle=data["style"], linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend(loc='upper right', frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig("fig_error_time.png")
    plt.close()

    # Função auxiliar para gerar gráficos de barras padronizados
    def plot_bar_metric(metric_key, ylabel, filename):
        plt.figure(figsize=(6, 4))
        values = [d[metric_key] for d in processed_data]
        bars = plt.bar(labels, values, color=colors, alpha=0.85, width=0.6, edgecolor='black', linewidth=0.8)
        
        # Adiciona rótulos de dados sobre as barras
        max_val = max(values) if values else 1
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (max_val * 0.02),
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='#333333')
            
        plt.ylabel(ylabel)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # --- PLOT 2: Steady-State Error (Acurácia) ---
    plot_bar_metric("ss_error", "Mean Steady-State Error (m)", "fig_steady_state_error.png")

    # --- PLOT 3: Robustness (Desvio Padrão sob Ondas) ---
    plot_bar_metric("std_dev", "Error Standard Deviation (m)", "fig_error_variability.png")

    # --- PLOT 4: Control Effort (Energia/Atuação) ---
    plot_bar_metric("effort", r"Control Effort ($\int |u| dt$)", "fig_control_effort.png")

    print("Success: 4 improved scientific figures generated.")
    print("- fig_error_time.png")
    print("- fig_steady_state_error.png")
    print("- fig_error_variability.png")
    print("- fig_control_effort.png")

if __name__ == "__main__":
    generate_scientific_plots()