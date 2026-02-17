import pandas as pd
import matplotlib.pyplot as plt

def generate_comparison():
    plt.figure(figsize=(10, 6))
    
    # Lista de arquivos e configurações de estilo
    experiments = [
        ("data_pid.csv", "PID Simples", "gray", "--"),
        ("data_cascaded_pid.csv", "PID Cascata", "blue", "-"),
        ("data_smc.csv", "Sliding Mode (SMC)", "green", "-"),
        ("data_ppo.csv", "PPO Agent", "red", "-"),
        ("data_mpc.csv", "MPC Controller", "orange", "-")
    ]
    
    for file, label, color, style in experiments:
        try:
            df = pd.read_csv(file)
            plt.plot(df["time"], df["error"], label=label, color=color, linestyle=style, linewidth=1.5)
        except FileNotFoundError:
            print(f"Aviso: Arquivo {file} nao encontrado. Pulando...")

    plt.title("Comparativo de Validacao: Controle de Station Keeping")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Erro de Posicao [m]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("validacao_final_completa.png", dpi=150)
    print("Gráfico gerado: validacao_final_completa.png")

if __name__ == "__main__":
    generate_comparison()