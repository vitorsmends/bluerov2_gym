from pathlib import Path
import optuna

# SCRIPT_DIR aponta para bluerov2_gym/examples
SCRIPT_DIR = Path(__file__).resolve().parent

# Subimos um nível (.parent.parent) para apontar para a raiz: bluerov2_gym/
BASE_DIR = SCRIPT_DIR.parent

DB_PATH = BASE_DIR / "optuna_results" / "ppo_bluerov_optuna.db"
OUTPUT_DIR = BASE_DIR / "optuna_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def export_best_model_info():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")

    study = optuna.load_study(
        study_name="ppo_bluerov_moderate_optuna",
        storage=f"sqlite:///{DB_PATH}",
    )

    best_trial = study.best_trial

    output_lines = [
        "=== BEST MODEL OPTUNA SUMMARY ===",
        f"Melhor trial: {best_trial.number}",
        f"Melhor valor da funcao objetivo: {best_trial.value}",
        "\nMelhores hiperparametros encontrados:",
    ]

    for key, value in best_trial.params.items():
        output_lines.append(f"  {key}: {value}")

    output_lines.append("=================================")
    output_text = "\n".join(output_lines)

    print(output_text)

    output_file = OUTPUT_DIR / "best_model_summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n[OK] Resumo do melhor modelo salvo em: {output_file}")


def main():
    export_best_model_info()


if __name__ == "__main__":
    main()