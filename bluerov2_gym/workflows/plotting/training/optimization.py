from pathlib import Path
import optuna

# Artifact locations are independent of the removed script directories.
BASE_DIR = Path(__file__).resolve().parents[4]

DB_PATH = BASE_DIR / "optuna_results" / "ppo_bluerov_optuna.db"
OUTPUT_DIR = BASE_DIR / "optuna_plots"



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


def main(settings=None):
    global OUTPUT_DIR, DB_PATH
    settings = settings or {}
    OUTPUT_DIR = Path(settings.get('output_dir', OUTPUT_DIR))
    DB_PATH = Path(settings.get('database', DB_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_best_model_info()


if __name__ == "__main__":
    main()