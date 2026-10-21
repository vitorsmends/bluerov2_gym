from pathlib import Path
import os

RESULTS_ROOT = Path(os.environ.get("BLUEROV_RESULTS_ROOT", "training_runs/ppo_physics_v2_seed42/evaluation"))

SCENARIOS = {
    "Moderate": RESULTS_ROOT / "moderate/trajectory_experiments",
    "Severe": RESULTS_ROOT / "severe/trajectory_experiments",
}

CONTROLLERS = {
    "PID":  {"prefix": "pid_",  "color": "#4E79A7"},
    "SMC":  {"prefix": "smc_",  "color": "#B07AA1"},
    "NMPC": {"prefix": "nmpc_", "color": "#E15759"},
    "PPO":  {"prefix": "ppo_",  "color": "#59A14F"},
}

OUTPUT_DIR = Path(os.environ.get("BLUEROV_FIGURES_ROOT", str(RESULTS_ROOT / "figures")))
TABLE_DIR = RESULTS_ROOT / "tables"

THRUSTER_COLUMNS = [f"T{i}" for i in range(1, 7)]

DPI = 300
FIGSIZE = (6.5, 4.2)
LINE_WIDTH = 1.8
BAND_ALPHA = 0.20
