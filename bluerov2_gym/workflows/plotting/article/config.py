from pathlib import Path
from ..paths import ROOT

SCENARIOS = {
    "Moderate": ROOT / Path("results-moderate/trajectory_experiments"),
    "Severe": ROOT / Path("results-severe/trajectory_experiments"),
}

CONTROLLERS = {
    "PID":  {"prefix": "pid_",  "color": "#4E79A7"},
    "SMC":  {"prefix": "smc_",  "color": "#B07AA1"},
    "NMPC": {"prefix": "nmpc_", "color": "#E15759"},
    "PPO":  {"prefix": "ppo_",  "color": "#59A14F"},
}

OUTPUT_DIR = ROOT / Path("oceans2026/figs")
TABLE_DIR = ROOT / Path("article_tables_simple")

THRUSTER_COLUMNS = [f"T{i}" for i in range(1, 7)]

DPI = 300
FIGSIZE = (6.5, 4.2)
LINE_WIDTH = 1.8
BAND_ALPHA = 0.20
