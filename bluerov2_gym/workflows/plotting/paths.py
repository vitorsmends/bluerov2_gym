"""Paths only: importing plotting does not load a controller or data file."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
T200_PATH = Path(__file__).resolve().parent / 'assets' / 't200_performance_16v.csv'
