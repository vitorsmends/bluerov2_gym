"""Environment utilities shared by the station-keeping experiments."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


# Allow this folder to be executed either from the project root or directly
# from inside stationkeeping_experiments/ without requiring PYTHONPATH=.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ENV_ID = "BlueRov-v0"


def register_env() -> None:
    """Register the BlueROV Gymnasium environment if needed."""
    try:
        register(
            id=ENV_ID,
            entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
            max_episode_steps=5000,
        )
    except gym.error.Error:
        # The environment may already be registered by the package itself.
        pass


def make_env(render_mode: str | None = None):
    """Create a BlueROV2 Gymnasium environment."""
    register_env()
    return gym.make(ENV_ID, render_mode=render_mode)


def scalar(value) -> float:
    """Convert Gym observation values to plain floats."""
    return float(np.asarray(value).reshape(-1)[0])


def obs_to_state(obs: dict) -> np.ndarray:
    """Convert a Dict observation into the 12-state vector."""
    return np.array(
        [
            scalar(obs["x"]),
            scalar(obs["y"]),
            scalar(obs["z"]),
            scalar(obs["roll"]),
            scalar(obs["pitch"]),
            scalar(obs["yaw"]),
            scalar(obs["u"]),
            scalar(obs["v"]),
            scalar(obs["w"]),
            scalar(obs["p"]),
            scalar(obs["q"]),
            scalar(obs["r"]),
        ],
        dtype=float,
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def compute_errors(state: np.ndarray, reference: np.ndarray) -> dict:
    """Compute common station-keeping errors."""
    position_error = float(np.linalg.norm(state[0:3] - reference[0:3]))
    velocity_error = float(np.linalg.norm(state[6:9] - reference[6:9]))
    yaw_error = wrap_angle(state[5] - reference[5])
    attitude_error = float(np.linalg.norm(state[3:6] - reference[3:6]))

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "yaw_error": yaw_error,
        "attitude_error": attitude_error,
    }


def ensure_dir_for_file(path: str | os.PathLike) -> None:
    """Create the parent folder for a file path."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: str | os.PathLike, header: Iterable[str], rows: list[list]) -> None:
    """Write experiment data to CSV."""
    ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)
