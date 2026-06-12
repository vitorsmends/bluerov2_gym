"""Environment and observation utilities for BlueROV2 experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

# Make the project root importable when running scripts directly from this
# directory, e.g. `python path_tracking_experiments/run_ppo.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_ID = "BlueRov-v0"

try:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
    )
except gym.error.Error:
    # The environment may already be registered by the package.
    pass


def make_env(render_mode=None, jonswap_params=None):
    # CORREÇÃO: O novo construtor não aceita jonswap_params no gym.make.
    # Carregamos as configs padrões do seu novo ambiente se necessário,
    # mas mantemos o gym.make limpo usando apenas os parâmetros suportados.
    return gym.make(
        ENV_ID,
        render_mode=render_mode,
    )

def scalar(value) -> float:
    """Convert scalar-like numpy values to a Python float."""
    return float(np.asarray(value).reshape(-1)[0])


def obs_to_state(obs: dict) -> np.ndarray:
    """Convert a Dict observation into the 12-state vector convention."""
    return np.array(
        [
            scalar(obs["x"]), scalar(obs["y"]), scalar(obs["z"]),
            scalar(obs["roll"]), scalar(obs["pitch"]), scalar(obs["yaw"]),
            scalar(obs["u"]), scalar(obs["v"]), scalar(obs["w"]),
            scalar(obs["p"]), scalar(obs["q"]), scalar(obs["r"]),
        ],
        dtype=np.float32,
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def tracking_errors(state: np.ndarray, reference: np.ndarray) -> dict:
    """Compute core path-tracking errors."""
    position_error = state[0:3] - reference[0:3]
    velocity_error = state[6:9] - reference[6:9]
    yaw_error = wrap_angle(float(state[5] - reference[5]))

    return {
        "position_error_vec": position_error,
        "velocity_error_vec": velocity_error,
        "tracking_error_m": float(np.linalg.norm(position_error)),
        "velocity_error": float(np.linalg.norm(velocity_error)),
        "yaw_error": yaw_error,
    }


def build_tracking_observation(obs: dict, reference: np.ndarray) -> dict:
    """Build the virtual observation used by the PPO trajectory policy.

    This reproduces the same observation transformation used during path-tracking
    PPO training: x, y and z become position errors in the vehicle yaw frame;
    u is replaced by the inertial surge-velocity tracking error; the other
    state channels are kept as in the original observation.
    """
    virtual_obs = {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}

    state = obs_to_state(obs)
    error_pos = state[0:3] - reference[0:3]
    error_vel = state[6:9] - reference[6:9]

    psi = float(state[5])
    c, s = np.cos(psi), np.sin(psi)

    err_x_body = error_pos[0] * c + error_pos[1] * s
    err_y_body = -error_pos[0] * s + error_pos[1] * c
    err_z_body = error_pos[2]

    virtual_obs["x"] = np.array([err_x_body], dtype=np.float32)
    virtual_obs["y"] = np.array([err_y_body], dtype=np.float32)
    virtual_obs["z"] = np.array([err_z_body], dtype=np.float32)

    # Keep the original PPO strategy: only replace u with the surge velocity
    # tracking error, preserving the remaining channels.
    virtual_obs["u"] = np.array([error_vel[0]], dtype=np.float32)

    return virtual_obs