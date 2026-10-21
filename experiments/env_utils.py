"""Environment and observation utilities for BlueROV2 experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

# Make the project root importable when running scripts directly from this
# directory, e.g. `python experiments/run_ppo.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_ID = "BlueRov-v0"

if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
    )


def make_env(render_mode=None, jonswap_params=None):
    from bluerov2_gym.envs.core.config_utils import load_yaml

    env_config = load_yaml(PROJECT_ROOT / "config/blue_rov_config.yaml")
    if jonswap_params is not None:
        env_config["jonswap"] = jonswap_params
    return gym.make(
        ENV_ID,
        render_mode=render_mode,
        env_config=env_config,
        dynamics_config=load_yaml(PROJECT_ROOT / "config/dynamics_config.yaml"),
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
    velocity_error = reference_velocity_error_body(state, reference)
    yaw_error = wrap_angle(float(state[5] - reference[5]))

    return {
        "position_error_vec": position_error,
        "velocity_error_vec": velocity_error,
        "tracking_error_m": float(np.linalg.norm(position_error)),
        "velocity_error": float(np.linalg.norm(velocity_error)),
        "yaw_error": yaw_error,
    }


def world_to_body_rotation(state):
    from bluerov2_gym.envs.core.dynamics import Dynamics
    return Dynamics.body_to_world_rotation(state[:6]).T


def reference_velocity_error_body(state, reference):
    """Vehicle minus reference velocity, both expressed in current body axes."""
    return state[6:9] - world_to_body_rotation(state) @ reference[6:9]


def build_tracking_observation(obs: dict, reference: np.ndarray) -> dict:
    """Build the virtual observation used by the PPO trajectory policy.

    Position errors retain the legacy yaw-frame transform. Only u is replaced
    by the surge error after transforming the inertial reference velocity to
    the current body frame. Other channels retain the fixed-policy convention.
    """
    virtual_obs = {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}

    state = obs_to_state(obs)
    error_pos = state[0:3] - reference[0:3]
    error_vel = reference_velocity_error_body(state, reference)

    psi = float(state[5])
    c, s = np.cos(psi), np.sin(psi)

    err_x_body = error_pos[0] * c + error_pos[1] * s
    err_y_body = -error_pos[0] * s + error_pos[1] * c
    err_z_body = error_pos[2]

    virtual_obs["x"] = np.array([err_x_body], dtype=np.float32)
    virtual_obs["y"] = np.array([err_y_body], dtype=np.float32)
    virtual_obs["z"] = np.array([err_z_body], dtype=np.float32)

    # Keep the original PPO strategy: only replace u with the surge velocity
    # tracking error in the body frame, preserving the remaining channels.
    virtual_obs["u"] = np.array([error_vel[0]], dtype=np.float32)

    return virtual_obs
