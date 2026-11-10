"""Environment and observation utilities for BlueROV2 experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

# Make the project root importable when running scripts directly from this
# directory, e.g. `python -m bluerov2_gym.workflows`.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_ID = "BlueRov-v0"

if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
    )


def make_env(render_mode=None, jonswap_params=None, dynamics_config=None, env_config=None):
    config = dict(env_config or {})
    if jonswap_params is not None:
        config['jonswap'] = jonswap_params
    return gym.make(
        ENV_ID,
        render_mode=render_mode,
        env_config=config or None,
        dynamics_config=dynamics_config,
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


OBSERVATION_ADAPTERS = ('legacy', 'legacy_yaw_error', 'goal_frame')


def body_to_world_rotation(roll, pitch, yaw):
    """R_WB, with the same Euler convention as Dynamics kinematics."""
    cp, sp = np.cos(roll), np.sin(roll)
    ct, st = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
        [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
        [-st, ct*sp, ct*cp],
    ], dtype=float)


def build_tracking_observation(obs: dict, reference: np.ndarray, adapter='legacy') -> dict:
    """Use the 0436ce3 adapter by default; goal_frame is experimental.

    Experimental goal_frame: position in a virtual world anchored at target yaw;
    roll/pitch unchanged, relative yaw, linear velocity error in actual BODY.
    ref[6:9] is WORLD linear velocity. ref[9:12] is current-BODY angular velocity
    (currently zero in all built-in reference generators).

    For a stationary zero pose/reference, the input convention is the training
    convention. A moving reference additionally introduces acceleration and
    rotating-frame effects; this adapter is not a proof of dynamics equivalence.
    """
    if adapter in {'legacy', 'legacy_yaw_error'}:
        virtual = build_legacy_tracking_observation(obs, reference)
        if adapter == 'legacy_yaw_error':
            delta = float(obs['yaw'][0]) - float(reference[5])
            virtual['yaw'] = np.array([wrap_angle(delta)], dtype=np.float32)
        return virtual
    if adapter != 'goal_frame':
        raise ValueError(f'Unknown observation adapter {adapter!r}; use {OBSERVATION_ADAPTERS}.')
    ref = np.asarray(reference, dtype=float).reshape(12)
    if not np.all(np.isfinite(ref)):
        raise ValueError('PPO reference must contain finite values.')
    if not np.allclose(ref[3:5], 0., rtol=0., atol=1e-12):
        raise ValueError('goal_frame supports level targets (reference roll/pitch must be zero).')
    virtual = {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}
    state = obs_to_state(obs).astype(float)
    goal_rotation = body_to_world_rotation(0., 0., ref[5])
    position = goal_rotation.T @ (state[:3] - ref[:3])
    actual_rotation = body_to_world_rotation(*state[3:6])
    velocity = state[6:9] - actual_rotation.T @ ref[6:9]
    angular_velocity = state[9:12] - ref[9:12]
    for keys, values in [(('x', 'y', 'z'), position), (('u', 'v', 'w'), velocity),
                         (('p', 'q', 'r'), angular_velocity)]:
        for key, value in zip(keys, values):
            virtual[key] = np.array([value], dtype=np.float32)
    # Preserve the original zero-reference yaw exactly (including float32's
    # representation of the branch boundary); only wrap a nonzero target.
    if ref[5] != 0.:
        virtual['yaw'] = np.array([wrap_angle(state[5] - ref[5])], dtype=np.float32)
    return virtual


def build_legacy_tracking_observation(obs: dict, reference: np.ndarray) -> dict:
    """Historical BODY position/u-only adapter, kept for explicit reproduction."""
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
