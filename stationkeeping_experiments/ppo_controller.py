"""PPO/DRL controller wrapper for BlueROV2 station keeping."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from base_controller import BaseController
from env_utils import ENV_ID, register_env


class PPOController(BaseController):
    """Wrapper around a trained Stable-Baselines3 PPO policy.

    The model is expected to output direct thruster commands compatible with
    the updated BlueROV2 environment action space.
    """

    def __init__(
        self,
        model_path: str = "bluerov_ppo",
        vecnormalize_path: str | None = "bluerov_vec_normalize.pkl",
        deterministic: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.vecnormalize_path = vecnormalize_path
        self.deterministic = deterministic

        register_env()

        # A VecNormalize object is needed only to apply the same observation
        # normalization used during training.
        self.vec_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode=None)])

        if vecnormalize_path is not None and Path(vecnormalize_path).exists():
            self.vec_env = VecNormalize.load(vecnormalize_path, self.vec_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
        else:
            self.vec_env = None

        self.model = PPO.load(model_path)

    def _state_to_obs(self, state: np.ndarray) -> dict:
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]
        return {key: np.array([state[i]], dtype=np.float32) for i, key in enumerate(keys)}

    def get_action(self, state: np.ndarray, reference: np.ndarray, t: float) -> np.ndarray:
        # For station keeping, PPO sees the error relative to the constant
        # reference. This mirrors the original reward/observation convention:
        # the agent tries to drive the observed error state to zero.
        obs_state = state.copy()
        obs_state[0:3] = state[0:3] - reference[0:3]
        obs_state[5] = np.arctan2(np.sin(state[5] - reference[5]), np.cos(state[5] - reference[5]))
        obs_state[6:9] = state[6:9] - reference[6:9]

        obs = self._state_to_obs(obs_state)

        if self.vec_env is not None:
            obs = self.vec_env.normalize_obs(obs)

        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        return action

    def close(self) -> None:
        if self.vec_env is not None:
            self.vec_env.close()
