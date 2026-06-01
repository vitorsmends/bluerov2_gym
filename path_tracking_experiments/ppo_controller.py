"""PPO controller for BlueROV2 path tracking."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from base_controller import BaseController
from env_utils import ENV_ID, build_tracking_observation


class PPOController(BaseController):
    name = "ppo"

    def __init__(
        self,
        model_path: str = "ppo_trajectory_final",
        vecnormalize_path: str = "vec_normalize.pkl",
    ):
        self.model_path = self._resolve_file(model_path)
        self.vecnormalize_path = self._resolve_file(vecnormalize_path)

        vec_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode=None)])
        self.vec_env = VecNormalize.load(str(self.vecnormalize_path), vec_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False

        self.model = PPO.load(str(self.model_path))

    @staticmethod
    def _resolve_file(path: str) -> Path:
        candidates = [
            Path(path),
            Path(path + ".zip") if not path.endswith(".zip") else Path(path),
            Path("models") / path,
            Path("models") / (path + ".zip" if not path.endswith(".zip") else path),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find file for path: {path}")

    def get_action(self, obs, state, reference, t):
        virtual_obs = build_tracking_observation(obs, reference)
        norm_obs = self.vec_env.normalize_obs(virtual_obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(action, -40.0, 40.0)
