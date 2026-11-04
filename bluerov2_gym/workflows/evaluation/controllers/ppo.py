"""PPO controller for BlueROV2 path tracking."""

from __future__ import annotations

import os
import time
import yaml
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .base import BaseController
from ..env_utils import ENV_ID, build_tracking_observation


class PPOController(BaseController):
    name = "ppo"

    def __init__(
        self,
        model_path: str = "ppo_trajectory_final",
        vecnormalize_path: str = "vec_normalize.pkl",
        env_config_path: str = "config/blue_rov_config.yaml",
        dynamics_config_path: str = "config/dynamics_config.yaml",
    ):
        self.model_path = self._resolve_file(model_path)
        self.vecnormalize_path = self._resolve_file(vecnormalize_path)

        # Carrega as configurações estruturadas em arquivos YAML para injetar no novo ambiente
        env_cfg = self._load_yaml_config(env_config_path)
        dyn_cfg = self._load_yaml_config(dynamics_config_path)

        # Instanciação adequada do DummyVecEnv repassando os argumentos esperados pelo novo construtor
        vec_env = DummyVecEnv([
            lambda: gym.make(
                ENV_ID, 
                render_mode=None, 
                env_config=env_cfg, 
                dynamics_config=dyn_cfg
            )
        ])
        
        self.vec_env = VecNormalize.load(str(self.vecnormalize_path), vec_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False

        self.model = PPO.load(str(self.model_path))

        self.last_metrics = self._empty_metrics()

    def reset(self):
        self.last_metrics = self._empty_metrics()

    @staticmethod
    def _empty_metrics():
        return {
            "controller_wall_time_s": 0.0,
            "controller_cpu_time_s": 0.0,
            "controller_frequency_hz": np.nan,
            "controller_prepare_time_s": 0.0,
            "controller_solver_time_s": 0.0,
            "controller_post_time_s": 0.0,
            "controller_success": 1,
        }

    @staticmethod
    def _load_yaml_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

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
        wall_total_start = time.perf_counter()
        cpu_total_start = time.process_time()

        prepare_start = time.perf_counter()

        virtual_obs = build_tracking_observation(obs, reference)
        norm_obs = self.vec_env.normalize_obs(virtual_obs)

        prepare_time = time.perf_counter() - prepare_start

        solver_start = time.perf_counter()

        action, _ = self.model.predict(
            norm_obs,
            deterministic=True,
        )

        solver_time = time.perf_counter() - solver_start

        post_start = time.perf_counter()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        post_time = time.perf_counter() - post_start

        wall_total = time.perf_counter() - wall_total_start
        cpu_total = time.process_time() - cpu_total_start

        self.last_metrics = {
            "controller_wall_time_s": float(wall_total),
            "controller_cpu_time_s": float(cpu_total),
            "controller_frequency_hz": float(1.0 / wall_total) if wall_total > 0.0 else np.nan,
            "controller_prepare_time_s": float(prepare_time),
            "controller_solver_time_s": float(solver_time),
            "controller_post_time_s": float(post_time),
            "controller_success": 1,
        }

        return action