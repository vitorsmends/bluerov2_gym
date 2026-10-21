from __future__ import annotations

import argparse
import os
import platform
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import scipy
import torch
import yaml

from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import __version__ as gymnasium_version
from stable_baselines3 import __version__ as stable_baselines3_version

from tqdm.auto import tqdm

from bluerov2_gym.envs.core.config_utils import deep_merge, load_yaml

DEFAULT_CONFIG = {
    "environment": {
        "id": "BlueRov-v0",
        "entry_point": "bluerov2_gym.envs:BlueRov",
        "max_episode_steps": 1000,
        "render_mode": None,
        "config": {
            "dt": 0.1,
        },
        "dynamics_config": {},
    },
    "paths": {
        "tensorboard_log": "./bluerov_tensorboard/",
        "model_dir": "./models/",
        "model_name": "bluerov_ppo_curriculum",
        "vecnormalize_name": "bluerov_vec_normalize_curriculum.pkl",
    },
    "vecnormalize": {
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
        "clip_reward": 10.0,
        "gamma": 0.99,
    },
    "checkpoint": {
        "save_freq": 100_000,
        "name_prefix": "bluerov_ppo_curriculum_checkpoint",
        "save_vecnormalize": True,
    },
    "ppo": {
        "policy": "MultiInputPolicy",
        "verbose": 1,
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            }
        },
    },
    "training": {
        "tb_log_name": "PPO_BlueROV2_Curriculum",
        "seed": 42,
    },
    "jonswap_defaults": {
        "directional_spread_deg": 25.0,
        "water_depth": 30.0,
        "x_eval": 0.0,
        "y_eval": 0.0,
        "z_eval": -2.0,
        "alpha_wave": 0.02,
        "alpha_noise": 0.30,
        "alpha_js": 0.0081,
        "enable_wave_force": True,
        "wave_force_gain": [35.0, 35.0, 50.0],
        "wave_force_application_point": [0.0, 0.0, 0.05],
        "max_wave_force": 25.0,
        "max_wave_moment": 8.0,
        "seed": 42,
    },
    "initial_jonswap": {
        "Hs": 0.0,
        "Tp": 8.0,
        "gamma": 1.0,
        "N": 32,
        "wave_dir": [1.0, 0.0],
        "scale": 0.0,
        "max_current": 0.0,
        "noise_std": 0.0,
        "directional_spread_deg": 0.0,
        "enable_wave_force": False,
    },
    "curriculum": [
        {
            "name": "No Waves",
            "timesteps": 200_000,
            "jonswap": {
                "Hs": 0.0, "Tp": 8.0, "gamma": 1.0, "N": 32,
                "wave_dir": [1.0, 0.0], "scale": 0.0,
                "max_current": 0.0, "noise_std": 0.0,
                "directional_spread_deg": 0.0,
                "enable_wave_force": False,
            },
        },
        {
            "name": "Calm Sea - Head Waves",
            "timesteps": 400_000,
            "jonswap": {
                "Hs": 1.25, "Tp": 7.03, "gamma": 1.5, "N": 64,
                "wave_dir": [1.0, 0.0], "scale": 0.25,
                "max_current": 0.14, "noise_std": 0.005,
                "directional_spread_deg": 15.0,
                "enable_wave_force": True,
                "max_wave_force": 15.0, "max_wave_moment": 5.0,
            },
        },
        {
            "name": "Moderate Sea - Quartering Waves",
            "timesteps": 600_000,
            "jonswap": {
                "Hs": 2.50, "Tp": 8.0, "gamma": 3.3, "N": 64,
                "wave_dir": [0.707, 0.707], "scale": 0.50,
                "max_current": 0.50, "noise_std": 0.010,
                "directional_spread_deg": 25.0,
                "enable_wave_force": True,
                "max_wave_force": 25.0, "max_wave_moment": 8.0,
            },
        },
        {
            "name": "Severe Sea - Beam Waves",
            "timesteps": 800_000,
            "jonswap": {
                "Hs": 4.50, "Tp": 8.0, "gamma": 3.3, "N": 64,
                "wave_dir": [0.0, 1.0], "scale": 0.80,
                "max_current": 0.80, "noise_std": 0.020,
                "directional_spread_deg": 25.0,
                "enable_wave_force": True,
                "max_wave_force": 35.0, "max_wave_moment": 10.0,
            },
        },
        {
            "name": "Extreme Sea - Following Waves",
            "timesteps": 1_000_000,
            "jonswap": {
                "Hs": 4.50, "Tp": 8.0, "gamma": 5.0, "N": 128,
                "wave_dir": [-1.0, 0.0], "scale": 1.20,
                "max_current": 1.50, "noise_std": 0.030,
                "directional_spread_deg": 35.0,
                "enable_wave_force": True,
                "wave_force_gain": [45.0, 45.0, 65.0],
                "max_wave_force": 40.0, "max_wave_moment": 12.0,
            },
        },
    ],
}


class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, phase_timesteps, phase_name, phase_index, n_phases, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_target = int(total_timesteps)
        self.phase_timesteps_target = int(phase_timesteps)
        self.phase_name = phase_name
        self.phase_index = phase_index
        self.n_phases = n_phases
        self.global_bar = None
        self.phase_bar = None
        self.last_global_n = 0
        self.last_phase_n = 0
        self.phase_start_num_timesteps = 0

    def _on_training_start(self):
        self.phase_start_num_timesteps = self.model.num_timesteps
        self.last_global_n = min(self.model.num_timesteps, self.total_timesteps_target)
        self.last_phase_n = 0
        self.global_bar = tqdm(total=self.total_timesteps_target, initial=self.last_global_n, desc="Total training", unit="steps", dynamic_ncols=True, leave=True)
        self.phase_bar = tqdm(total=self.phase_timesteps_target, initial=0, desc=f"Phase {self.phase_index}/{self.n_phases}: {self.phase_name}", unit="steps", dynamic_ncols=True, leave=False)

    def _on_step(self):
        current_global_n = min(self.model.num_timesteps, self.total_timesteps_target)
        current_phase_n = min(self.model.num_timesteps - self.phase_start_num_timesteps, self.phase_timesteps_target)
        global_delta = current_global_n - self.last_global_n
        phase_delta = current_phase_n - self.last_phase_n
        if global_delta > 0:
            self.global_bar.update(global_delta)
            self.last_global_n = current_global_n
        if phase_delta > 0:
            self.phase_bar.update(phase_delta)
            self.last_phase_n = current_phase_n
        return True

    def _on_training_end(self):
        if self.phase_bar is not None:
            remaining_phase = self.phase_timesteps_target - self.last_phase_n
            if remaining_phase > 0:
                self.phase_bar.update(remaining_phase)
            self.phase_bar.close()
        if self.global_bar is not None:
            self.global_bar.refresh()


def make_env(env_id, render_mode=None, env_config=None, dynamics_config=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode, env_config=env_config, dynamics_config=dynamics_config)
        return Monitor(env)
    return _init


def build_jonswap_params(defaults, overrides):
    params = deep_merge(defaults, overrides)
    tuple_fields = ["wave_dir", "wave_force_gain", "wave_force_application_point"]
    for key in tuple_fields:
        if key in params:
            params[key] = tuple(float(v) for v in params[key])
    params["Hs"] = float(params["Hs"])
    params["Tp"] = float(params["Tp"])
    params["gamma"] = float(params["gamma"])
    params["N"] = int(params["N"])
    return params


def set_curriculum_level(env, jonswap_params):
    base_env = env.envs[0].unwrapped
    base_env.jonswap_params = jonswap_params.copy()
    if hasattr(base_env.dynamics, "set_jonswap_params"):
        base_env.dynamics.set_jonswap_params(**base_env.jonswap_params)
    else:
        base_env.dynamics.reset(jonswap_params=base_env.jonswap_params)


def load_training_config(config_path=None):
    return deep_merge(DEFAULT_CONFIG, load_yaml(config_path))


ABLATIONS = (
    "position_only",
    "no_velocity_penalty",
    "no_yaw_penalty",
    "no_stability_penalty",
    "no_success_bonus",
    "no_curriculum",
    "no_direct_wave_force",
    "no_current",
    "no_noise",
    "learning_rate_1e-4",
    "learning_rate_6e-4",
)


def apply_ablation(config, ablation):
    if ablation is None:
        return
    config.setdefault("experiment", {})["ablation"] = ablation
    if ablation == "position_only":
        config["environment"]["config"]["reward"]["weights"].update({
            "velocity": 0.0,
            "yaw": 0.0,
            "stability": 0.0,
        })
    elif ablation == "no_velocity_penalty":
        config["environment"]["config"]["reward"]["weights"]["velocity"] = 0.0
    elif ablation == "no_yaw_penalty":
        config["environment"]["config"]["reward"]["weights"]["yaw"] = 0.0
    elif ablation == "no_stability_penalty":
        config["environment"]["config"]["reward"]["weights"]["stability"] = 0.0
    elif ablation == "no_success_bonus":
        config["environment"]["config"]["reward"]["success"]["bonus"] = 0.0
    elif ablation == "no_curriculum":
        final_stage = deepcopy(config["curriculum"][-1])
        final_stage["name"] = "Final sea state from start"
        final_stage["timesteps"] = sum(stage["timesteps"] for stage in config["curriculum"])
        config["curriculum"] = [final_stage]
    elif ablation == "no_direct_wave_force":
        for stage in config["curriculum"]:
            stage["jonswap"]["enable_wave_force"] = False
    elif ablation == "no_current":
        for stage in config["curriculum"]:
            stage["jonswap"]["max_current"] = 0.0
    elif ablation == "no_noise":
        for stage in config["curriculum"]:
            stage["jonswap"]["noise_std"] = 0.0
    elif ablation == "learning_rate_1e-4":
        config["ppo"]["learning_rate"] = 1e-4
    elif ablation == "learning_rate_6e-4":
        config["ppo"]["learning_rate"] = 6e-4
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def train_model_with_curriculum(config_path=None, seed=None, separate_run=False, ablation=None):
    config = load_training_config(config_path)
    apply_ablation(config, ablation)

    training_cfg = config["training"]
    run_seed = int(training_cfg.get("seed", 42) if seed is None else seed)
    if run_seed < 0:
        raise ValueError("Training seeds must be non-negative integers.")
    training_cfg["seed"] = run_seed
    config["jonswap_defaults"]["seed"] = run_seed

    dynamics_config = config["environment"].get("dynamics_config", {})
    dynamics_config_file = config["environment"].get("dynamics_config_file")
    if dynamics_config_file:
        dynamics_path = Path(dynamics_config_file)
        if config_path is not None and not dynamics_path.is_absolute():
            dynamics_path = Path(config_path).expanduser().resolve().parent / dynamics_path
        dynamics_config = deep_merge(load_yaml(dynamics_path), dynamics_config)
    env_section = config["environment"]
    paths = config["paths"]

    env_id = env_section["id"]
    if env_id not in gym.envs.registry:
        register(
            id=env_id,
            entry_point=env_section["entry_point"],
            max_episode_steps=int(env_section["max_episode_steps"]),
        )

    log_dir = Path(paths["tensorboard_log"])
    model_dir = Path(paths["model_dir"])
    run_suffix = ""
    if separate_run or ablation:
        run_suffix = f"_ablation_{ablation}" if ablation else ""
        run_suffix += f"_seed_{run_seed}"
    model_path = model_dir / f"{paths['model_name']}{run_suffix}"
    vecnorm_name = paths["vecnormalize_name"]
    vecnorm_path = model_dir / (
        f"{vecnorm_name.removesuffix('.pkl')}{run_suffix}.pkl"
    )
    run_name = f"{training_cfg['tb_log_name']}{run_suffix}"
    manifest_path = model_dir / f"{model_path.name}_config.yaml"
    if separate_run or ablation:
        existing_logs = [
            path for path in log_dir.glob(f"{run_name}*")
            if path.is_dir() and list(path.glob("events.out.tfevents.*"))
        ]
        checkpoint_glob = (
            f"{config['checkpoint']['name_prefix']}{run_suffix}_*_steps.zip"
        )
        existing_outputs = [
            path for path in (
                Path(f"{model_path}.zip"),
                vecnorm_path,
                manifest_path,
                *model_dir.glob(checkpoint_glob),
            )
            if path.exists()
            and not (path == manifest_path and path.stat().st_size == 0)
        ]
        if existing_logs or existing_outputs:
            raise FileExistsError(
                f"Outputs already exist for training run '{run_name}'. "
                "Use a new seed or archive the existing run before retrying."
            )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    manifest = {
        "python_version": str(platform.python_version()),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "torch_version": str(torch.__version__),
        "gymnasium_version": str(gymnasium_version),
        "stable_baselines3_version": str(stable_baselines3_version),
        "seed": run_seed,
        "config": config,
    }
    temporary_manifest_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with temporary_manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    os.replace(temporary_manifest_path, manifest_path)

    initial_jonswap = build_jonswap_params(config["jonswap_defaults"], config["initial_jonswap"])
    env_config = deep_merge(env_section.get("config", {}), {"jonswap": initial_jonswap})

    env = DummyVecEnv([
        make_env(
            env_id=env_id,
            render_mode=env_section.get("render_mode"),
            env_config=env_config,
            dynamics_config=dynamics_config,
        )
    ])
    env = VecNormalize(env, **config["vecnormalize"])

    checkpoint_cfg = config["checkpoint"]
    checkpoint_callback = CheckpointCallback(
        save_freq=int(checkpoint_cfg["save_freq"]),
        save_path=str(model_dir),
        name_prefix=f"{checkpoint_cfg['name_prefix']}{run_suffix}",
        save_vecnormalize=bool(checkpoint_cfg["save_vecnormalize"]),
    )

    ppo_cfg = dict(config["ppo"])
    policy = ppo_cfg.pop("policy")
    verbose = ppo_cfg.pop("verbose")
    model = PPO(
        policy=policy,
        env=env,
        verbose=verbose,
        tensorboard_log=str(log_dir),
        seed=run_seed,
        **ppo_cfg,
    )

    lessons = []
    for lesson_cfg in config["curriculum"]:
        lessons.append({
            "name": lesson_cfg["name"],
            "timesteps": int(lesson_cfg["timesteps"]),
            "jonswap": build_jonswap_params(config["jonswap_defaults"], lesson_cfg["jonswap"]),
        })

    total_curriculum_timesteps = sum(lesson["timesteps"] for lesson in lessons)
    print("Iniciando treinamento estruturado por currículo...")
    print(f"Semente de treinamento: {run_seed}")
    print(f"Total de timesteps planejado: {total_curriculum_timesteps:,}")

    is_first_iteration = True
    for index, lesson in enumerate(lessons):
        jonswap_params = lesson["jonswap"]
        print(f"\n[CURRICULUM] Etapa {index + 1}/{len(lessons)} | {lesson['name']}")
        print(
            "[CURRICULUM] "
            f"Hs={jonswap_params['Hs']} m | Tp={jonswap_params['Tp']} s | "
            f"gamma={jonswap_params['gamma']} | N={jonswap_params['N']} | "
            f"max_current={jonswap_params['max_current']} m/s | "
            f"scale={jonswap_params['scale']} | "
            f"wave_force={jonswap_params['enable_wave_force']}"
        )

        set_curriculum_level(env, jonswap_params)
        env.reset()
        progress_callback = TrainingProgressCallback(
            total_timesteps=total_curriculum_timesteps,
            phase_timesteps=lesson["timesteps"],
            phase_name=lesson["name"],
            phase_index=index + 1,
            n_phases=len(lessons),
        )
        callback = CallbackList([checkpoint_callback, progress_callback])
        model.learn(
            total_timesteps=lesson["timesteps"],
            callback=callback,
            tb_log_name=run_name,
            reset_num_timesteps=is_first_iteration,
        )
        is_first_iteration = False

    print("Treinamento por currículo finalizado com sucesso.")
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    env.close()
    print(f"Modelo salvo em: {model_path}")
    print(f"VecNormalize salvo em: {vecnorm_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO with a configurable curriculum.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML configuration file.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run an independent training for each seed, saving separate logs and models.",
    )
    parser.add_argument(
        "--ablation",
        choices=ABLATIONS,
        help="Apply a predefined one-factor ablation while preserving the configured timestep budget.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seeds:
        if any(seed < 0 for seed in args.seeds):
            raise ValueError("--seeds values must be non-negative integers.")
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("--seeds must contain unique seed values.")
        for run_seed in args.seeds:
            train_model_with_curriculum(
                config_path=args.config,
                seed=run_seed,
                separate_run=True,
                ablation=args.ablation,
            )
    else:
        train_model_with_curriculum(config_path=args.config, ablation=args.ablation)
