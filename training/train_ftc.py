from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tqdm.auto import tqdm

from bluerov2_gym.envs.core.actuator_faults import FaultFactory
from bluerov2_gym.envs.core.config_utils import deep_merge, load_yaml
from bluerov2_gym.training.fault_sampler import sample_faults_for_episode
import bluerov2_gym


class TrainingProgressCallback(BaseCallback):
    """
    Display progress for the complete FTC curriculum and for the current phase.
    """

    def __init__(
        self,
        total_timesteps: int,
        phase_timesteps: int,
        phase_name: str,
        phase_index: int,
        n_phases: int,
        initial_model_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.total_timesteps_target = int(total_timesteps)
        self.phase_timesteps_target = int(phase_timesteps)
        self.phase_name = str(phase_name)
        self.phase_index = int(phase_index)
        self.n_phases = int(n_phases)
        self.initial_model_timesteps = int(initial_model_timesteps)

        self.global_bar = None
        self.phase_bar = None

        self.phase_start_num_timesteps = 0
        self.last_global_n = 0
        self.last_phase_n = 0

    def _on_training_start(self) -> None:
        self.phase_start_num_timesteps = int(self.model.num_timesteps)

        self.last_global_n = max(
            0,
            min(
                int(self.model.num_timesteps)
                - self.initial_model_timesteps,
                self.total_timesteps_target,
            ),
        )
        self.last_phase_n = 0

        self.global_bar = tqdm(
            total=self.total_timesteps_target,
            initial=self.last_global_n,
            desc="FTC curriculum",
            unit="steps",
            dynamic_ncols=True,
            leave=True,
        )

        self.phase_bar = tqdm(
            total=self.phase_timesteps_target,
            initial=0,
            desc=(
                f"Phase {self.phase_index}/{self.n_phases}: "
                f"{self.phase_name}"
            ),
            unit="steps",
            dynamic_ncols=True,
            leave=False,
        )

    def _on_step(self) -> bool:
        current_global_n = max(
            0,
            min(
                int(self.model.num_timesteps)
                - self.initial_model_timesteps,
                self.total_timesteps_target,
            ),
        )

        current_phase_n = max(
            0,
            min(
                int(self.model.num_timesteps)
                - self.phase_start_num_timesteps,
                self.phase_timesteps_target,
            ),
        )

        global_delta = current_global_n - self.last_global_n
        phase_delta = current_phase_n - self.last_phase_n

        if global_delta > 0 and self.global_bar is not None:
            self.global_bar.update(global_delta)
            self.last_global_n = current_global_n

        if phase_delta > 0 and self.phase_bar is not None:
            self.phase_bar.update(phase_delta)
            self.last_phase_n = current_phase_n

        return True

    def _on_training_end(self) -> None:
        if self.phase_bar is not None:
            self.phase_bar.close()

        if self.global_bar is not None:
            self.global_bar.close()


class FaultCurriculumCallback(BaseCallback):
    """
    Sample and install actuator faults for one curriculum phase.

    The JONSWAP condition is not modified here. It remains fixed at the severe
    condition throughout FTC fine-tuning.

    A new fault realization is sampled:
      1. when the phase starts;
      2. whenever an episode terminates.

    DummyVecEnv automatically resets an environment before callbacks receive
    the ``done`` signal. Installing a newly created FaultManager after that
    automatic reset is still valid because the new manager has a clean internal
    state and is applied before the first action of the following episode.
    """

    def __init__(
        self,
        stage_cfg: dict,
        phase_name: str,
        seed: int,
        log_every_episodes: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.stage_cfg = dict(stage_cfg or {})
        self.phase_name = str(phase_name)
        self.rng = np.random.default_rng(int(seed))
        self.log_every_episodes = max(1, int(log_every_episodes))

        self.episode_count = 0
        self.last_fault_description = "nominal"

    def _on_training_start(self) -> None:
        self._sample_and_install_fault()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        if dones is not None and bool(np.any(dones)):
            finished_episodes = int(np.count_nonzero(dones))
            self.episode_count += finished_episodes

            self._sample_and_install_fault()

            if self.episode_count % self.log_every_episodes == 0:
                print(
                    f"[FTC] Phase={self.phase_name} | "
                    f"episodes={self.episode_count} | "
                    f"next_fault={self.last_fault_description}"
                )

        return True

    def _sample_and_install_fault(self) -> None:
        base_env = get_base_env(self.training_env)
        n_thrusters = int(base_env.action_space.shape[0])

        sampled_faults = sample_faults_for_episode(
            stage_cfg=self.stage_cfg,
            n_thrusters=n_thrusters,
            rng=self.rng,
        )

        if sampled_faults:
            fault_manager = FaultFactory.build_manager(sampled_faults)
            base_env.set_fault_manager(fault_manager)
        elif hasattr(base_env, "clear_fault_manager"):
            base_env.clear_fault_manager()
        else:
            base_env.set_fault_manager(None)

        self.last_fault_description = describe_faults(sampled_faults)

        self.logger.record(
            "ftc/number_of_active_faults",
            len(sampled_faults),
        )


def make_env(
    env_id: str,
    render_mode=None,
    env_config: dict | None = None,
    dynamics_config: dict | None = None,
):
    def _init():
        env = gym.make(
            env_id,
            render_mode=render_mode,
            env_config=env_config,
            dynamics_config=dynamics_config,
        )
        return Monitor(env)

    return _init


def build_jonswap_params(
    defaults: dict,
    overrides: dict,
) -> dict:
    params = deep_merge(defaults, overrides)

    tuple_fields = [
        "wave_dir",
        "wave_force_gain",
        "wave_force_application_point",
    ]

    for key in tuple_fields:
        if key in params:
            params[key] = tuple(float(v) for v in params[key])

    params["Hs"] = float(params["Hs"])
    params["Tp"] = float(params["Tp"])
    params["gamma"] = float(params["gamma"])
    params["N"] = int(params["N"])

    if "max_current" in params:
        params["max_current"] = float(params["max_current"])

    if "scale" in params:
        params["scale"] = float(params["scale"])

    if "enable_wave_force" in params:
        params["enable_wave_force"] = bool(
            params["enable_wave_force"]
        )

    return params


def get_base_env(vec_env):
    """
    Access BlueRov through VecNormalize, DummyVecEnv and Monitor.
    """
    current_env = vec_env

    if isinstance(current_env, VecNormalize):
        current_env = current_env.venv

    if not hasattr(current_env, "envs"):
        raise TypeError(
            "Expected a VecEnv exposing the 'envs' attribute."
        )

    if len(current_env.envs) != 1:
        raise ValueError(
            "This training script currently expects exactly one environment."
        )

    return current_env.envs[0].unwrapped


def set_fixed_jonswap(
    vec_env,
    jonswap_params: dict,
) -> None:
    """
    Install the same severe JONSWAP parameters for the complete FTC training.
    """
    base_env = get_base_env(vec_env)
    base_env.jonswap_params = jonswap_params.copy()

    if hasattr(base_env.dynamics, "set_jonswap_params"):
        base_env.dynamics.set_jonswap_params(
            **base_env.jonswap_params
        )
    else:
        base_env.dynamics.reset(
            jonswap_params=base_env.jonswap_params
        )



def describe_faults(
    faults_cfg: list[dict] | None,
) -> str:
    if not faults_cfg:
        return "nominal"

    descriptions = []

    for fault_cfg in faults_cfg:
        descriptions.append(
            f"{fault_cfg['type']}"
            f"(thruster={fault_cfg['thruster']}, "
            f"params={fault_cfg.get('params', {})})"
        )

    return "; ".join(descriptions)


def describe_fault_stage(stage_cfg: dict) -> str:
    candidates = stage_cfg.get("candidates", [])

    if not candidates:
        return "nominal"

    candidate_names = [
        str(candidate["type"])
        for candidate in candidates
    ]

    return (
        f"activation_probability="
        f"{stage_cfg.get('activation_probability', 1.0)}, "
        f"number_of_faults="
        f"{stage_cfg.get('number_of_faults', 1)}, "
        f"candidates={candidate_names}"
    )


def load_training_config(config_path=None) -> tuple[dict, Path]:

    if config_path is None:
        resolved_config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "ppo_ftc_curriculum.yaml"
        )
    else:
        resolved_config_path = (
            Path(config_path)
            .expanduser()
            .resolve()
        )

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration file not found: {resolved_config_path}"
        )

    config = load_yaml(resolved_config_path)

    if not isinstance(config, dict):
        raise ValueError(
            f"Training configuration must be a YAML mapping: {resolved_config_path}"
        )

    return config, resolved_config_path


def resolve_relative_path(
    path_value: str,
    config_path: Path,
) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return (config_path.parent / path).resolve()


def model_file_exists(model_path: Path) -> bool:
    if model_path.is_file():
        return True

    if model_path.suffix != ".zip":
        return model_path.with_suffix(".zip").is_file()

    return False


def validate_environment_interface(vec_env) -> None:
    base_env = get_base_env(vec_env)

    if not hasattr(base_env, "set_fault_manager"):
        raise AttributeError(
            "BlueRov must expose set_fault_manager(fault_manager)."
        )

    if not hasattr(base_env, "clear_fault_manager"):
        print(
            "[FTC] clear_fault_manager() is unavailable; "
            "set_fault_manager(None) will be used to clear faults."
        )

    if not hasattr(base_env, "fault_manager"):
        raise AttributeError(
            "BlueRov must expose the fault_manager attribute."
        )


def train_ftc_with_curriculum(
    config_path=None,
) -> None:
    config, resolved_config_path = load_training_config(
        config_path
    )

    env_section = config["environment"]
    paths = config["paths"]
    pretrained = config["pretrained"]
    training_cfg = config.get("training", {})

    dynamics_config = env_section.get(
        "dynamics_config",
        {},
    )
    dynamics_config_file = env_section.get(
        "dynamics_config_file"
    )

    if dynamics_config_file:
        dynamics_path = resolve_relative_path(
            dynamics_config_file,
            resolved_config_path,
        )

        dynamics_config = deep_merge(
            load_yaml(dynamics_path),
            dynamics_config,
        )

    env_id = env_section["id"]

    if env_id not in gym.envs.registry:
        register(
            id=env_id,
            entry_point=env_section["entry_point"],
            max_episode_steps=int(
                env_section["max_episode_steps"]
            ),
        )

    log_dir = resolve_relative_path(
        paths["tensorboard_log"],
        resolved_config_path,
    )
    model_dir = resolve_relative_path(
        paths["model_dir"],
        resolved_config_path,
    )

    model_path = model_dir / paths["model_name"]
    vecnorm_path = model_dir / paths["vecnormalize_name"]

    pretrained_model_path = resolve_relative_path(
        pretrained["model_path"],
        resolved_config_path,
    )
    pretrained_vecnorm_path = resolve_relative_path(
        pretrained["vecnormalize_path"],
        resolved_config_path,
    )

    if not model_file_exists(pretrained_model_path):
        raise FileNotFoundError(
            "Pretrained PPO model not found: "
            f"{pretrained_model_path}"
        )

    if not pretrained_vecnorm_path.is_file():
        raise FileNotFoundError(
            "Pretrained VecNormalize not found: "
            f"{pretrained_vecnorm_path}"
        )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    severe_jonswap = build_jonswap_params(
        config["jonswap_defaults"],
        config["severe_jonswap"],
    )

    env_config = deep_merge(
        env_section.get("config", {}),
        {"jonswap": severe_jonswap},
    )

    dummy_env = DummyVecEnv(
        [
            make_env(
                env_id=env_id,
                render_mode=env_section.get(
                    "render_mode"
                ),
                env_config=env_config,
                dynamics_config=dynamics_config,
            )
        ]
    )

    env = VecNormalize.load(
        str(pretrained_vecnorm_path),
        dummy_env,
    )

    vecnorm_cfg = config.get(
        "vecnormalize_finetuning",
        {},
    )

    env.training = bool(
        vecnorm_cfg.get(
            "update_statistics",
            True,
        )
    )
    env.norm_reward = bool(
        vecnorm_cfg.get(
            "norm_reward",
            True,
        )
    )

    validate_environment_interface(env)
    set_fixed_jonswap(
        env,
        severe_jonswap,
    )

    model = PPO.load(
        str(pretrained_model_path),
        env=env,
        tensorboard_log=str(log_dir),
        device=training_cfg.get(
            "device",
            "auto",
        ),
    )

    checkpoint_cfg = config["checkpoint"]

    checkpoint_callback = CheckpointCallback(
        save_freq=int(
            checkpoint_cfg["save_freq"]
        ),
        save_path=str(model_dir),
        name_prefix=checkpoint_cfg["name_prefix"],
        save_vecnormalize=bool(
            checkpoint_cfg.get(
                "save_vecnormalize",
                True,
            )
        ),
    )

    lessons: list[dict[str, Any]] = []

    for lesson_cfg in config["curriculum"]:
        lessons.append(
            {
                "name": str(lesson_cfg["name"]),
                "timesteps": int(
                    lesson_cfg["timesteps"]
                ),
                "fault_stage": dict(
                    lesson_cfg.get(
                        "fault_stage",
                        {},
                    )
                ),
            }
        )

    if not lessons:
        raise ValueError(
            "The FTC curriculum cannot be empty."
        )

    total_curriculum_timesteps = sum(
        lesson["timesteps"]
        for lesson in lessons
    )

    initial_model_timesteps = int(
        model.num_timesteps
    )
    base_seed = int(
        training_cfg.get(
            "seed",
            42,
        )
    )

    print(
        "Starting FTC curriculum fine-tuning..."
    )
    print(
        "The JONSWAP severe condition will remain "
        "fixed throughout training."
    )
    print(
        "[SEVERE JONSWAP] "
        f"Hs={severe_jonswap['Hs']} m | "
        f"Tp={severe_jonswap['Tp']} s | "
        f"gamma={severe_jonswap['gamma']} | "
        f"N={severe_jonswap['N']} | "
        f"max_current="
        f"{severe_jonswap.get('max_current')} m/s | "
        f"wave_force="
        f"{severe_jonswap.get('enable_wave_force')}"
    )
    print(
        f"Loaded nominal PPO with "
        f"{initial_model_timesteps:,} accumulated timesteps."
    )
    print(
        f"Additional FTC timesteps: "
        f"{total_curriculum_timesteps:,}"
    )

    for index, lesson in enumerate(lessons):
        phase_number = index + 1

        print(
            f"\n[FTC CURRICULUM] Phase "
            f"{phase_number}/{len(lessons)} | "
            f"{lesson['name']}"
        )
        print(
            "[FTC CURRICULUM] Distribution: "
            f"{describe_fault_stage(lesson['fault_stage'])}"
        )

        # Clear the previous phase fault before resetting the environment.
        base_env = get_base_env(env)
        if hasattr(base_env, "clear_fault_manager"):
            base_env.clear_fault_manager()
        else:
            base_env.set_fault_manager(None)

        # The environmental condition is explicitly reapplied but never changed.
        set_fixed_jonswap(
            env,
            severe_jonswap,
        )
        env.reset()

        fault_callback = FaultCurriculumCallback(
            stage_cfg=lesson["fault_stage"],
            phase_name=lesson["name"],
            seed=base_seed + index,
            log_every_episodes=int(
                training_cfg.get(
                    "fault_log_every_episodes",
                    100,
                )
            ),
        )

        progress_callback = TrainingProgressCallback(
            total_timesteps=total_curriculum_timesteps,
            phase_timesteps=lesson["timesteps"],
            phase_name=lesson["name"],
            phase_index=phase_number,
            n_phases=len(lessons),
            initial_model_timesteps=initial_model_timesteps,
        )

        callback = CallbackList(
            [
                checkpoint_callback,
                fault_callback,
                progress_callback,
            ]
        )

        model.learn(
            total_timesteps=lesson["timesteps"],
            callback=callback,
            tb_log_name=training_cfg[
                "tb_log_name"
            ],
            reset_num_timesteps=False,
        )

        phase_model_path = (
            model_dir
            / f"{paths['model_name']}_phase_{phase_number}"
        )
        phase_vecnorm_path = (
            model_dir
            / (
                f"{Path(paths['vecnormalize_name']).stem}"
                f"_phase_{phase_number}.pkl"
            )
        )

        model.save(
            str(phase_model_path)
        )
        env.save(
            str(phase_vecnorm_path)
        )

        print(
            f"[FTC CURRICULUM] Phase {phase_number} saved."
        )

    final_base_env = get_base_env(env)
    if hasattr(final_base_env, "clear_fault_manager"):
        final_base_env.clear_fault_manager()
    else:
        final_base_env.set_fault_manager(None)

    model.save(
        str(model_path)
    )
    env.save(
        str(vecnorm_path)
    )
    env.close()

    print(
        "FTC curriculum fine-tuning completed."
    )
    print(
        f"FTC model saved to: {model_path}.zip"
    )
    print(
        f"VecNormalize saved to: {vecnorm_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing PPO policy under a fixed severe "
            "sea state using an actuator-fault curriculum."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML configuration file. "
            "Defaults to config/ppo_ftc_curriculum.yaml."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_ftc_with_curriculum(
        config_path=args.config
    )
