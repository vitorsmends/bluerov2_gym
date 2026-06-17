import os

import gymnasium as gym
import numpy as np

from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tqdm.auto import tqdm


ENV_ID = "BlueRov-v0"

LOG_DIR = "./bluerov_tensorboard/"
MODEL_DIR = "./models/"
MODEL_PATH = os.path.join(MODEL_DIR, "bluerov_ppo_curriculum")
VECNORM_PATH = os.path.join(
    MODEL_DIR,
    "bluerov_vec_normalize_curriculum.pkl",
)


if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs:BlueRov",
        max_episode_steps=1000,
    )


class TrainingProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        phase_timesteps: int,
        phase_name: str,
        phase_index: int,
        n_phases: int,
        verbose: int = 0,
    ):
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

    def _on_training_start(self) -> None:
        self.phase_start_num_timesteps = self.model.num_timesteps

        self.last_global_n = min(
            self.model.num_timesteps,
            self.total_timesteps_target,
        )

        self.last_phase_n = 0

        self.global_bar = tqdm(
            total=self.total_timesteps_target,
            initial=self.last_global_n,
            desc="Total training",
            unit="steps",
            dynamic_ncols=True,
            leave=True,
        )

        self.phase_bar = tqdm(
            total=self.phase_timesteps_target,
            initial=0,
            desc=f"Phase {self.phase_index}/{self.n_phases}: {self.phase_name}",
            unit="steps",
            dynamic_ncols=True,
            leave=False,
        )

    def _on_step(self) -> bool:
        current_global_n = min(
            self.model.num_timesteps,
            self.total_timesteps_target,
        )

        current_phase_n = min(
            self.model.num_timesteps - self.phase_start_num_timesteps,
            self.phase_timesteps_target,
        )

        global_delta = current_global_n - self.last_global_n
        phase_delta = current_phase_n - self.last_phase_n

        if global_delta > 0:
            self.global_bar.update(global_delta)
            self.last_global_n = current_global_n

        if phase_delta > 0:
            self.phase_bar.update(phase_delta)
            self.last_phase_n = current_phase_n

        return True

    def _on_training_end(self) -> None:
        if self.phase_bar is not None:
            remaining_phase = self.phase_timesteps_target - self.last_phase_n
            if remaining_phase > 0:
                self.phase_bar.update(remaining_phase)
            self.phase_bar.close()

        if self.global_bar is not None:
            self.global_bar.refresh()


def make_env(render_mode=None, env_config=None):
    def _init():
        env = gym.make(
            ENV_ID,
            render_mode=render_mode,
            env_config=env_config,
        )
        env = Monitor(env)
        return env

    return _init


def build_jonswap_params(
    *,
    Hs,
    Tp,
    gamma,
    N,
    wave_dir,
    scale,
    max_current,
    noise_std,
    directional_spread_deg=25.0,
    water_depth=30.0,
    z_eval=-2.0,
    alpha_wave=0.02,
    alpha_noise=0.30,
    alpha_js=0.0081,
    enable_wave_force=True,
    wave_force_gain=(35.0, 35.0, 50.0),
    wave_force_application_point=(0.0, 0.0, 0.05),
    max_wave_force=25.0,
    max_wave_moment=8.0,
    seed=42,
):
    return {
        "Hs": float(Hs),
        "Tp": float(Tp),
        "gamma": float(gamma),
        "N": int(N),
        "wave_dir": tuple(float(v) for v in wave_dir),
        "directional_spread_deg": float(directional_spread_deg),
        "water_depth": float(water_depth),
        "x_eval": 0.0,
        "y_eval": 0.0,
        "z_eval": float(z_eval),
        "scale": float(scale),
        "max_current": float(max_current),
        "alpha_wave": float(alpha_wave),
        "noise_std": float(noise_std),
        "alpha_noise": float(alpha_noise),
        "alpha_js": float(alpha_js),
        "enable_wave_force": bool(enable_wave_force),
        "wave_force_gain": tuple(float(v) for v in wave_force_gain),
        "wave_force_application_point": tuple(
            float(v) for v in wave_force_application_point
        ),
        "max_wave_force": float(max_wave_force),
        "max_wave_moment": float(max_wave_moment),
        "seed": int(seed),
    }


def set_curriculum_level(env, jonswap_params):
    base_env = env.envs[0].unwrapped

    base_env.jonswap_params = jonswap_params.copy()

    if hasattr(base_env.dynamics, "set_jonswap_params"):
        base_env.dynamics.set_jonswap_params(**base_env.jonswap_params)
    else:
        base_env.dynamics.reset(jonswap_params=base_env.jonswap_params)


def train_model_with_curriculum():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    initial_jonswap = build_jonswap_params(
        Hs=0.0,
        Tp=8.0,
        gamma=1.0,
        N=32,
        wave_dir=(1.0, 0.0),
        scale=0.0,
        max_current=0.0,
        noise_std=0.0,
        directional_spread_deg=0.0,
        enable_wave_force=False,
    )

    env_config = {
        "dt": 0.1,
        "jonswap": initial_jonswap,
    }

    env = DummyVecEnv(
        [
            make_env(
                render_mode=None,
                env_config=env_config,
            )
        ]
    )

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="bluerov_ppo_curriculum_checkpoint",
        save_vecnormalize=True,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
    )

    lessons = [
        {
            "name": "No Waves",
            "timesteps": 200_000,
            "jonswap": build_jonswap_params(
                Hs=0.0,
                Tp=8.0,
                gamma=1.0,
                N=32,
                wave_dir=(1.0, 0.0),
                scale=0.0,
                max_current=0.0,
                noise_std=0.0,
                directional_spread_deg=0.0,
                enable_wave_force=False,
            ),
        },
        {
            "name": "Calm Sea - Head Waves",
            "timesteps": 400_000,
            "jonswap": build_jonswap_params(
                Hs=1.25,
                Tp=7.03,
                gamma=1.5,
                N=64,
                wave_dir=(1.0, 0.0),
                scale=0.25,
                max_current=0.14,
                noise_std=0.005,
                directional_spread_deg=15.0,
                enable_wave_force=True,
                max_wave_force=15.0,
                max_wave_moment=5.0,
            ),
        },
        {
            "name": "Moderate Sea - Quartering Waves",
            "timesteps": 600_000,
            "jonswap": build_jonswap_params(
                Hs=2.50,
                Tp=8.0,
                gamma=3.3,
                N=64,
                wave_dir=(0.707, 0.707),
                scale=0.50,
                max_current=0.50,
                noise_std=0.010,
                directional_spread_deg=25.0,
                enable_wave_force=True,
                max_wave_force=25.0,
                max_wave_moment=8.0,
            ),
        },
        {
            "name": "Severe Sea - Beam Waves",
            "timesteps": 800_000,
            "jonswap": build_jonswap_params(
                Hs=4.50,
                Tp=8.0,
                gamma=3.3,
                N=64,
                wave_dir=(0.0, 1.0),
                scale=0.80,
                max_current=0.80,
                noise_std=0.020,
                directional_spread_deg=25.0,
                enable_wave_force=True,
                max_wave_force=35.0,
                max_wave_moment=10.0,
            ),
        },
        {
            "name": "Extreme Sea - Following Waves",
            "timesteps": 1_000_000,
            "jonswap": build_jonswap_params(
                Hs=4.50,
                Tp=8.0,
                gamma=5.0,
                N=128,
                wave_dir=(-1.0, 0.0),
                scale=1.20,
                max_current=1.50,
                noise_std=0.030,
                directional_spread_deg=35.0,
                enable_wave_force=True,
                wave_force_gain=(45.0, 45.0, 65.0),
                max_wave_force=40.0,
                max_wave_moment=12.0,
            ),
        },
    ]

    total_curriculum_timesteps = sum(lesson["timesteps"] for lesson in lessons)

    print("Iniciando treinamento estruturado por currículo...")
    print(f"Total de timesteps planejado: {total_curriculum_timesteps:,}")

    is_first_iteration = True

    for index, lesson in enumerate(lessons):
        jonswap_params = lesson["jonswap"]

        print(
            "\n[CURRICULUM] "
            f"Etapa {index + 1}/{len(lessons)} | {lesson['name']}"
        )
        print(
            "[CURRICULUM] "
            f"Hs={jonswap_params['Hs']} m | "
            f"Tp={jonswap_params['Tp']} s | "
            f"gamma={jonswap_params['gamma']} | "
            f"N={jonswap_params['N']} | "
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

        callback = CallbackList(
            [
                checkpoint_callback,
                progress_callback,
            ]
        )

        model.learn(
            total_timesteps=lesson["timesteps"],
            callback=callback,
            tb_log_name="PPO_BlueROV2_Curriculum",
            reset_num_timesteps=is_first_iteration,
        )

        is_first_iteration = False

    print("Treinamento por currículo finalizado com sucesso.")

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    env.close()

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"VecNormalize salvo em: {VECNORM_PATH}")


if __name__ == "__main__":
    train_model_with_curriculum()