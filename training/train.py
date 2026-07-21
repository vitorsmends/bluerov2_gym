import os

import gymnasium as gym

from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tqdm.auto import tqdm


ENV_ID = "BlueRov-v0"

LOG_DIR = "./bluerov_tensorboard/"
MODEL_DIR = "./models/"
MODEL_PATH = os.path.join(MODEL_DIR, "bluerov_ppo")
VECNORM_PATH = os.path.join(MODEL_DIR, "bluerov_vec_normalize.pkl")

TOTAL_TIMESTEPS = 1_000_000


if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs:BlueRov",
        max_episode_steps=1000,
    )


class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps_target = int(total_timesteps)
        self.progress_bar = None
        self.last_n = 0

    def _on_training_start(self) -> None:
        self.last_n = min(self.model.num_timesteps, self.total_timesteps_target)

        self.progress_bar = tqdm(
            total=self.total_timesteps_target,
            initial=self.last_n,
            desc="Training progress",
            unit="steps",
            dynamic_ncols=True,
            leave=True,
        )

    def _on_step(self) -> bool:
        current_n = min(self.model.num_timesteps, self.total_timesteps_target)
        delta = current_n - self.last_n

        if delta > 0:
            self.progress_bar.update(delta)
            self.last_n = current_n

        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            remaining = self.total_timesteps_target - self.last_n
            if remaining > 0:
                self.progress_bar.update(remaining)
            self.progress_bar.close()


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
        "wave_force_application_point": tuple(float(v) for v in wave_force_application_point),
        "max_wave_force": float(max_wave_force),
        "max_wave_moment": float(max_wave_moment),
        "seed": int(seed),
    }


def train_model():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    moderate_jonswap = build_jonswap_params(
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
        seed=42,
    )

    env_config = {
        "dt": 0.1,
        "jonswap": moderate_jonswap,
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
        name_prefix="bluerov_ppo_moderate_checkpoint",
        save_vecnormalize=True,
    )

    progress_callback = TrainingProgressCallback(
        total_timesteps=TOTAL_TIMESTEPS,
    )

    callback = CallbackList(
        [
            checkpoint_callback,
            progress_callback,
        ]
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

    print("Iniciando treinamento sem curriculum learning...")
    print("Cenário: Moderate Sea - Quartering Waves")
    print(f"Total de timesteps: {TOTAL_TIMESTEPS:,}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        tb_log_name="PPO_BlueROV2_Moderate",
        reset_num_timesteps=True,
    )

    print("Treinamento finalizado com sucesso.")

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    env.close()

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"VecNormalize salvo em: {VECNORM_PATH}")


if __name__ == "__main__":
    train_model()