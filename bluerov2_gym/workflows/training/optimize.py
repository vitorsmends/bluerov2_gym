import gc
import logging
import os

import gymnasium as gym
import optuna
import torch

from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tqdm.auto import tqdm


ENV_ID = "BlueRov-v0"

LOG_DIR = "./bluerov_tensorboard/"
MODEL_DIR = "./models/"
OPTUNA_DIR = "./optuna_results/"

OPTUNA_DB_PATH = os.path.join(OPTUNA_DIR, "ppo_bluerov_optuna.db")
OPTUNA_LOG_PATH = os.path.join(OPTUNA_DIR, "optuna_training.log")

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "bluerov_ppo_optuna_best")
FINAL_VECNORM_PATH = os.path.join(MODEL_DIR, "bluerov_vec_normalize_optuna_best.pkl")

TOTAL_TIMESTEPS_TRIAL = 200_000
TOTAL_TIMESTEPS_FINAL = 1_000_000

N_TRIALS = 30
N_EVAL_EPISODES = 5
SEED = 42


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OPTUNA_DIR, exist_ok=True)


logging.basicConfig(
    filename=OPTUNA_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
        desc: str = "Training progress",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps_target = int(total_timesteps)
        self.desc = desc
        self.progress_bar = None
        self.last_n = 0

    def _on_training_start(self) -> None:
        self.last_n = min(
            self.model.num_timesteps,
            self.total_timesteps_target,
        )

        self.progress_bar = tqdm(
            total=self.total_timesteps_target,
            initial=self.last_n,
            desc=self.desc,
            unit="steps",
            dynamic_ncols=True,
            leave=True,
        )

    def _on_step(self) -> bool:
        current_n = min(
            self.model.num_timesteps,
            self.total_timesteps_target,
        )

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


def get_env_config(seed=42):
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
        seed=seed,
    )

    return {
        "dt": 0.1,
        "jonswap": moderate_jonswap,
    }


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


def make_vec_env(env_config, gamma=0.99, training=True):
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
        gamma=gamma,
    )

    env.training = training
    env.norm_reward = training

    return env


def sample_ppo_params(trial):
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True,
    )

    n_steps = trial.suggest_categorical(
        "n_steps",
        [1024, 2048, 4096, 8192],
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [64, 128, 256, 512],
    )

    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    n_epochs = trial.suggest_int(
        "n_epochs",
        5,
        15,
    )

    gamma = trial.suggest_float(
        "gamma",
        0.97,
        0.999,
    )

    gae_lambda = trial.suggest_float(
        "gae_lambda",
        0.90,
        0.98,
    )

    clip_range = trial.suggest_float(
        "clip_range",
        0.10,
        0.30,
    )

    ent_coef = trial.suggest_float(
        "ent_coef",
        1e-5,
        0.03,
        log=True,
    )

    vf_coef = trial.suggest_float(
        "vf_coef",
        0.3,
        1.0,
    )

    max_grad_norm = trial.suggest_float(
        "max_grad_norm",
        0.3,
        1.0,
    )

    net_arch_name = trial.suggest_categorical(
        "net_arch",
        [
            "small",
            "medium",
            "large",
        ],
    )

    if net_arch_name == "small":
        net_arch = dict(
            pi=[128, 128],
            vf=[128, 128],
        )
    elif net_arch_name == "medium":
        net_arch = dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    else:
        net_arch = dict(
            pi=[512, 512],
            vf=[512, 512],
        )

    activation_name = trial.suggest_categorical(
        "activation_fn",
        [
            "tanh",
            "relu",
        ],
    )

    if activation_name == "tanh":
        activation_fn = torch.nn.Tanh
    else:
        activation_fn = torch.nn.ReLU

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }


def objective(trial):
    logging.info("=" * 80)
    logging.info(f"Iniciando trial {trial.number}")

    env_config = get_env_config(seed=SEED)
    ppo_params = sample_ppo_params(trial)

    logging.info(f"Parâmetros do trial {trial.number}: {ppo_params}")

    train_env = None
    eval_env = None
    model = None

    try:
        train_env = make_vec_env(
            env_config=env_config,
            gamma=ppo_params["gamma"],
            training=True,
        )

        eval_env = make_vec_env(
            env_config=env_config,
            gamma=ppo_params["gamma"],
            training=False,
        )

        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            verbose=0,
            tensorboard_log=LOG_DIR,
            seed=SEED,
            **ppo_params,
        )

        progress_callback = TrainingProgressCallback(
            total_timesteps=TOTAL_TIMESTEPS_TRIAL,
            desc=f"Optuna trial {trial.number}",
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_TRIAL,
            callback=progress_callback,
            tb_log_name=f"PPO_BlueROV2_Optuna_Trial_{trial.number}",
            reset_num_timesteps=True,
        )

        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
        )

        trial.set_user_attr("mean_reward", float(mean_reward))
        trial.set_user_attr("std_reward", float(std_reward))

        logging.info(
            f"Trial {trial.number} finalizado | "
            f"mean_reward={mean_reward:.4f} | "
            f"std_reward={std_reward:.4f}"
        )

        print(
            f"\nTrial {trial.number} finalizado | "
            f"mean_reward={mean_reward:.4f} | "
            f"std_reward={std_reward:.4f}"
        )

        return mean_reward

    except Exception as error:
        logging.exception(f"Erro no trial {trial.number}: {error}")
        raise error

    finally:
        if train_env is not None:
            train_env.close()

        if eval_env is not None:
            eval_env.close()

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def optimize_hyperparameters():
    study = optuna.create_study(
        study_name="ppo_bluerov_moderate_optuna",
        direction="maximize",
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        ),
    )

    print("\nIniciando otimização com Optuna...")
    print(f"Número de trials: {N_TRIALS}")
    print(f"Timesteps por trial: {TOTAL_TIMESTEPS_TRIAL:,}")
    print(f"Logs salvos em: {OPTUNA_LOG_PATH}")
    print(f"Banco Optuna salvo em: {OPTUNA_DB_PATH}")

    logging.info("Iniciando otimização com Optuna")
    logging.info(f"N_TRIALS={N_TRIALS}")
    logging.info(f"TOTAL_TIMESTEPS_TRIAL={TOTAL_TIMESTEPS_TRIAL}")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    print("\nOtimização finalizada.")
    print(f"Melhor recompensa média: {study.best_value:.4f}")
    print("\nMelhores hiperparâmetros:")

    logging.info("Otimização finalizada")
    logging.info(f"Melhor recompensa média: {study.best_value:.4f}")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")
        logging.info(f"{key}: {value}")

    return study.best_params


def build_final_params(best_params):
    net_arch_name = best_params["net_arch"]

    if net_arch_name == "small":
        net_arch = dict(
            pi=[128, 128],
            vf=[128, 128],
        )
    elif net_arch_name == "medium":
        net_arch = dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    else:
        net_arch = dict(
            pi=[512, 512],
            vf=[512, 512],
        )

    if best_params["activation_fn"] == "tanh":
        activation_fn = torch.nn.Tanh
    else:
        activation_fn = torch.nn.ReLU

    return {
        "learning_rate": best_params["learning_rate"],
        "n_steps": best_params["n_steps"],
        "batch_size": best_params["batch_size"],
        "n_epochs": best_params["n_epochs"],
        "gamma": best_params["gamma"],
        "gae_lambda": best_params["gae_lambda"],
        "clip_range": best_params["clip_range"],
        "ent_coef": best_params["ent_coef"],
        "vf_coef": best_params["vf_coef"],
        "max_grad_norm": best_params["max_grad_norm"],
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }


def train_final_model(best_params):
    env_config = get_env_config(seed=SEED)
    final_params = build_final_params(best_params)

    env = make_vec_env(
        env_config=env_config,
        gamma=final_params["gamma"],
        training=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="bluerov_ppo_optuna_best_checkpoint",
        save_vecnormalize=True,
    )

    progress_callback = TrainingProgressCallback(
        total_timesteps=TOTAL_TIMESTEPS_FINAL,
        desc="Final training",
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
        seed=SEED,
        **final_params,
    )

    print("\nIniciando treinamento final com os melhores hiperparâmetros...")
    print("Cenário: Moderate Sea - Quartering Waves")
    print(f"Total de timesteps: {TOTAL_TIMESTEPS_FINAL:,}")

    logging.info("Iniciando treinamento final")
    logging.info(f"Parâmetros finais: {final_params}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS_FINAL,
        callback=callback,
        tb_log_name="PPO_BlueROV2_Moderate_Optuna_Best",
        reset_num_timesteps=True,
    )

    model.save(FINAL_MODEL_PATH)
    env.save(FINAL_VECNORM_PATH)

    env.close()

    print("\nTreinamento finalizado com sucesso.")
    print(f"Modelo salvo em: {FINAL_MODEL_PATH}")
    print(f"VecNormalize salvo em: {FINAL_VECNORM_PATH}")

    logging.info("Treinamento finalizado")
    logging.info(f"Modelo salvo em: {FINAL_MODEL_PATH}")
    logging.info(f"VecNormalize salvo em: {FINAL_VECNORM_PATH}")


if __name__ == "__main__":
    best_params = optimize_hyperparameters()
    train_final_model(best_params)