
from importlib import resources

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.config_utils import load_yaml
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer
from bluerov2_gym.envs.core.actuator_faults import FaultManager


DEFAULT_ENV_CONFIG = {
    "dt": 0.1,
    "jonswap": None,
    "reward": {},
    "action_space": {
        "low": -40.0,
        "high": 40.0,
        "shape": [6],
    },
    "termination": {
        "max_abs_z": 20.0,
        "max_abs_x": 30.0,
        "max_abs_y": 30.0,
        "max_abs_roll": 1.5,
        "max_abs_pitch": 1.5,
    },
    "metadata": {
        "render_fps": 30,
    },
}


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, env_config=None, dynamics_config=None):
        super().__init__()

        self.render_mode = render_mode

        if isinstance(env_config, (str, bytes)):
            env_config = load_yaml(env_config)
        if isinstance(dynamics_config, (str, bytes)):
            dynamics_config = load_yaml(dynamics_config)

        cfg = env_config if env_config is not None else {}

        self.dt = float(cfg.get("dt", 0.1))
        self.jonswap_params = cfg.get("jonswap", None)
        self.termination_config = cfg.get("termination", {})

        action_cfg = cfg.get("action_space", {})
        self.action_low = float(action_cfg.get("low", -40.0))
        self.action_high = float(action_cfg.get("high", 40.0))
        action_shape = tuple(action_cfg.get("shape", [6]))

        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.renderer = BlueRovRenderer(render_mode=render_mode)
        self.reward_fn = Reward(config=cfg.get("reward", None))

        merged = dict(dynamics_config or {})
        merged.setdefault("dt", self.dt)

        self.dynamics = Dynamics(
            dynamics_config=merged,
            jonswap_params=self.jonswap_params,
        )

        self.fault_manager = None

        self.state_keys = [
            "x","y","z","roll","pitch","yaw",
            "u","v","w","p","q","r"
        ]
        self.state = {k:0.0 for k in self.state_keys}

        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            shape=action_shape,
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            k: spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            for k in self.state_keys
        })

    def set_fault_manager(self, fault_manager: FaultManager | None):
        self.fault_manager = fault_manager

    def clear_fault_manager(self):
        self.fault_manager = None

    def _process_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_low, self.action_high)

        if self.fault_manager is not None:
            action = self.fault_manager.apply(action)

        return action

    def _get_obs(self):
        return {k: np.array([self.state[k]], dtype=np.float32)
                for k in self.state_keys}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        for k in self.state_keys:
            self.state[k] = 0.0

        reset_jonswap = None if options is None else options.get("jonswap_params")

        if reset_jonswap is not None:
            self.jonswap_params = reset_jonswap.copy()
            self.dynamics.reset(jonswap_params=self.jonswap_params)
        else:
            self.dynamics.reset()

        if self.fault_manager is not None:
            self.fault_manager.reset()

        self.reward_fn.reset()
        return self._get_obs(), {}

    def step(self, action):
        action = self._process_action(action)

        self.dynamics.step(self.state, action)

        obs = self._get_obs()
        reward = self.reward_fn.get_reward(obs, action)

        tcfg = self.termination_config
        terminated = (
            abs(self.state["z"]) > float(tcfg.get("max_abs_z",20.0))
            or abs(self.state["x"]) > float(tcfg.get("max_abs_x",30.0))
            or abs(self.state["y"]) > float(tcfg.get("max_abs_y",30.0))
            or abs(self.state["roll"]) > float(tcfg.get("max_abs_roll",1.5))
            or abs(self.state["pitch"]) > float(tcfg.get("max_abs_pitch",1.5))
        )

        truncated = False

        pos_error = float(np.sqrt(
            self.state["x"]**2 +
            self.state["y"]**2 +
            self.state["z"]**2
        ))

        yaw_error = float(abs(np.arctan2(
            np.sin(self.state["yaw"]),
            np.cos(self.state["yaw"])
        )))

        info = {k: float(self.state[k]) for k in self.state_keys}
        info["metrics/position_error_euclidean"] = pos_error
        info["metrics/yaw_error_rad"] = yaw_error

        if self.fault_manager is not None:
            info["fault_manager"] = self.fault_manager.__class__.__name__

        self._append_dynamics_info(info)

        if self.render_mode == "human":
            self.step_sim()

        return obs, float(reward), terminated, truncated, info

    def _append_dynamics_info(self, info: dict):
        vector_keys = [
            "tau", "tau_wave", "tau_total", "thrusters", "nu_current",
            "nu_wave_raw", "nu_rel", "nu_dot", "coriolis_rb",
            "coriolis_added", "damping", "restoring",
        ]
        scalar_keys = ["wave_elevation"]

        for key in vector_keys:
            if key in self.state:
                info[key] = np.asarray(self.state[key], dtype=np.float32).copy()

        for key in scalar_keys:
            if key in self.state:
                info[key] = float(self.state[key])

        if "nu_current" in self.state:
            value = np.asarray(self.state["nu_current"], dtype=float)
            info["metrics/current_velocity_norm"] = float(np.linalg.norm(value))

        if "nu_wave_raw" in self.state:
            value = np.asarray(self.state["nu_wave_raw"], dtype=float)
            info["metrics/raw_wave_velocity_norm"] = float(np.linalg.norm(value))

        if "tau_wave" in self.state:
            value = np.asarray(self.state["tau_wave"], dtype=float)
            info["metrics/wave_force_norm"] = float(np.linalg.norm(value[0:3]))
            info["metrics/wave_moment_norm"] = float(np.linalg.norm(value[3:6]))

        if "tau" in self.state:
            value = np.asarray(self.state["tau"], dtype=float)
            info["metrics/control_force_norm"] = float(np.linalg.norm(value[0:3]))
            info["metrics/control_moment_norm"] = float(np.linalg.norm(value[3:6]))

        if "tau_total" in self.state:
            value = np.asarray(self.state["tau_total"], dtype=float)
            info["metrics/total_force_norm"] = float(np.linalg.norm(value[0:3]))
            info["metrics/total_moment_norm"] = float(np.linalg.norm(value[3:6]))

        if "thrusters" in self.state:
            value = np.asarray(self.state["thrusters"], dtype=float)
            info["metrics/thruster_l2_norm"] = float(np.linalg.norm(value))
            info["metrics/thruster_abs_sum"] = float(np.sum(np.abs(value)))

    def render(self):
        if self.render_mode == "human":
            self.renderer.render(self.model_path)

    def step_sim(self):
        if self.render_mode == "human":
            self.renderer.step_sim(self.state)

    def close(self):
        pass
