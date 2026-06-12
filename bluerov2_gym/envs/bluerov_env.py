from importlib import resources
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, env_config: dict | None = None, dynamics_config: dict | None = None):
        super().__init__()
        self.render_mode = render_mode

        cfg = env_config if env_config is not None else {}
        self.dt = cfg.get("dt", 0.1)
        self.jonswap_params = cfg.get("jonswap", None)

        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.renderer = BlueRovRenderer(render_mode=render_mode)
        self.reward_fn = Reward()
        self.dynamics = Dynamics(dynamics_config=dynamics_config, jonswap_params=self.jonswap_params)

        self.state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]
        self.state = {key: 0.0 for key in self.state_keys}

        self.action_space = spaces.Box(low=-40.0, high=40.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {key: spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) for key in self.state_keys}
        )

    def _get_obs(self):
        return {key: np.array([self.state[key]], dtype=np.float32) for key in self.state_keys}

    def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)

            # CORREÇÃO: Voltando exatamente ao que era antes. 
            # O robô nasce fixo na origem absoluta para focar puramente em regulação local.
            for key in self.state_keys:
                self.state[key] = 0.0

            reset_jonswap_params = None
            if options is not None:
                reset_jonswap_params = options.get("jonswap_params", None)

            if reset_jonswap_params is not None:
                self.jonswap_params = reset_jonswap_params.copy()
                self.dynamics.reset(jonswap_params=self.jonswap_params)
            else:
                self.dynamics.reset()

            self.reward_fn.reset()
            return self._get_obs(), {}
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        self.dynamics.step(self.state, action)
        obs = self._get_obs()
        reward = self.reward_fn.get_reward(obs, action)

        terminated = False
        if abs(self.state["z"]) > 20.0:
            terminated = True
        if abs(self.state["x"]) > 30.0 or abs(self.state["y"]) > 30.0:
            terminated = True
        if abs(self.state["roll"]) > 1.5 or abs(self.state["pitch"]) > 1.5:
            terminated = True

        truncated = False
        
        pos_error = float(np.sqrt(self.state["x"]**2 + self.state["y"]**2 + self.state["z"]**2))
        yaw_error = float(abs(np.arctan2(np.sin(self.state["yaw"]), np.cos(self.state["yaw"]))))

        info = {
            "x": float(self.state["x"]),
            "y": float(self.state["y"]),
            "z": float(self.state["z"]),
            "roll": float(self.state["roll"]),
            "pitch": float(self.state["pitch"]),
            "yaw": float(self.state["yaw"]),
            "metrics/position_error_euclidean": pos_error,
            "metrics/yaw_error_rad": yaw_error,
        }

        if "tau" in self.state:
            info["tau"] = self.state["tau"]
        if "thrusters" in self.state:
            info["thrusters"] = self.state["thrusters"]
        if "nu_current" in self.state:
            info["nu_current"] = self.state["nu_current"]

        if self.render_mode == "human":
            self.step_sim()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.renderer.render(self.model_path)

    def step_sim(self):
        if self.render_mode == "human":
            self.renderer.step_sim(self.state)

    def close(self):
        pass