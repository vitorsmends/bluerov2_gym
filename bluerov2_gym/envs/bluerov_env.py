from importlib import resources

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.dt = 0.1

        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.renderer = BlueRovRenderer(render_mode=render_mode)
        self.reward_fn = Reward()
        self.dynamics = Dynamics()

        self.state_keys = [
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "u", "v", "w",
            "p", "q", "r",
        ]

        self.state = {key: 0.0 for key in self.state_keys}

        # Direct thruster commands: [T1, T2, T3, T4, T5, T6]
        self.action_space = spaces.Box(
            low=-40.0,
            high=40.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                )
                for key in self.state_keys
            }
        )

    def _get_obs(self):
        return {
            key: np.array([self.state[key]], dtype=np.float32)
            for key in self.state_keys
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        for key in self.state_keys:
            self.state[key] = 0.0

        self.dynamics.reset()
        self.reward_fn.reset()

        obs = self._get_obs()
        info = {}

        return obs, info

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

        info = {
            "x": float(self.state["x"]),
            "y": float(self.state["y"]),
            "z": float(self.state["z"]),
            "roll": float(self.state["roll"]),
            "pitch": float(self.state["pitch"]),
            "yaw": float(self.state["yaw"]),
        }

        if "tau" in self.state:
            info["tau"] = self.state["tau"]

        if "thrusters" in self.state:
            info["thrusters"] = self.state["thrusters"]

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