import os
import json
from datetime import datetime
import gymnasium as gym
import numpy as np
import math
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import bluerov2_gym.envs.bluerov_env as original_env

# 1. BASE ENVIRONMENT REGISTRATION
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except:
    pass

# ==========================================
# 2. TRAJECTORY GENERATOR (THE MASTER)
# ==========================================
class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0  
        self.speed = 0.15   # Training speed
        self.z_target = -0.5 

    def get_state_at_time(self, t):
        t_s = t * self.speed
        
        # Position (Figure 8)
        x = self.radius * math.sin(t_s)
        y = self.radius * math.sin(t_s) * math.cos(t_s)
        
        # Z (Smooth ramp then hold)
        if t < 10.0:
            z = (self.z_target / 10.0) * t
        else:
            z = self.z_target

        # Velocity (Derivative for Feedforward)
        vx = self.radius * math.cos(t_s) * self.speed
        vy = self.radius * (math.cos(t_s)**2 - math.sin(t_s)**2) * self.speed
        vz = 0.0

        # Desired Yaw (Looking forward)
        yaw = math.atan2(vy, vx)

        return np.array([x, y, z]), np.array([0, 0, yaw]), np.array([vx, vy, vz])

# ==========================================
# 3. CUSTOM ENVIRONMENT (INHERITANCE)
# ==========================================
class TrajectoryTrackingEnv(original_env.BlueRov):
    def __init__(self):
        super().__init__(render_mode=None) # Call original init without render
        self.traj = TrajectoryGenerator()
        self.current_t = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        # RANDOM RESET: Start at any point in the trajectory
        self.current_t = np.random.uniform(0, 50.0) 
        
        # Get where it should be at this time
        target_pos, target_att, target_vel = self.traj.get_state_at_time(self.current_t)
        
        # Place the robot physically there (with a bit of initial noise/error)
        noise_pos = np.random.uniform(-0.2, 0.2, 3)
        initial_pos = target_pos + noise_pos
        
        # Reset the physical simulator to this position
        self.state['x'] = initial_pos[0]
        self.state['y'] = initial_pos[1]
        self.state['z'] = initial_pos[2]
        self.state['roll'] = 0.0
        self.state['pitch'] = 0.0
        self.state['yaw'] = target_att[2]
        
        self.state['u'] = target_vel[0]
        self.state['v'] = target_vel[1]
        self.state['w'] = target_vel[2]
        self.state['p'] = 0.0
        self.state['q'] = 0.0
        self.state['r'] = 0.0
        
        return self._get_error_obs(), {}

    def _get_error_obs(self):
        # CALCULATE REAL ERROR (Virtual Observation)
        # PPO needs to see the DIFFERENCE, not the absolute position
        tgt_pos, tgt_att, tgt_vel = self.traj.get_state_at_time(self.current_t)
        
        curr_pos = np.array([self.state['x'], self.state['y'], self.state['z']])
        curr_vel = np.array([self.state['u'], self.state['v'], self.state['w']])
        
        error_pos = curr_pos - tgt_pos
        error_vel = curr_vel - tgt_vel # CRUCIAL: Teaches velocity tracking
        
        # Rotate error to body frame (Makes learning much easier)
        psi = self.state['yaw']
        c, s = np.cos(psi), np.sin(psi)
        
        err_x_body =  error_pos[0]*c + error_pos[1]*s
        err_y_body = -error_pos[0]*s + error_pos[1]*c
        err_z_body =  error_pos[2]
        
        # Replace original observation with ERROR observation
        obs = {
            'x': np.array([err_x_body], dtype=np.float32),
            'y': np.array([err_y_body], dtype=np.float32),
            'z': np.array([err_z_body], dtype=np.float32),
            'roll': np.array([self.state['roll']], dtype=np.float32),
            'pitch': np.array([self.state['pitch']], dtype=np.float32),
            'yaw': np.array([self.state['yaw'] - tgt_att[2]], dtype=np.float32),
            'u': np.array([error_vel[0]], dtype=np.float32),
            'v': np.array([error_vel[1]], dtype=np.float32),
            'w': np.array([error_vel[2]], dtype=np.float32),
            'p': np.array([self.state['p']], dtype=np.float32),
            'q': np.array([self.state['q']], dtype=np.float32),
            'r': np.array([self.state['r']], dtype=np.float32)
        }
        return obs

    def step(self, action):
        # 1. Advance time
        self.current_t += self.dt
        
        # 2. Execute action in the physical simulator
        self.dynamics.step(self.state, action)
        
        # 3. Get error observation
        obs_error = self._get_error_obs()
        
        # 4. Calculate reward
        reward = self.reward_fn.get_reward(obs_error, action)
        
        # 5. Termination conditions
        terminated = False
        dist = np.sqrt(obs_error['x'][0]**2 + obs_error['y'][0]**2 + obs_error['z'][0]**2)
        
        # If it moves too far away, end episode (fail)
        if dist > 3.0: 
            terminated = True
            reward -= 10.0 # Severe punishment
            
        # Depth limit
        if abs(self.state["z"]) > 20.0:
            terminated = True
            
        # Stability limit (Capsizing)
        if abs(self.state["roll"]) > 1.5 or abs(self.state["pitch"]) > 1.5:
            terminated = True

        truncated = False

        return obs_error, reward, terminated, truncated, {}

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    print("[INFO] Starting PPO training configuration...")
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_traj_{now}"
    base_dir = "trained_models"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    hyperparameters = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "total_timesteps": 1_000_000
    }

    json_path = os.path.join(run_dir, "hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)

    def make_env():
        env = TrajectoryTrackingEnv()
        env = Monitor(env, run_dir)
        return env

    # Create vectorized environment with normalization
    # Normalization is CRITICAL for PPO to work well with physics
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # PPO Model
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=hyperparameters["learning_rate"],
        n_steps=hyperparameters["n_steps"],
        batch_size=hyperparameters["batch_size"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"]
    )

    # Configure Custom Logger
    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Checkpoint callback
    checkpoints_path = os.path.join(run_dir, 'checkpoints')
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=checkpoints_path, 
        name_prefix='ppo_traj'
    )

    print(f"[INFO] Training running. Data being saved to: {run_dir}")
    model.learn(total_timesteps=hyperparameters["total_timesteps"], callback=checkpoint_callback)

    # Save final model and normalization stats
    model.save(os.path.join(run_dir, "ppo_trajectory_final"))
    env.save(os.path.join(run_dir, "vec_normalize.pkl"))
    print("[INFO] Training completed successfully!")

if __name__ == "__main__":
    train()