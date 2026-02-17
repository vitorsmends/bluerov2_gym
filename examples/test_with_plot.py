import time
import csv
import numpy as np
import gymnasium as gym

# =========================================================
# Matplotlib backend (headless-safe)
# =========================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =========================================================
# Register environment (CORRECT ENTRY POINT)
# =========================================================
register(
    id="BlueRov-v0",
    entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
    max_episode_steps=100,
)


# =========================================================
# Plot trajectory (simple and robust)
# =========================================================
def plot_trajectory(traj_x, traj_y, traj_z, episode_id):
    fig = plt.figure(figsize=(10, 4))

    # --- 3D trajectory ---
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(traj_x, traj_y, traj_z)
    ax.scatter(traj_x[0], traj_y[0], traj_z[0], c="g", label="start")
    ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], c="r", label="end")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D trajectory")
    ax.legend()
    ax.grid(True)

    # --- Position vs time ---
    t = np.arange(len(traj_x))
    ax2 = fig.add_subplot(122)
    ax2.plot(t, traj_x, label="x")
    ax2.plot(t, traj_y, label="y")
    ax2.plot(t, traj_z, label="z")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Position [m]")
    ax2.set_title("Position vs time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    filename = f"trajectory_episode_{episode_id}.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved plot: {filename}")


# =========================================================
# Save trajectory to CSV
# =========================================================
def save_csv(traj_x, traj_y, traj_z, episode_id):
    filename = f"trajectory_episode_{episode_id}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "z"])
        for i, (x, y, z) in enumerate(zip(traj_x, traj_y, traj_z)):
            writer.writerow([i, x, y, z])

    print(f"[INFO] Saved CSV: {filename}")


# =========================================================
# Run trained PPO agent
# =========================================================
def test_agent():
    print("[INFO] Initializing environment...")
    env = gym.make("BlueRov-v0", render_mode="human")

    print("[INFO] Loading PPO model...")
    model = PPO.load("bluerov_ppo")

    print("[INFO] Loading VecNormalize stats...")
    vec_env = DummyVecEnv([lambda: gym.make("BlueRov-v0")])
    vec_env = VecNormalize.load("bluerov_vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    episodes = 5

    for episode in range(episodes):
        print(f"\n[INFO] Starting episode {episode + 1}")
        obs, _ = env.reset()
        env.render()

        traj_x, traj_y, traj_z = [], [], []
        episode_reward = 0.0
        step_count = 0

        while True:
            # Normalize observation
            obs_norm = vec_env.normalize_obs(obs)

            # Agent action
            action, _ = model.predict(obs_norm, deterministic=True)

            # Environment step
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            # Update MeshCat visualization (IMPORTANT FIX)
            env.unwrapped.step_sim()
            time.sleep(0.1)

            # Log trajectory
            traj_x.append(obs["x"][0])
            traj_y.append(obs["y"][0])
            traj_z.append(obs["z"][0])

            step_count += 1

            print(
                f"Step {step_count:03d} | "
                f"x={obs['x'][0]:.2f}, "
                f"y={obs['y'][0]:.2f}, "
                f"z={obs['z'][0]:.2f}"
            )

            if terminated or truncated:
                print(
                    f"[INFO] Episode finished in {step_count} steps | "
                    f"Total reward = {episode_reward:.2f}"
                )
                break

        # Save results
        save_csv(traj_x, traj_y, traj_z, episode + 1)
        plot_trajectory(traj_x, traj_y, traj_z, episode + 1)

    env.close()
    print("[INFO] All episodes completed.")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    test_agent()
