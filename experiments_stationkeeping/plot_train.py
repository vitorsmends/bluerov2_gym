import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 0. CONFIGURAÇÃO
# ==========================================
INPUT_DIR = "training_logs"
OUTPUT_DIR = "plots_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(name):
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {pdf_path}")
    plt.close()


# ==========================================
# 1. LOAD DATA
# ==========================================
episodes_df = pd.read_csv(os.path.join(INPUT_DIR, "ppo_training_episodes.csv"))
rollouts_df = pd.read_csv(os.path.join(INPUT_DIR, "ppo_training_rollouts.csv"))


# ==========================================
# 2. UTIL (SMOOTHING)
# ==========================================
def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")


# ==========================================
# 3. CONFIG ESTÉTICA
# ==========================================
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
})

COLOR_MAIN = "#1f77b4"


# ==========================================
# 4. PLOTS
# ==========================================

# ------------------------------------------
# FIG 1: REWARD POR EPISÓDIO
# ------------------------------------------
def plot_episode_reward():
    rewards = episodes_df["episode_reward"].values

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(rewards, alpha=0.3, linewidth=1.0, label="Raw")
    
    smoothed = moving_average(rewards, window=50)
    plt.plot(
        np.arange(len(smoothed)),
        smoothed,
        linewidth=2.0,
        label="Smoothed"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_plot("fig_training_episode_reward")


# ------------------------------------------
# FIG 2: COMPRIMENTO DO EPISÓDIO
# ------------------------------------------
def plot_episode_length():
    lengths = episodes_df["episode_length"].values

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(lengths, alpha=0.3, linewidth=1.0, label="Raw")

    smoothed = moving_average(lengths, window=50)
    plt.plot(np.arange(len(smoothed)), smoothed, linewidth=2.0, label="Smoothed")

    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_plot("fig_training_episode_length")


# ------------------------------------------
# FIG 3: REWARD MÉDIO (ROLLOUT)
# ------------------------------------------
def plot_rollout_reward():
    x = rollouts_df["num_timesteps"].values
    y = rollouts_df["rollout/ep_rew_mean"].values

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(x, y, linewidth=1.8)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_rollout_reward")


# ------------------------------------------
# FIG 4: LOSS TOTAL
# ------------------------------------------
def plot_loss():
    x = rollouts_df["num_timesteps"].values
    y = rollouts_df["train/loss"].values

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(x, y, linewidth=1.8)

    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_loss")


# ------------------------------------------
# FIG 5: VALUE LOSS
# ------------------------------------------
def plot_value_loss():
    x = rollouts_df["num_timesteps"].values
    y = rollouts_df["train/value_loss"].values

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(x, y, linewidth=1.8)

    plt.xlabel("Timesteps")
    plt.ylabel("Value Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_value_loss")


# ------------------------------------------
# FIG 6: POLICY LOSS
# ------------------------------------------
def plot_policy_loss():
    x = rollouts_df["num_timesteps"].values
    y = rollouts_df["train/policy_gradient_loss"].values

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(x, y, linewidth=1.8)

    plt.xlabel("Timesteps")
    plt.ylabel("Policy Gradient Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_policy_loss")


# ------------------------------------------
# FIG 7: KL DIVERGENCE
# ------------------------------------------
def plot_kl():
    x = rollouts_df["num_timesteps"].values
    y = rollouts_df["train/approx_kl"].values

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(x, y, linewidth=1.8)

    plt.xlabel("Timesteps")
    plt.ylabel("Approx KL")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_kl")


# ------------------------------------------
# FIG 8: ENTROPY
# ------------------------------------------
def plot_entropy():
    x = rollouts_df["num_timesteps"].values
    y = rollouts_df["train/entropy_loss"].values

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(x, y, linewidth=1.8)

    plt.xlabel("Timesteps")
    plt.ylabel("Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_entropy")


# ------------------------------------------
# FIG 9: TRACKING ERROR
# ------------------------------------------
def plot_tracking_error():
    err = episodes_df["tracking_dist"].values

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(err, alpha=0.3, linewidth=1.0)

    smoothed = moving_average(err, window=50)
    plt.plot(np.arange(len(smoothed)), smoothed, linewidth=2.0)

    plt.xlabel("Episode")
    plt.ylabel("Tracking Error [m]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot("fig_training_tracking_error")


# ==========================================
# 5. MAIN
# ==========================================
def main():
    plot_episode_reward()
    plot_episode_length()
    plot_rollout_reward()
    plot_loss()
    plot_value_loss()
    plot_policy_loss()
    plot_kl()
    plot_entropy()
    plot_tracking_error()

    print("\n[OK] Todos os plots gerados.")


if __name__ == "__main__":
    main()