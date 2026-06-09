import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("results/trajectory_experiments")
OUTPUT_DIR = Path("results/plots_trajectory_experiments")
TABLE_DIR = Path("results/tables_trajectory_experiments")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = {
    # "PID":  {"key": "pid",  "color": "#4E79A7", "ls": "-"},
    # "SMC":  {"key": "smc",  "color": "#B07AA1", "ls": "-"},
    "NMPC": {"key": "nmpc", "color": "#E15759", "ls": "-"},
    "PPO":  {"key": "ppo",  "color": "#59A14F", "ls": "-"},
}


TRAJECTORIES = {
    "square": "Square",
    "figure_eight_depth": "Figure Eight with Depth Variation",
    "circle": "Circle",
    "letter_b": "Letter B",
    "star": "Star",
}


def save_plot(name):
    plt.savefig(OUTPUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {name}")


def load_results():
    data = {}

    required_cols = {
        "trajectory", "controller", "time",
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "u", "v", "w",
        "x_ref", "y_ref", "z_ref",
        "yaw_ref",
        "tracking_error_m",
        "T1", "T2", "T3", "T4", "T5", "T6",
        "control_effort",
        "control_effort_normalized",
    }

    for traj_key in TRAJECTORIES:
        data[traj_key] = {}

        for ctrl_name, cfg in CONFIG.items():
            path = INPUT_DIR / f"{cfg['key']}_{traj_key}.csv"

            if not path.exists():
                print(f"[WARN] Missing file: {path}")
                continue

            df = pd.read_csv(path)

            missing = required_cols - set(df.columns)
            if missing:
                print(f"[WARN] {ctrl_name}/{traj_key} skipped. Missing: {sorted(missing)}")
                continue

            df = df.sort_values("time").reset_index(drop=True)
            data[traj_key][ctrl_name] = df

    return data


def build_summary(data):
    rows = []

    for traj_key, ctrl_data in data.items():
        for ctrl_name, df in ctrl_data.items():
            err = df["tracking_error_m"].to_numpy(dtype=float)
            effort = df["control_effort"].to_numpy(dtype=float)

            rows.append({
                "trajectory": traj_key,
                "trajectory_label": TRAJECTORIES[traj_key],
                "controller": ctrl_name,
                "n_steps": len(df),
                "duration_s": float(df["time"].max() - df["time"].min()),
                "mean_error_m": float(np.mean(err)),
                "rmse_error_m": float(np.sqrt(np.mean(err ** 2))),
                "p95_error_m": float(np.percentile(err, 95)),
                "max_error_m": float(np.max(err)),
                "final_error_m": float(err[-1]),
                "mean_control_effort": float(np.mean(effort)),
                "total_control_effort": float(np.sum(effort)),
                "mean_normalized_effort": float(np.mean(df["control_effort_normalized"])),
            })

    summary = pd.DataFrame(rows)
    path = TABLE_DIR / "trajectory_experiments_summary.csv"
    summary.to_csv(path, index=False)

    print(f"[OK] Saved table: {path}")
    print(summary.round(4).to_string(index=False))

    return summary


def get_reference_df(ctrl_data):
    first_df = next(iter(ctrl_data.values()))
    return first_df


def plot_xy_for_trajectory(traj_key, ctrl_data):
    if not ctrl_data:
        return

    plt.figure(figsize=(6.5, 4.2))

    ref = get_reference_df(ctrl_data)
    plt.plot(
        ref["x_ref"], ref["y_ref"],
        "k--",
        linewidth=2.0,
        label="Reference",
    )

    for ctrl_name, df in ctrl_data.items():
        cfg = CONFIG[ctrl_name]

        plt.plot(
            df["x"], df["y"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.8,
            label=ctrl_name,
        )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    # plt.title(TRAJECTORIES[traj_key])
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(f"xy_{traj_key}")


def plot_3d_for_trajectory(traj_key, ctrl_data):
    if not ctrl_data:
        return

    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    ref = get_reference_df(ctrl_data)
    ax.plot(
        ref["x_ref"], ref["y_ref"], ref["z_ref"],
        "k--",
        linewidth=2.0,
        label="Reference",
    )

    for ctrl_name, df in ctrl_data.items():
        cfg = CONFIG[ctrl_name]

        ax.plot(
            df["x"], df["y"], df["z"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.6,
            label=ctrl_name,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    # ax.set_title(TRAJECTORIES[traj_key])
    ax.invert_zaxis()
    ax.legend(frameon=True)
    plt.tight_layout()

    save_plot(f"trajectory_3d_{traj_key}")


def plot_error_for_trajectory(traj_key, ctrl_data):
    if not ctrl_data:
        return

    plt.figure(figsize=(6.5, 4.2))

    for ctrl_name, df in ctrl_data.items():
        cfg = CONFIG[ctrl_name]

        plt.plot(
            df["time"],
            df["tracking_error_m"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.8,
            label=ctrl_name,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Euclidean Tracking Error [m]")
    # plt.title(TRAJECTORIES[traj_key])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(f"tracking_error_{traj_key}")


def plot_depth_for_trajectory(traj_key, ctrl_data):
    if not ctrl_data:
        return

    plt.figure(figsize=(6.5, 4.2))

    ref = get_reference_df(ctrl_data)
    plt.plot(
        ref["time"],
        ref["z_ref"],
        "k--",
        linewidth=2.0,
        label="Reference",
    )

    for ctrl_name, df in ctrl_data.items():
        cfg = CONFIG[ctrl_name]

        plt.plot(
            df["time"],
            df["z"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.8,
            label=ctrl_name,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Z [m]")
    # plt.title(f"Depth Response - {TRAJECTORIES[traj_key]}")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(f"depth_response_{traj_key}")


def plot_control_effort_for_trajectory(traj_key, ctrl_data):
    if not ctrl_data:
        return

    plt.figure(figsize=(6.5, 4.2))

    for ctrl_name, df in ctrl_data.items():
        cfg = CONFIG[ctrl_name]

        plt.plot(
            df["time"],
            df["control_effort"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.8,
            label=ctrl_name,
        )

    plt.xlabel("Time [s]")
    plt.ylabel(r"Control Effort $\sum_i T_i^2$ [N$^2$]")
    # plt.title(TRAJECTORIES[traj_key])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(f"control_effort_{traj_key}")


def plot_yaw_for_trajectory(traj_key, ctrl_data):
    if not ctrl_data:
        return

    plt.figure(figsize=(6.5, 4.2))

    ref = get_reference_df(ctrl_data)
    plt.plot(
        ref["time"],
        ref["yaw_ref"],
        "k--",
        linewidth=2.0,
        label="Reference",
    )

    for ctrl_name, df in ctrl_data.items():
        cfg = CONFIG[ctrl_name]

        plt.plot(
            df["time"],
            df["yaw"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=1.8,
            label=ctrl_name,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Yaw [rad]")
    # plt.title(f"Yaw Response - {TRAJECTORIES[traj_key]}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(f"yaw_response_{traj_key}")


def plot_all_trajectories_xy_reference(data):
    plt.figure(figsize=(6.5, 4.8))

    for traj_key, ctrl_data in data.items():
        if not ctrl_data:
            continue

        ref = get_reference_df(ctrl_data)

        plt.plot(
            ref["x_ref"],
            ref["y_ref"],
            linewidth=1.8,
            label=TRAJECTORIES[traj_key],
        )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot("all_reference_trajectories_xy")


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    data = load_results()

    if not any(data[traj] for traj in data):
        print("[ERROR] No valid trajectory experiment CSV files found.")
        return

    build_summary(data)

    plot_all_trajectories_xy_reference(data)

    for traj_key, ctrl_data in data.items():
        plot_xy_for_trajectory(traj_key, ctrl_data)
        plot_3d_for_trajectory(traj_key, ctrl_data)
        # plot_error_for_trajectory(traj_key, ctrl_data)
        # plot_depth_for_trajectory(traj_key, ctrl_data)
        # plot_yaw_for_trajectory(traj_key, ctrl_data)
        # plot_control_effort_for_trajectory(traj_key, ctrl_data)

    print("[OK] All individual trajectory plots generated.")


if __name__ == "__main__":
    main()