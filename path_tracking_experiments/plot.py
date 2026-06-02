import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = "results/path_tracking"
OUTPUT_DIR = "plots_path_tracking"
TABLE_DIR = "tables_path_tracking"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


CONFIG = {
    "PID": {"file": "pid.csv", "ls": "-"},
    "MPC": {"file": "mpc.csv", "ls": "--"},
    "PPO": {"file": "ppo.csv", "ls": "-."},
}


def save_plot(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {name}")


def load_results():
    dfs = {}

    for ctrl, cfg in CONFIG.items():
        path = os.path.join(INPUT_DIR, cfg["file"])

        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        required = {"time", "x", "y", "z", "x_ref", "y_ref", "z_ref"}
        missing = required - set(df.columns)

        if missing:
            print(f"[WARN] {ctrl} skipped. Missing columns: {missing}")
            continue

        if "tracking_error_m" not in df.columns:
            df["tracking_error_m"] = np.linalg.norm(
                df[["x", "y", "z"]].values - df[["x_ref", "y_ref", "z_ref"]].values,
                axis=1,
            )

        thruster_cols = [f"T{i}" for i in range(1, 7)]
        if all(c in df.columns for c in thruster_cols):
            u = df[thruster_cols].values
            df["control_effort"] = np.sum(u**2, axis=1)
            df["control_effort_normalized"] = np.mean((u / 40.0) ** 2, axis=1)
            df["cumulative_effort"] = np.cumsum(df["control_effort"].values) * np.mean(np.diff(df["time"].values))
        else:
            df["control_effort"] = np.nan
            df["control_effort_normalized"] = np.nan
            df["cumulative_effort"] = np.nan

        dfs[ctrl] = df

    return dfs


def build_summary(dfs):
    rows = []

    for ctrl, df in dfs.items():
        err = df["tracking_error_m"].values
        effort = df["control_effort"].values

        rows.append({
            "controller": ctrl,
            "duration_s": df["time"].max(),
            "n_steps": len(df),
            "mean_error_m": np.mean(err),
            "rmse_error_m": np.sqrt(np.mean(err**2)),
            "max_error_m": np.max(err),
            "p95_error_m": np.percentile(err, 95),
            "final_error_m": err[-1],
            "mean_control_effort": np.nanmean(effort),
            "total_control_effort": np.nansum(effort),
            "mean_normalized_effort": np.nanmean(df["control_effort_normalized"].values),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(TABLE_DIR, "summary_path_tracking.csv"), index=False)
    print(summary.round(4).to_string(index=False))
    return summary


def plot_xy(dfs):
    plt.figure(figsize=(6.5, 4.2))

    first = next(iter(dfs.values()))
    plt.plot(first["x_ref"], first["y_ref"], "k--", linewidth=2.0, label="Reference")

    for ctrl, df in dfs.items():
        plt.plot(df["x"], df["y"], linestyle=CONFIG[ctrl]["ls"], linewidth=1.8, label=ctrl)

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("path_tracking_xy")


def plot_3d(dfs):
    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    first = next(iter(dfs.values()))
    ax.plot(first["x_ref"], first["y_ref"], first["z_ref"], "k--", linewidth=2.0, label="Reference")

    for ctrl, df in dfs.items():
        ax.plot(df["x"], df["y"], df["z"], linestyle=CONFIG[ctrl]["ls"], linewidth=1.6, label=ctrl)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.invert_zaxis()
    ax.legend(frameon=True)
    plt.tight_layout()
    save_plot("path_tracking_3d")


def plot_error(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for ctrl, df in dfs.items():
        plt.plot(df["time"], df["tracking_error_m"], linestyle=CONFIG[ctrl]["ls"], linewidth=1.8, label=ctrl)

    plt.xlabel("Time [s]")
    plt.ylabel("Euclidean Tracking Error [m]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("path_tracking_error")


def plot_control_effort(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for ctrl, df in dfs.items():
        plt.plot(df["time"], df["control_effort"], linestyle=CONFIG[ctrl]["ls"], linewidth=1.8, label=ctrl)

    plt.xlabel("Time [s]")
    plt.ylabel(r"Control Effort $\sum_i T_i^2$ [N$^2$]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("control_effort")


def plot_cumulative_effort(dfs):
    plt.figure(figsize=(6.5, 4.2))

    for ctrl, df in dfs.items():
        plt.plot(df["time"], df["cumulative_effort"], linestyle=CONFIG[ctrl]["ls"], linewidth=1.8, label=ctrl)

    plt.xlabel("Time [s]")
    plt.ylabel(r"Cumulative Effort $\int \sum_i T_i^2 dt$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("cumulative_control_effort")


def plot_thrusters(dfs):
    thruster_cols = [f"T{i}" for i in range(1, 7)]

    for ctrl, df in dfs.items():
        if not all(c in df.columns for c in thruster_cols):
            continue

        plt.figure(figsize=(6.5, 4.2))

        for col in thruster_cols:
            plt.plot(df["time"], df[col], linewidth=1.3, label=col)

        plt.xlabel("Time [s]")
        plt.ylabel("Thruster Command [N]")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, ncol=3)
        plt.tight_layout()
        save_plot(f"thrusters_{ctrl.lower()}")


def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    dfs = load_results()

    if not dfs:
        print("[ERROR] No valid CSV files found.")
        return

    build_summary(dfs)

    plot_xy(dfs)
    plot_3d(dfs)
    plot_error(dfs)
    plot_control_effort(dfs)
    plot_cumulative_effort(dfs)
    plot_thrusters(dfs)

    print("[OK] All plots generated.")


if __name__ == "__main__":
    main()
