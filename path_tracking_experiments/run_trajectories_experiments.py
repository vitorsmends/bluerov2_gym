from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from env_utils import make_env, obs_to_state, tracking_errors
from env_utils import wrap_angle

from smc_controller import SMCController
from pid_controller import PIDController
from nmpc_controller import NMPCController
from ppo_controller import PPOController


OUTPUT_DIR = Path("results/trajectory_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.1
STEPS = 1000
CONTROLLERS = ["pid", "smc", "nmpc", "ppo"]


class BaseTrajectory:
    name = "base"

    def position(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def get_reference(self, t: float) -> np.ndarray:
        dt = 1e-3

        p = self.position(t)
        p_prev = self.position(max(0.0, t - dt))
        p_next = self.position(t + dt)

        v_world = (p_next - p_prev) / (2.0 * dt)

        yaw = np.arctan2(v_world[1], v_world[0]) if np.linalg.norm(v_world[:2]) > 1e-6 else 0.0

        ref = np.zeros(12, dtype=float)
        ref[0:3] = p
        ref[3] = 0.0
        ref[4] = 0.0
        ref[5] = wrap_angle(yaw)

        ref[6:9] = v_world
        ref[9:12] = 0.0

        return ref


class CircleTrajectory(BaseTrajectory):
    name = "circle"

    def __init__(self, radius=1.0, depth=-0.6, period=50.0):
        self.radius = radius
        self.depth = depth
        self.omega = 2.0 * np.pi / period

    def position(self, t):
        return np.array([
            self.radius * np.cos(self.omega * t),
            self.radius * np.sin(self.omega * t),
            self.depth,
        ])


class FigureEightDepthTrajectory(BaseTrajectory):
    name = "figure_eight_depth"

    def __init__(self, a=1.0, depth=-0.6, depth_amp=0.25, period=60.0):
        self.a = a
        self.depth = depth
        self.depth_amp = depth_amp
        self.omega = 2.0 * np.pi / period

    def position(self, t):
        s = self.omega * t
        return np.array([
            self.a * np.sin(s),
            self.a * np.sin(s) * np.cos(s),
            self.depth + self.depth_amp * np.sin(0.5 * s),
        ])


class SquareTrajectory(BaseTrajectory):
    name = "square"

    def __init__(self, side=1.6, depth=-0.6, period=80.0):
        self.side = side
        self.depth = depth
        self.period = period
        h = side / 2.0
        self.points = np.array([
            [-h, -h, depth],
            [ h, -h, depth],
            [ h,  h, depth],
            [-h,  h, depth],
            [-h, -h, depth],
        ])

    def position(self, t):
        phase = (t % self.period) / self.period
        segment = min(int(phase * 4), 3)
        local = phase * 4 - segment
        return (1.0 - local) * self.points[segment] + local * self.points[segment + 1]


class StarTrajectory(BaseTrajectory):
    name = "star"

    def __init__(self, radius_outer=1.2, radius_inner=0.45, depth=-0.6, period=100.0):
        self.depth = depth
        self.period = period

        points = []
        for i in range(10):
            r = radius_outer if i % 2 == 0 else radius_inner
            angle = np.pi / 2.0 + i * np.pi / 5.0
            points.append([r * np.cos(angle), r * np.sin(angle), depth])

        points.append(points[0])
        self.points = np.array(points, dtype=float)

    def position(self, t):
        phase = (t % self.period) / self.period
        n_seg = len(self.points) - 1
        segment = min(int(phase * n_seg), n_seg - 1)
        local = phase * n_seg - segment
        return (1.0 - local) * self.points[segment] + local * self.points[segment + 1]


class LetterBTrajectory(BaseTrajectory):
    name = "letter_b"

    def __init__(self, scale=1.0, depth=-0.6, period=100.0):
        self.depth = depth
        self.period = period

        points = []

        # ==========================
        # Haste vertical
        # ==========================
        x_stem = -0.6 * scale

        y_bottom = -1.0 * scale
        y_mid = 0.0
        y_top = 1.0 * scale

        for y in np.linspace(y_bottom, y_top, 40):
            points.append([x_stem, y, depth])

        # ==========================
        # Lóbulo superior
        # ==========================
        cx = -0.05 * scale
        cy = 0.5 * scale

        rx = 0.65 * scale
        ry = 0.5 * scale

        theta = np.linspace(np.pi/2, -np.pi/2, 60)

        for th in theta:
            x = cx + rx * np.cos(th)
            y = cy + ry * np.sin(th)
            points.append([x, y, depth])

        # volta para a cintura
        points.append([x_stem, y_mid, depth])

        # ==========================
        # Lóbulo inferior
        # (ligeiramente maior)
        # ==========================
        cx = -0.02 * scale
        cy = -0.5 * scale

        rx = 0.72 * scale
        ry = 0.55 * scale

        theta = np.linspace(np.pi/2, -np.pi/2, 70)

        for th in theta:
            x = cx + rx * np.cos(th)
            y = cy + ry * np.sin(th)
            points.append([x, y, depth])

        # fecha na base da haste
        points.append([x_stem, y_bottom, depth])

        self.points = np.array(points, dtype=float)

    def position(self, t):
        phase = (t % self.period) / self.period

        n_seg = len(self.points) - 1

        segment = min(
            int(phase * n_seg),
            n_seg - 1,
        )

        local = phase * n_seg - segment

        return (
            (1.0 - local) * self.points[segment]
            + local * self.points[segment + 1]
        )

def make_controller(controller_name: str, dynamics, trajectory):
    name = controller_name.lower()

    if name == "smc":
        return SMCController(dynamics=dynamics, dt=DT)

    if name == "pid":
        return PIDController(dynamics=dynamics, dt=DT)

    if name == "nmpc":
        return NMPCController(trajectory=trajectory, dynamics=dynamics, dt=DT)

    if name == "ppo":
        return PPOController()

    raise ValueError(f"Unknown controller: {controller_name}")


def run_single_trajectory(controller_name: str, trajectory):
    env = make_env(render_mode=None)
    dynamics = env.unwrapped.dynamics

    controller = make_controller(controller_name, dynamics, trajectory)

    obs, _ = env.reset()

    if hasattr(controller, "reset"):
        controller.reset()

    output_csv = OUTPUT_DIR / f"{controller.name}_{trajectory.name}.csv"

    rows = []

    print(f"\n[INFO] Running {controller.name.upper()} on trajectory: {trajectory.name}")

    for k in range(STEPS):
        t = k * DT

        state = obs_to_state(obs)
        reference = trajectory.get_reference(t)
        errors = tracking_errors(state, reference)

        action = controller.get_action(
            obs=obs,
            state=state,
            reference=reference,
            t=t,
        )

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        obs, reward, terminated, truncated, info = env.step(action)

        rows.append([
            trajectory.name,
            controller.name,
            t,

            state[0], state[1], state[2],
            state[3], state[4], state[5],
            state[6], state[7], state[8],
            state[9], state[10], state[11],

            reference[0], reference[1], reference[2],
            reference[3], reference[4], reference[5],
            reference[6], reference[7], reference[8],
            reference[9], reference[10], reference[11],

            errors["tracking_error_m"],
            errors["velocity_error"],
            errors["yaw_error"],
            reward,

            action[0], action[1], action[2],
            action[3], action[4], action[5],

            np.sum(action ** 2),
            np.mean((action / 40.0) ** 2),
        ])

        if k % 50 == 0:
            print(
                f"[{trajectory.name}] step={k:04d}/{STEPS} | "
                f"t={t:5.1f}s | error={errors['tracking_error_m']:.3f} m | "
                f"reward={reward:.3f}"
            )

        if terminated or truncated:
            print(f"[INFO] Episode finished at t={t:.1f}s")
            break

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "trajectory",
            "controller",
            "time",

            "x", "y", "z",
            "roll", "pitch", "yaw",
            "u", "v", "w",
            "p", "q", "r",

            "x_ref", "y_ref", "z_ref",
            "roll_ref", "pitch_ref", "yaw_ref",
            "u_ref", "v_ref", "w_ref",
            "p_ref", "q_ref", "r_ref",

            "tracking_error_m",
            "velocity_error",
            "yaw_error",
            "reward",

            "T1", "T2", "T3", "T4", "T5", "T6",

            "control_effort",
            "control_effort_normalized",
        ])

        writer.writerows(rows)

    env.close()

    print(f"[OK] Saved: {output_csv}")


def main():
    trajectories = [
        SquareTrajectory(),
        FigureEightDepthTrajectory(),
        CircleTrajectory(),
        LetterBTrajectory(),
        StarTrajectory(),
    ]

    for controller_name in CONTROLLERS:
        for trajectory in trajectories:
            run_single_trajectory(
                controller_name=controller_name,
                trajectory=trajectory,
            )

    print("\n[OK] All controller-trajectory experiments finished.")


if __name__ == "__main__":
    main()