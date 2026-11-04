from __future__ import annotations

import numpy as np

from .env_utils import wrap_angle

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

        yaw = (
            np.arctan2(v_world[1], v_world[0])
            if np.linalg.norm(v_world[:2]) > 1e-6
            else 0.0
        )

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

        return (
            (1.0 - local) * self.points[segment]
            + local * self.points[segment + 1]
        )


class StarTrajectory(BaseTrajectory):
    name = "star"

    def __init__(
        self,
        radius_outer=1.2,
        radius_inner=0.45,
        depth=-0.6,
        period=100.0,
    ):
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

        return (
            (1.0 - local) * self.points[segment]
            + local * self.points[segment + 1]
        )


class LetterBTrajectory(BaseTrajectory):
    name = "letter_b"

    def __init__(self, scale=1.0, depth=-0.6, period=100.0):
        self.depth = depth
        self.period = period

        points = []

        x_stem = -0.6 * scale
        y_bottom = -1.0 * scale
        y_mid = 0.0
        y_top = 1.0 * scale

        for y in np.linspace(y_bottom, y_top, 40):
            points.append([x_stem, y, depth])

        cx = -0.05 * scale
        cy = 0.5 * scale
        rx = 0.65 * scale
        ry = 0.5 * scale

        theta = np.linspace(np.pi / 2.0, -np.pi / 2.0, 60)

        for th in theta:
            x = cx + rx * np.cos(th)
            y = cy + ry * np.sin(th)
            points.append([x, y, depth])

        points.append([x_stem, y_mid, depth])

        cx = -0.02 * scale
        cy = -0.5 * scale
        rx = 0.72 * scale
        ry = 0.55 * scale

        theta = np.linspace(np.pi / 2.0, -np.pi / 2.0, 70)

        for th in theta:
            x = cx + rx * np.cos(th)
            y = cy + ry * np.sin(th)
            points.append([x, y, depth])

        points.append([x_stem, y_bottom, depth])

        self.points = np.array(points, dtype=float)

    def position(self, t):
        phase = (t % self.period) / self.period
        n_seg = len(self.points) - 1
        segment = min(int(phase * n_seg), n_seg - 1)
        local = phase * n_seg - segment

        return (
            (1.0 - local) * self.points[segment]
            + local * self.points[segment + 1]
        )


TRAJECTORY_TYPES = {
    "circle": CircleTrajectory,
    "figure_eight_depth": FigureEightDepthTrajectory,
    "square": SquareTrajectory,
    "star": StarTrajectory,
    "letter_b": LetterBTrajectory,
}

def create_trajectory(name: str, params: dict | None = None):
    try:
        trajectory_cls = TRAJECTORY_TYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown trajectory: {name}") from exc
    return trajectory_cls(**(params or {}))
