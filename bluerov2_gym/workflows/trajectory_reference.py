"""Analytic reference recovered from 889a766/experiments_traj/train_ppo.py."""
import math
import numpy as np


class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0
        self.speed = 0.15
        self.z_target = -0.5

    def get_state_at_time(self, t):
        t_s = t * self.speed

        x = self.radius * math.sin(t_s)
        y = self.radius * math.sin(t_s) * math.cos(t_s)

        if t < 10.0:
            z = (self.z_target / 10.0) * t
        else:
            z = self.z_target

        vx = self.radius * math.cos(t_s) * self.speed
        vy = self.radius * (math.cos(t_s) ** 2 - math.sin(t_s) ** 2) * self.speed
        vz = 0.0

        yaw = math.atan2(vy, vx)

        return (
            np.array([x, y, z], dtype=np.float32),
            np.array([0.0, 0.0, yaw], dtype=np.float32),
            np.array([vx, vy, vz], dtype=np.float32),
        )


    def get_reference(self, t):
        position, attitude, velocity = self.get_state_at_time(t)
        reference = np.zeros(12, dtype=np.float32)
        reference[:3] = position
        reference[3:6] = attitude
        reference[6:9] = velocity
        return reference

