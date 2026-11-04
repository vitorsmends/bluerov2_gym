from __future__ import annotations

import numpy as np


class StationKeepingReference:
    def __init__(self, position, yaw_rad):
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.yaw_rad = float(yaw_rad)

    def get_reference(self, t: float) -> np.ndarray:
        ref = np.zeros(12, dtype=float)
        ref[0:3] = self.position
        ref[5] = self.yaw_rad
        return ref
