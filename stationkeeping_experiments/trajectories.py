"""Reference trajectories for BlueROV2 experiments."""

from __future__ import annotations

import numpy as np


class StationKeepingReference:
    """Constant reference for station-keeping experiments.

    The returned reference follows the 12-state convention used by the
    BlueROV2 environment:

        [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = -0.5,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        self.reference = np.array(
            [
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    def get_reference(self, t: float) -> np.ndarray:
        """Return the reference state at time ``t``."""
        return self.reference.copy()
