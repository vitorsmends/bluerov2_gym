"""Base controller interface for station-keeping experiments."""

from __future__ import annotations

import numpy as np


class BaseController:
    """Minimal controller interface.

    Controllers must return direct thruster commands:

        [T1, T2, T3, T4, T5, T6]

    in Newtons.
    """

    def reset(self) -> None:
        """Reset controller internal states, if any."""
        pass

    def get_action(self, state: np.ndarray, reference: np.ndarray, t: float) -> np.ndarray:
        """Return a 6-dimensional thruster command."""
        raise NotImplementedError
