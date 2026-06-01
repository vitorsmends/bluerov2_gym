"""Base controller interface for path-tracking experiments."""

from __future__ import annotations

import numpy as np


class BaseController:
    """Common interface implemented by all controllers."""

    name = "base"

    def reset(self):
        """Reset internal controller state before a new experiment."""
        pass

    def get_action(self, obs: dict, state: np.ndarray, reference: np.ndarray, t: float) -> np.ndarray:
        """Return direct thruster commands [T1, ..., T6] in Newtons."""
        raise NotImplementedError
