"""Optional body-wrench policy interface shared by training and evaluation."""
import numpy as np
from gymnasium import spaces


class PolicyActionAllocation:
    """Five controlled body axes; pitch remains coupled to the physical vehicle.

    The policy emits [-1, 1] commands, scaled to N/Nm. The known allocation
    matrix maps minimum-norm thrust back to the simulator's six thrusters.
    A common scale factor preserves the selected wrench direction at saturation.
    """
    axes = [0, 1, 2, 3, 5]

    def __init__(self, dynamics, settings=None):
        self.settings = settings or {}
        self.mode = self.settings.get('mode', 'thrusters')
        if self.mode not in {'thrusters', 'body_wrench'}:
            raise ValueError('action_control.mode must be thrusters or body_wrench.')
        self.enabled = self.mode == 'body_wrench'
        if not self.enabled:
            return
        self.limits = np.asarray(self.settings.get('wrench_limits', [80., 80., 60., 8., 20.]), dtype=float)
        if self.limits.shape != (5,) or not np.all(np.isfinite(self.limits)) or np.any(self.limits <= 0):
            raise ValueError('wrench_limits must contain five positive finite N/Nm limits: X,Y,Z,K,N.')
        self.matrix = np.asarray(dynamics.allocation_matrix, dtype=float)
        controlled = self.matrix[self.axes]
        if controlled.shape != (5, 6) or np.linalg.matrix_rank(controlled) != 5:
            raise ValueError('Allocation requires full rank on X,Y,Z,K,N with six thrusters.')
        self.inverse = np.linalg.pinv(controlled)
        self.low, self.high = dynamics.thruster_min, dynamics.thruster_max
        if not self.low < 0 < self.high:
            raise ValueError('Allocation requires reversible thrusters.')
        self.action_space = spaces.Box(-1., 1., shape=(5,), dtype=np.float32)

    def convert(self, action):
        if not self.enabled:
            return action, {}
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.all(np.isfinite(action)):
            raise ValueError('Body-wrench action must contain five finite values.')
        desired = np.clip(action, -1., 1.) * self.limits
        thrust = self.inverse @ desired
        ratio = max(1., float(np.max(thrust / self.high)), float(np.max(thrust / self.low)))
        thrust = np.clip(thrust / ratio, self.low, self.high).astype(np.float32)
        actual = self.matrix @ thrust
        return thrust, {'allocation/scale': 1. / ratio,
                        'allocation/requested_wrench': desired.copy(),
                        'allocation/achieved_wrench': actual.copy()}
