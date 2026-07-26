import numpy as np
from .base import ActuatorFault

class StuckThruster(ActuatorFault):
    def __init__(self,stuck_value:float):
        self.stuck_value=float(stuck_value)

    def apply(self,action:np.ndarray,thruster:int)->np.ndarray:
        out=action.copy()
        out[thruster]=self.stuck_value
        return out
