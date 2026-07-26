import numpy as np
from .base import ActuatorFault

class CompleteOutage(ActuatorFault):
    def apply(self,action:np.ndarray,thruster:int)->np.ndarray:
        out=action.copy()
        out[thruster]=0.0
        return out
