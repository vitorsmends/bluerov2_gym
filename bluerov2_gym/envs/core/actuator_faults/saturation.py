import numpy as np
from .base import ActuatorFault

class ReducedSaturation(ActuatorFault):
    def __init__(self,limit:float):
        self.limit=abs(float(limit))

    def apply(self,action:np.ndarray,thruster:int)->np.ndarray:
        out=action.copy()
        out[thruster]=np.clip(out[thruster],-self.limit,self.limit)
        return out
