import numpy as np
from .base import ActuatorFault

class LossOfEffectiveness(ActuatorFault):
    def __init__(self,effectiveness:float):
        self.effectiveness=float(effectiveness)

    def apply(self,action:np.ndarray,thruster:int)->np.ndarray:
        out=action.copy()
        out[thruster]*=self.effectiveness
        return out
