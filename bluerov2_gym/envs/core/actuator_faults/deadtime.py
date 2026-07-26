from collections import deque
import numpy as np
from .base import ActuatorFault

class DeadTime(ActuatorFault):
    def __init__(self,delay_steps:int):
        self.delay_steps=int(delay_steps)
        self.buffer=deque()

    def reset(self):
        self.buffer.clear()

    def apply(self,action:np.ndarray,thruster:int)->np.ndarray:
        out=action.copy()
        self.buffer.append(out[thruster])
        if len(self.buffer)<=self.delay_steps:
            out[thruster]=0.0
        else:
            out[thruster]=self.buffer.popleft()
        return out
