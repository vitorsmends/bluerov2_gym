from abc import ABC, abstractmethod
import numpy as np

class ActuatorFault(ABC):
    def reset(self):
        pass

    @abstractmethod
    def apply(self, action: np.ndarray, thruster: int) -> np.ndarray:
        ...
