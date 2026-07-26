import numpy as np

class FaultEntry:
    def __init__(self, fault, thruster:int):
        self.fault=fault
        self.thruster=thruster

class FaultManager:
    def __init__(self):
        self.entries=[]

    def clear(self):
        self.entries.clear()

    def add_fault(self, fault, thruster:int):
        self.entries.append(FaultEntry(fault,thruster))

    def reset(self):
        for e in self.entries:
            e.fault.reset()

    def apply(self, action: np.ndarray)->np.ndarray:
        out=action.copy()
        for e in self.entries:
            out=e.fault.apply(out,e.thruster)
        return out
