from __future__ import annotations

import json
import numpy as np

from dynamics import Dynamics

def main():

    M_diag = np.diag(Dynamics.M)
    D_lin = Dynamics.D_lin

    zeta = np.array([1.2, 1.2, 1.3, 1.4, 1.4, 1.2])
    wn = np.array([0.45, 0.45, 0.50, 0.55, 0.55, 0.50])

    kp = M_diag * wn**2

    kd = 2.0 * zeta * wn * M_diag - D_lin
    kd = np.maximum(kd, 0.05 * D_lin)

    ki = 0.03 * kp
    ki[3] = 0.0
    ki[4] = 0.0

    output = {
        "dt": Dynamics.dt,
        "M_diag": M_diag.tolist(),
        "D_lin": D_lin.tolist(),
        "zeta": zeta.tolist(),
        "wn": wn.tolist(),
        "kp": kp.tolist(),
        "ki": ki.tolist(),
        "kd": kd.tolist(),
        "thruster_limit": Dynamics.thruster_max,
        "max_delta_thrust": 8.0,
        "steps_recommended": 1000,
        "simulation_time_s": 1000 * Dynamics.dt,
    }

    print(json.dumps(output, indent=4))

    with open("pid_design_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\nSaved: pid_design_output.json")


if __name__ == "__main__":
    main()