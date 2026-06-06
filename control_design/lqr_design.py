"""
LQR control design for BlueROV2 6-DoF dynamics.

This script:
    1. Defines an equilibrium point.
    2. Numerically linearizes the full nonlinear model.
    3. Builds A and B matrices.
    4. Designs an LQR controller.
    5. Prints and saves the controller data.

State:
    x = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]

Input:
    u = [T1, T2, T3, T4, T5, T6]
"""

from __future__ import annotations

import copy
import json
import numpy as np
from scipy.linalg import solve_continuous_are


# Import your Dynamics class here
from dynamics import Dynamics


def wrap_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def bryson_diag(max_values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    max_values = np.asarray(max_values, dtype=float)

    if np.any(max_values <= 0.0):
        raise ValueError("All maximum values must be positive.")

    return np.diag(scale / (max_values ** 2))


def dict_to_vector(state: dict) -> np.ndarray:
    return np.array(
        [
            state["x"],
            state["y"],
            state["z"],
            state["roll"],
            state["pitch"],
            state["yaw"],
            state["u"],
            state["v"],
            state["w"],
            state["p"],
            state["q"],
            state["r"],
        ],
        dtype=float,
    )


def vector_to_dict(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).reshape(12)

    return {
        "x": float(x[0]),
        "y": float(x[1]),
        "z": float(x[2]),
        "roll": float(x[3]),
        "pitch": float(x[4]),
        "yaw": float(x[5]),
        "u": float(x[6]),
        "v": float(x[7]),
        "w": float(x[8]),
        "p": float(x[9]),
        "q": float(x[10]),
        "r": float(x[11]),
    }


def dynamics_discrete_step(dyn, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Uses a copy of the dynamics object to avoid advancing the real JONSWAP state.
    """

    dyn_local = copy.deepcopy(dyn)

    state = vector_to_dict(x)
    next_state = dyn_local.step(state, u)

    return dict_to_vector(next_state)


def discrete_to_continuous(A_d: np.ndarray, B_d: np.ndarray, dt: float):
    """
    First-order approximation:
        A_c = (A_d - I) / dt
        B_c = B_d / dt

    This is sufficient for initial LQR design.
    """

    n = A_d.shape[0]

    A_c = (A_d - np.eye(n)) / dt
    B_c = B_d / dt

    return A_c, B_c


def numerical_linearization(
    dyn,
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    eps_x: float = 1e-5,
    eps_u: float = 1e-4,
):
    """
    Numerically linearizes the discrete nonlinear dynamics:

        x[k+1] = f(x[k], u[k])

    around (x_eq, u_eq).

    Returns:
        A_d, B_d
    """

    x_eq = np.asarray(x_eq, dtype=float).reshape(12)
    u_eq = np.asarray(u_eq, dtype=float).reshape(6)

    n = x_eq.size
    m = u_eq.size

    A_d = np.zeros((n, n), dtype=float)
    B_d = np.zeros((n, m), dtype=float)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps_x

        x_plus = x_eq + dx
        x_minus = x_eq - dx

        f_plus = dynamics_discrete_step(dyn, x_plus, u_eq)
        f_minus = dynamics_discrete_step(dyn, x_minus, u_eq)

        derivative = (f_plus - f_minus) / (2.0 * eps_x)

        A_d[:, i] = derivative

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps_u

        u_plus = u_eq + du
        u_minus = u_eq - du

        f_plus = dynamics_discrete_step(dyn, x_eq, u_plus)
        f_minus = dynamics_discrete_step(dyn, x_eq, u_minus)

        derivative = (f_plus - f_minus) / (2.0 * eps_u)

        B_d[:, j] = derivative

    return A_d, B_d


def design_lqr(A_c: np.ndarray, B_c: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """
    Continuous-time LQR:

        u = -K x

    Solves:
        A.T P + P A - P B R^-1 B.T P + Q = 0

    Then:
        K = R^-1 B.T P
    """

    P = solve_continuous_are(A_c, B_c, Q, R)
    K = np.linalg.solve(R, B_c.T @ P)

    eig_closed_loop = np.linalg.eigvals(A_c - B_c @ K)

    return K, P, eig_closed_loop


def main():
    # ------------------------------------------------------------
    # 1. Load your full nonlinear dynamics
    # ------------------------------------------------------------
    dyn = Dynamics()

    dt = dyn.dt

    # Important:
    # For pure control design, reset the wave state.
    # Later we can linearize under different current operating points.
    dyn.reset()

    # ------------------------------------------------------------
    # 2. Define equilibrium point
    # ------------------------------------------------------------
    x_eq = np.zeros(12, dtype=float)

    # Example:
    # position = 0
    # attitude = level
    # body velocities = 0
    #
    # If your z convention is positive downward/upward, adjust here.
    x_eq[0] = 0.0       # x
    x_eq[1] = 0.0       # y
    x_eq[2] = 0.0       # z
    x_eq[3] = 0.0       # roll
    x_eq[4] = 0.0       # pitch
    x_eq[5] = 0.0       # yaw
    x_eq[6:12] = 0.0    # body velocities

    # Initial equilibrium thrust.
    # For a perfectly neutrally buoyant ROV, this can start at zero.
    # If the vehicle slowly sinks/rises, we later compute a trim input.
    u_eq = np.zeros(6, dtype=float)

    # ------------------------------------------------------------
    # 3. Numerical linearization
    # ------------------------------------------------------------
    A_d, B_d = numerical_linearization(
        dyn=dyn,
        x_eq=x_eq,
        u_eq=u_eq,
        eps_x=1e-5,
        eps_u=1e-4,
    )

    A_c, B_c = discrete_to_continuous(A_d, B_d, dt)

    # ------------------------------------------------------------
    # 4. LQR weighting design
    # ------------------------------------------------------------
    state_max = np.array(
        [
            0.25,                 # x error [m]
            0.25,                 # y error [m]
            0.20,                 # z error [m]
            np.deg2rad(10.0),     # roll error [rad]
            np.deg2rad(10.0),     # pitch error [rad]
            np.deg2rad(15.0),     # yaw error [rad]
            0.30,                 # u velocity error [m/s]
            0.30,                 # v velocity error [m/s]
            0.20,                 # w velocity error [m/s]
            np.deg2rad(20.0),     # p rate error [rad/s]
            np.deg2rad(20.0),     # q rate error [rad/s]
            np.deg2rad(25.0),     # r rate error [rad/s]
        ],
        dtype=float,
    )

    input_max = np.ones(6) * 40.0

    Q = bryson_diag(state_max, scale=1.0)
    R = bryson_diag(input_max, scale=0.05)

    # ------------------------------------------------------------
    # 5. LQR design
    # ------------------------------------------------------------
    K, P, eig_cl = design_lqr(A_c, B_c, Q, R)

    # ------------------------------------------------------------
    # 6. Print report
    # ------------------------------------------------------------
    np.set_printoptions(precision=6, suppress=True)

    print("\n" + "=" * 80)
    print("LQR CONTROL DESIGN REPORT")
    print("=" * 80)

    print(f"\nSampling time: {dt:.3f} s")

    print("\nEquilibrium state x_eq:")
    print(x_eq)

    print("\nEquilibrium input u_eq:")
    print(u_eq)

    print("\nDiscrete-time A_d matrix:")
    print(A_d)

    print("\nDiscrete-time B_d matrix:")
    print(B_d)

    print("\nContinuous-time A_c matrix:")
    print(A_c)

    print("\nContinuous-time B_c matrix:")
    print(B_c)

    print("\nQ diagonal:")
    print(np.diag(Q))

    print("\nR diagonal:")
    print(np.diag(R))

    print("\nLQR gain K:")
    print(K)

    print("\nClosed-loop eigenvalues:")
    print(eig_cl)

    print("=" * 80 + "\n")

    # ------------------------------------------------------------
    # 7. Save data
    # ------------------------------------------------------------
    output = {
        "dt": dt,
        "x_eq": x_eq.tolist(),
        "u_eq": u_eq.tolist(),
        "A_d": A_d.tolist(),
        "B_d": B_d.tolist(),
        "A_c": A_c.tolist(),
        "B_c": B_c.tolist(),
        "Q_diag": np.diag(Q).tolist(),
        "R_diag": np.diag(R).tolist(),
        "K": K.tolist(),
        "closed_loop_eigenvalues_real": np.real(eig_cl).tolist(),
        "closed_loop_eigenvalues_imag": np.imag(eig_cl).tolist(),
    }

    with open("lqr_design_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("Saved: lqr_design_output.json")


if __name__ == "__main__":
    main()