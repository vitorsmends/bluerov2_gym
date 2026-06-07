from __future__ import annotations

import json
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from dynamics import Dynamics


def bryson_diag(max_values, scale=1.0):
    max_values = np.asarray(max_values, dtype=float)
    return np.diag(scale / (max_values**2))


def dict_to_vector(state):
    return np.array([
        state["x"], state["y"], state["z"],
        state["roll"], state["pitch"], state["yaw"],
        state["u"], state["v"], state["w"],
        state["p"], state["q"], state["r"],
    ], dtype=float)


def vector_to_dict(x):
    x = np.asarray(x, dtype=float).reshape(12)
    return {
        "x": float(x[0]), "y": float(x[1]), "z": float(x[2]),
        "roll": float(x[3]), "pitch": float(x[4]), "yaw": float(x[5]),
        "u": float(x[6]), "v": float(x[7]), "w": float(x[8]),
        "p": float(x[9]), "q": float(x[10]), "r": float(x[11]),
    }


def disable_disturbance(dyn):
    def zero_current():
        return np.zeros(3, dtype=float)

    dyn._jonswap_current = zero_current


def step_discrete(dyn, x, u):
    state = vector_to_dict(x.copy())
    next_state = dyn.step(state, u.copy())
    return dict_to_vector(next_state)


def find_trim_input(dyn, x_eq):
    """Find constant thruster input that keeps the vehicle close to equilibrium."""

    def objective(u):
        x_next = step_discrete(dyn, x_eq, u)
        dx = x_next - x_eq

        # Prioritize velocity/attitude drift, not absolute position integration.
        return (
            100.0 * np.sum(dx[6:12] ** 2)
            + 10.0 * np.sum(dx[3:6] ** 2)
            + 1e-3 * np.sum(u ** 2)
        )

    result = minimize(
        objective,
        x0=np.zeros(6),
        method="SLSQP",
        bounds=[(-40.0, 40.0)] * 6,
        options={"ftol": 1e-10, "maxiter": 300, "disp": False},
    )

    if not result.success:
        print("[WARN] Trim optimization failed. Using best available solution.")

    return np.clip(result.x, -40.0, 40.0)


def numerical_linearization(dyn, x_eq, u_eq, eps_x=1e-5, eps_u=1e-4):
    n = 12
    m = 6

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps_x

        fp = step_discrete(dyn, x_eq + dx, u_eq)
        fm = step_discrete(dyn, x_eq - dx, u_eq)

        A[:, i] = (fp - fm) / (2.0 * eps_x)

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps_u

        fp = step_discrete(dyn, x_eq, u_eq + du)
        fm = step_discrete(dyn, x_eq, u_eq - du)

        B[:, j] = (fp - fm) / (2.0 * eps_u)

    return A, B


def controllability_rank(A, B):
    n = A.shape[0]
    C = B

    Ak = np.eye(n)
    blocks = []

    for _ in range(n):
        blocks.append(Ak @ B)
        Ak = Ak @ A

    C = np.hstack(blocks)
    return np.linalg.matrix_rank(C), C.shape[0]


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    eig_cl = np.linalg.eigvals(A - B @ K)
    return K, P, eig_cl


def main():
    dyn = Dynamics()
    dyn.reset()
    disable_disturbance(dyn)

    dt = dyn.dt

    x_eq = np.zeros(12, dtype=float)

    # Trim is important because u_eq = 0 may not be a true equilibrium.
    u_eq = find_trim_input(dyn, x_eq)

    A_d, B_d = numerical_linearization(
        dyn=dyn,
        x_eq=x_eq,
        u_eq=u_eq,
        eps_x=1e-5,
        eps_u=1e-4,
    )

    # Less aggressive than your previous design.
    state_max = np.array([
        0.50,                 # x [m]
        0.50,                 # y [m]
        0.40,                 # z [m]
        np.deg2rad(20.0),     # roll [rad]
        np.deg2rad(20.0),     # pitch [rad]
        np.deg2rad(25.0),     # yaw [rad]
        0.50,                 # u [m/s]
        0.50,                 # v [m/s]
        0.40,                 # w [m/s]
        np.deg2rad(35.0),     # p [rad/s]
        np.deg2rad(35.0),     # q [rad/s]
        np.deg2rad(40.0),     # r [rad/s]
    ])

    input_max = np.ones(6) * 40.0

    Q = bryson_diag(state_max, scale=1.0)

    # Larger R = less aggressive thruster commands.
    R = bryson_diag(input_max, scale=2.0)

    rank, n = controllability_rank(A_d, B_d)

    K, P, eig_cl = dlqr(A_d, B_d, Q, R)

    np.set_printoptions(precision=6, suppress=True)

    print("\n" + "=" * 80)
    print("DISCRETE LQR CONTROL DESIGN REPORT")
    print("=" * 80)

    print(f"\nSampling time dt: {dt:.3f} s")

    print("\nEquilibrium state x_eq:")
    print(x_eq)

    print("\nTrim input u_eq:")
    print(u_eq)

    print("\nControllability:")
    print(f"  rank = {rank} / {n}")

    print("\nOpen-loop eigenvalues:")
    print(np.linalg.eigvals(A_d))

    print("\nClosed-loop eigenvalues:")
    print(eig_cl)

    print("\nMax |closed-loop eigenvalue|:")
    print(np.max(np.abs(eig_cl)))

    print("\nQ diagonal:")
    print(np.diag(Q))

    print("\nR diagonal:")
    print(np.diag(R))

    print("\nLQR gain K:")
    print(K)

    if np.max(np.abs(eig_cl)) < 1.0:
        print("\n[OK] Discrete closed-loop system is locally stable.")
    else:
        print("\n[WARN] Closed-loop eigenvalues are outside the unit circle.")

    print("=" * 80 + "\n")

    output = {
        "dt": dt,
        "x_eq": x_eq.tolist(),
        "u_eq": u_eq.tolist(),
        "A_d": A_d.tolist(),
        "B_d": B_d.tolist(),
        "Q_diag": np.diag(Q).tolist(),
        "R_diag": np.diag(R).tolist(),
        "K": K.tolist(),
        "open_loop_eigenvalues_real": np.real(np.linalg.eigvals(A_d)).tolist(),
        "open_loop_eigenvalues_imag": np.imag(np.linalg.eigvals(A_d)).tolist(),
        "closed_loop_eigenvalues_real": np.real(eig_cl).tolist(),
        "closed_loop_eigenvalues_imag": np.imag(eig_cl).tolist(),
        "max_abs_closed_loop_eigenvalue": float(np.max(np.abs(eig_cl))),
        "controllability_rank": int(rank),
        "controllability_dimension": int(n),
    }

    with open("lqr_design_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("Saved: lqr_design_output.json")


if __name__ == "__main__":
    main()