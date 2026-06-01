"""Common execution loop for BlueROV2 station-keeping experiments."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from env_utils import compute_errors, make_env, obs_to_state, write_csv


CSV_HEADER = [
    "time",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "x_ref",
    "y_ref",
    "z_ref",
    "yaw_ref",
    "position_error_m",
    "velocity_error",
    "yaw_error_rad",
    "attitude_error",
    "reward",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
]


def run_stationkeeping_experiment(
    controller,
    reference,
    output_csv: str | Path,
    steps: int = 800,
    dt: float = 0.1,
    render_mode: str | None = None,
):
    """Run a station-keeping experiment and save the log to CSV."""
    env = make_env(render_mode=render_mode)
    obs, _ = env.reset()

    if hasattr(controller, "reset"):
        controller.reset()

    rows = []
    start = time.time()

    for k in range(steps):
        t = k * dt
        state = obs_to_state(obs)
        reference_state = reference.get_reference(t)

        action = controller.get_action(state=state, reference=reference_state, t=t)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        obs, reward, terminated, truncated, info = env.step(action)

        errors = compute_errors(state, reference_state)

        rows.append(
            [
                t,
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
                state[5],
                state[6],
                state[7],
                state[8],
                state[9],
                state[10],
                state[11],
                reference_state[0],
                reference_state[1],
                reference_state[2],
                reference_state[5],
                errors["position_error"],
                errors["velocity_error"],
                errors["yaw_error"],
                errors["attitude_error"],
                reward,
                action[0],
                action[1],
                action[2],
                action[3],
                action[4],
                action[5],
            ]
        )

        if k % 50 == 0:
            print(
                f"[{controller.__class__.__name__}] "
                f"step={k:04d}/{steps} | t={t:5.1f}s | "
                f"error={errors['position_error']:.3f} m | reward={reward:.3f}"
            )

        if terminated or truncated:
            print(f"[INFO] Episode finished at t={t:.1f}s")
            break

    write_csv(output_csv, CSV_HEADER, rows)

    if hasattr(controller, "close"):
        controller.close()
    env.close()

    elapsed = time.time() - start
    print(f"[INFO] Saved results to: {output_csv}")
    print(f"[INFO] Elapsed time: {elapsed:.2f} s")
