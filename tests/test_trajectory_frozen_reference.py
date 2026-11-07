"""The archived recipe must agree with the migrated reference on current physics."""
import hashlib
from pathlib import Path

import numpy as np

from bluerov2_gym.workflows.configuration import resolve
from bluerov2_gym.workflows.training.reference_sources import trajectory_889a766
from bluerov2_gym.workflows.training.trajectory_env import TrajectoryTrackingEnv
from bluerov2_gym.workflows.training.trajectory_frozen_env import (
    Frozen889TrajectoryEnv, SOURCE_SHA256,
)


def test_historical_source_is_unchanged():
    assert hashlib.sha256(Path(trajectory_889a766.__file__).read_bytes()).hexdigest() == SOURCE_SHA256
    assert Frozen889TrajectoryEnv.reset is trajectory_889a766.TrajectoryTrackingEnv.reset


def test_frozen_and_migrated_recipes_agree_on_current_physics():
    root = Path(__file__).resolve().parents[1]
    _, config = resolve(root / 'configs/training/trajectory_889_frozen.yaml')
    kwargs = dict(env_config=config['environment']['config'],
                  dynamics_config=config['environment']['dynamics_config'])
    original_rng = np.random.get_state()
    environments = [Frozen889TrajectoryEnv(**kwargs), TrajectoryTrackingEnv(**kwargs)]
    try:
        traces = []
        actions = np.random.default_rng(17).uniform(-45, 45, size=(32, 6))
        for env in environments:
            np.random.seed(42)
            obs, _ = env.reset(seed=42)
            trace = [(obs, env.current_t)]
            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                trace.append((obs, reward, terminated, truncated, info['tracking_error_m']))
                if terminated or truncated:
                    break
            traces.append(trace)
        assert len(traces[0]) == len(traces[1])
        for frozen, migrated in zip(*traces):
            for key in frozen[0]:
                np.testing.assert_array_equal(frozen[0][key], migrated[0][key])
            assert frozen[1:] == migrated[1:]
    finally:
        np.random.set_state(original_rng)
        for env in environments:
            env.close()
