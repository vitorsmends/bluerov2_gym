"""Plateau detection aligned with independent curriculum mastery assessments."""
import math


def update_progress(state, result, settings):
    """Count stagnation only after a full mastery window and local warmup.

    Each indicator keeps its own meaningful best; deterioration followed by
    recovery to an old level cannot indefinitely reset patience.
    """
    state = dict(state or {})
    window = result.get('mastery_window')
    metrics = window if window is not None else result
    if window is not None and window['batches'] < window['required_batches']:
        return state
    values = dict(success=metrics['success_rate'],
                  worst_case=min(metrics['group_success_rates'].values()),
                  rmse=metrics['mean_rmse_m'])
    best = dict(state.get('best', {}))
    changed = []
    for key, value in values.items():
        if value is None or not math.isfinite(value):
            continue
        threshold = settings.get('min_delta_m', .005) if key == 'rmse' else settings.get('min_success_delta', .025)
        improvement = (best[key] - value if key == 'rmse' else value - best[key]) if key in best else float('inf')
        if improvement > 0 and improvement >= threshold:
            best[key] = value
            changed.append(key)
    checks = state.get('checks', 0) + 1
    stale = 0 if changed or checks <= settings.get('warmup_evaluations', 5) else state.get('stale', 0) + 1
    return dict(best=best, current=values, checks=checks, stale=stale, improved=changed)
