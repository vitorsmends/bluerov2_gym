"""Keep stable mastery distinct from advancement for harder-wave exposure."""


def credit_window(streak, passed, completed_evaluations, settings):
    size = settings.get('rolling_batches', 1)
    if settings.get('nonoverlapping_windows', False) and completed_evaluations % size:
        return streak
    return streak + 1 if passed else 0


def plateau_advances(settings):
    return settings.get('on_plateau', 'stop') == 'advance'


def completion_status(history):
    return 'completed' if all(s['status'] == 'mastered' for s in history) else 'exposure_completed'


def early_skip_reason(status, settings):
    """Optional exposure shortcut; protect acquisition and genuine mastery progress."""
    if not settings.get('enabled', False):
        return None
    if status.get('index', 0) < settings.get('first_stage', 5):
        return None
    if status.get('evaluations', 0) < settings.get('min_evaluations', 15):
        return None
    if status.get('successful_batches', 0) or status.get('last_evaluation', {}).get('passed'):
        return None
    progress = status.get('progress_monitor', {})
    if not progress.get('active', False):
        return None
    if progress.get('stale', 0) < settings.get('patience_evaluations', 10):
        return None
    import math
    rmse = progress.get('current', {}).get('rmse')
    if rmse is None or not math.isfinite(rmse):
        return None
    return (f'Early exposure skipping after {progress["stale"]} checks without meaningful '
            'progress; acquisition/transfer phases protected; mastery NOT satisfied.')
