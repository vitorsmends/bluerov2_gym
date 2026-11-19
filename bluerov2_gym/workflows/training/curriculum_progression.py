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
