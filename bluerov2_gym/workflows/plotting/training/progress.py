"""Progress estimates from recorded steps and TensorBoard wall timestamps."""
import math


FINISHED = {'completed', 'failed', 'interrupted', 'early_stopped'}


def training_progress(config, step, samples, started_at, now, status):
    training = config.get('training', {})
    lessons = config.get('curriculum', [])
    by_success = (training.get('task') in {'trajectory_curriculum', 'trajectory_curriculum_legacy'} or
                  config.get('curriculum_transition', {}).get('mode') == 'success')
    quantum = max(1, int(config.get('ppo', {}).get('n_steps', 1)))
    target = None
    if not by_success:
        if lessons:
            budgets = [lesson.get('timesteps') for lesson in lessons]
            if all(isinstance(n, (int, float)) and math.isfinite(n) and n > 0 for n in budgets):
                target = sum(math.ceil(n / quantum) * quantum for n in budgets)
        else:
            budget = training.get('total_timesteps')
            if isinstance(budget, (int, float)) and math.isfinite(budget) and budget > 0:
                target = math.ceil(budget / quantum) * quantum

    samples = sorted((s, t) for s, t in samples if math.isfinite(t))
    end = samples[-1][1] if status in FINISHED and samples else now
    elapsed = max(0., end - started_at) if started_at is not None else None
    speed = None
    recent = samples[-11:]
    if len(recent) >= 2:
        delta_steps = recent[-1][0] - recent[0][0]
        delta_time = recent[-1][1] - recent[0][1]
        if delta_steps > 0 and delta_time > 0:
            speed = delta_steps / delta_time
    elif recent and started_at is not None:
        duration = recent[-1][1] - started_at
        if duration > 0 and recent[-1][0] > 0:
            speed = recent[-1][0] / duration

    # A stopped writer is not evidence that the training process is still alive.
    cadence = ((recent[-1][1] - recent[0][1]) / (len(recent) - 1)
               if len(recent) > 1 else 0.)
    stale = bool(samples and now - samples[-1][1] > max(60., 4 * cadence))
    remaining = None
    if status == 'completed':
        remaining = 0.
    elif target and speed and not stale and status not in FINISHED:
        # At the last rollout PPO may still be optimizing/saving the final model.
        if step < target:
            remaining = (target - step) / speed
    percent = min(100., max(0., step / target * 100)) if target else None
    if status == 'completed' and target:
        percent = 100.
    return dict(total_steps=target, percent=percent, elapsed_seconds=elapsed,
                remaining_seconds=remaining, steps_per_second=speed,
                estimated_total_seconds=elapsed + remaining
                if elapsed is not None and remaining is not None else None,
                unlimited=by_success, stale=stale)
