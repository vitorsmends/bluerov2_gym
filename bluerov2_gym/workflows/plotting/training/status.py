"""Describe saved training phases without loading a simulator or plotting backend."""
import math
from ...configuration import merge


def training_phase(config, step, runtime=None):
    """Describe the phase belonging to the latest recorded rollout, not wall time."""
    lessons = config.get('curriculum', [])
    task = config.get('training', {}).get('task')
    environment = config.get('environment', {})
    sea = merge(environment.get('dynamics_config', {}).get('jonswap', {}),
                environment.get('config', {}).get('jonswap', {}))
    boundaries = []
    if lessons and runtime and 1 <= runtime.get('index', 0) <= len(lessons):
        index = runtime['index']
        lesson = lessons[index - 1]
        sea = merge(sea, merge(config.get('jonswap_defaults', {}), lesson.get('jonswap', {})))
        label = f'Fase {index}/{len(lessons)}: {lesson["name"]}'
        if runtime.get('mode') == 'success':
            label += (f' — objetivo: {runtime.get("successful_episodes", 0)}/'
                      f'{runtime.get("required_episodes", 3)} episódios consecutivos')
            if runtime.get('status') == 'budget_exhausted':
                label += ' — limite atingido sem cumprir o objetivo'
            if runtime.get('status') == 'early_stopped':
                label += ' — early stopping por estagnação'
            elif runtime.get('stale_windows') is not None:
                label += f' — {runtime["stale_windows"]} janelas sem melhora'
        boundaries = [item['end_step'] for item in runtime.get('history', [])
                      if item['index'] < index and 'end_step' in item]
    elif lessons and config.get('curriculum_transition', {}).get('mode') == 'success':
        label = 'Currículo por objetivo — aguardando registro da fase ativa'
    elif lessons and task not in {'stationkeeping', 'stationkeeping_no_curriculum'}:
        rollout = int(config.get('ppo', {}).get('n_steps', 4096))
        end = 0
        selected = None
        for index, lesson in enumerate(lessons):
            start = end
            end += math.ceil(lesson['timesteps'] / rollout) * rollout
            if index < len(lessons) - 1:
                boundaries.append(end)
            if selected is None and step <= end:
                selected = (index, lesson, start, end)
        if selected is None:
            selected = (len(lessons) - 1, lessons[-1], start, end)
        index, lesson, start, end = selected
        progress = min(100., max(0., 100. * (step - start) / (end - start)))
        label = f'Fase {index + 1}/{len(lessons)}: {lesson["name"]} — {progress:.1f}% da fase'
        sea = merge(sea, merge(config.get('jonswap_defaults', {}), lesson.get('jonswap', {})))
    elif task == 'trajectory_tracking':
        label = 'Treinamento de trajetória — figura oito de 889a766 (sem currículo)'
    elif config:
        label = 'Cenário fixo — sem currículo'
    else:
        label = 'Aguardando configuração do treinamento'
    details = []
    for key, name, unit in [('Hs', 'Hs', 'm'), ('Tp', 'Tp', 's'),
                            ('max_current', 'Limite de velocidade da água', 'm/s')]:
        if key in sea:
            details.append(f'{name}={sea[key]:g} {unit}')
    if 'scale' in sea:
        details.append(f'escala={sea["scale"]:g}')
    return label, ' | '.join(details), boundaries

