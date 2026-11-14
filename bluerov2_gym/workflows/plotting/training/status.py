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
        if runtime.get('mode') == 'physics_wave_success':
            label += (f' — validações aprovadas: {runtime.get("successful_batches", 0)}/'
                      f'{runtime.get("required_batches", 3)}')
            window = runtime.get('last_evaluation', {}).get('mastery_window')
            if window:
                label += f' — janela: {window["batches"]}/{window["required_batches"]} avaliações'
                if 'success_rate' in window:
                    label += f' (sucesso {window["success_rate"]:.1%})'
            context = runtime.get('effective_context') or {}
            sea = merge(sea, context.get('jonswap', {}))
            if context.get('sea_case'):
                label += f' — {context["sea_case"]}'
            label += f' — mecânica: {runtime.get("mechanics_model", "?")}'
            if context and context.get('stage_index', index-1) != index-1:
                label += f' — revisão: {context["name"]}'
            if runtime.get('status') in {'interrupted', 'early_stopped', 'failed', 'completed'}:
                label += f' — {runtime["status"]}'
        elif runtime.get('mode') == 'trajectory_success':
            required = runtime.get('required_episodes', 3)
            counts = runtime.get('trajectory_successes', {})
            covered = sum(value >= required for value in counts.values())
            label += f' — trajetórias aprovadas: {covered}/{len(counts)}'
            if runtime.get('current_trajectory'):
                label += f' — atual: {runtime["current_trajectory"]}'
            if runtime.get('status') == 'interrupted':
                label += ' — interrompido'
        elif runtime.get('mode') == 'success':
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
        if key == 'max_current' and sea.get(key, 0) >= 1e5:
            details.append('Velocidade da água sem saturação ativa')
        elif key in sea:
            details.append(f'{name}={sea[key]:g} {unit}')
    if 'scale' in sea:
        details.append(f'escala={sea["scale"]:g}')
    return label, ' | '.join(details), boundaries

