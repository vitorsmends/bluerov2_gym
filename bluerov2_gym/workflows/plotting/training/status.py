"""Describe saved training phases without loading a simulator or plotting backend."""
import math
from ...configuration import merge


def execution_status(runtime=None, manifest=None):
    """Report recorded termination explicitly; stale logs alone do not imply exit."""
    runtime, manifest = runtime or {}, manifest or {}
    terminal = {'completed', 'exposure_completed', 'early_stopped', 'interrupted', 'failed'}
    state = runtime.get('status')
    if state not in terminal:
        state = manifest.get('status') if manifest.get('status') in terminal else state
    if state not in terminal | {'training'}:
        state = 'preparing' if manifest.get('status') == 'created' else 'unknown'
    labels = dict(exposure_completed='Exposição a todas as fases concluída, sem domínio completo', completed='Treinamento concluído', early_stopped='Treinamento encerrado por early stopping',
                  interrupted='Treinamento interrompido', failed='Treinamento encerrado com falha',
                  training='Treinamento em andamento', preparing='Preparando treinamento',
                  unknown='Estado da execução não disponível')
    reasons = dict(exposure_completed='Todos os níveis foram visitados; algumas fases avançaram por estagnação, sem aprovação.', completed='A execução concluiu o protocolo de treinamento.',
                   interrupted='A execução foi interrompida antes da conclusão.',
                   failed='A execução terminou com erro.',
                   early_stopped='A avaliação independente acionou o critério de parada por estagnação.',
                   training='A execução ainda não registrou encerramento.',
                   preparing='Aguardando os primeiros registros do treinador.',
                   unknown='Não há registro suficiente para confirmar o estado.')
    reason = reasons[state]
    if state == 'early_stopped' and runtime.get('stale_evaluations') is not None:
        reason += f' {runtime["stale_evaluations"]} avaliações sem melhora significativa. O currículo não foi concluído.'
    progress = runtime.get('progress_monitor', {})
    if state == 'training' and progress.get('active') is False:
        reason = 'Aquisição inicial: a paciência de parada aguarda o nível mínimo de sucesso configurado.'
    return dict(unmastered_stages=runtime.get('unmastered_stages', []), state=state, label=labels[state], terminal=state in terminal, reason=reason,
                recorded_reason=runtime.get('stop_reason') or manifest.get('reason') or manifest.get('error'),
                ended_at=runtime.get('updated_at') if state in terminal else None,
                step=runtime.get('step'), phase=runtime.get('name'))


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
            evidence = 'janelas independentes aprovadas' if runtime.get('mastery_evidence') == 'nonoverlapping_windows' else 'validações aprovadas'
            label += (f' — {evidence}: {runtime.get("successful_batches", 0)}/'
                      f'{runtime.get("required_batches", 3)}')
            window = runtime.get('last_evaluation', {}).get('mastery_window')
            if window:
                label += f' — janela: {window["batches"]}/{window["required_batches"]} avaliações'
                if 'success_rate' in window:
                    label += f' (sucesso {window["success_rate"]:.1%})'
            progress = runtime.get('progress_monitor', {})
            if progress.get('active') is False:
                label += ' — aquisição inicial: paciência de parada suspensa'
            context = runtime.get('effective_context') or {}
            sea = merge(sea, context.get('jonswap', {}))
            if context.get('sea_case'):
                label += f' — {context["sea_case"]}'
            skip = runtime.get('early_phase_skipping', {})
            if skip:
                label += (' — early skipping protegido' if skip.get('protected') else
                          f' — early skipping: {skip.get("stale_evaluations", 0)}/{skip.get("patience_evaluations", 10)} sem melhora')
            skipped = [h['index'] for h in runtime.get('history', []) if h.get('status') == 'advanced_on_plateau']
            if skipped:
                label += ' — fases sem domínio: ' + ', '.join(map(str, skipped))
            label += f' — mecânica: {runtime.get("mechanics_model", "?")}'
            if context and context.get('stage_index', index-1) != index-1:
                label += f' — revisão: {context["name"]}'
            if runtime.get('status') in {'interrupted', 'early_stopped', 'failed', 'completed', 'exposure_completed'}:
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

