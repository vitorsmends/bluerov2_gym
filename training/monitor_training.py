from __future__ import annotations

import argparse
import html
import math
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_LOG_DIR = Path("ppo_traj_tensorboard")
DEFAULT_OUTPUT = DEFAULT_LOG_DIR / "training_progress.html"
DEFAULT_TOTAL_TIMESTEPS = 1_000_000


def find_latest_run(log_dir: Path) -> Path:
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"Nenhum evento do TensorBoard encontrado em {log_dir}")
    return max(event_files, key=lambda path: path.stat().st_mtime).parent


def read_scalars(run_dir: Path) -> tuple[dict[str, list[tuple[int, float]]], float]:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"Nenhum evento do TensorBoard encontrado em {run_dir}")

    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()

    scalars = {}
    wall_times = []
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if events:
            scalars[tag] = [(event.step, event.value) for event in events]
            wall_times.extend(event.wall_time for event in events)

    if not wall_times:
        raise RuntimeError(f"O TensorBoard ainda não publicou métricas em {run_dir}")
    return scalars, max(wall_times) - min(wall_times)


def format_value(value: float) -> str:
    if not math.isfinite(value):
        return "n/d"
    return f"{value:.4g}"


def make_chart(title: str, tag: str, values: list[tuple[int, float]]) -> str:
    width, height = 760, 260
    left, right, top, bottom = 64, 18, 30, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    escaped_title = html.escape(title)

    if not values:
        return (
            f'<section class="chart"><h2>{escaped_title}</h2>'
            '<p class="empty">Aguardando esta métrica.</p></section>'
        )

    finite_values = [(step, value) for step, value in values if math.isfinite(value)]
    if not finite_values:
        return (
            f'<section class="chart"><h2>{escaped_title}</h2>'
            '<p class="empty">Sem valores finitos.</p></section>'
        )

    min_step = min(step for step, _ in finite_values)
    max_step = max(step for step, _ in finite_values)
    min_value = min(value for _, value in finite_values)
    max_value = max(value for _, value in finite_values)
    if max_step == min_step:
        max_step += 1
    if max_value == min_value:
        padding = max(abs(max_value) * 0.05, 0.5)
        min_value -= padding
        max_value += padding

    points = []
    for step, value in finite_values:
        x = left + (step - min_step) / (max_step - min_step) * plot_width
        y = top + (max_value - value) / (max_value - min_value) * plot_height
        points.append(f"{x:.1f},{y:.1f}")

    x_label = f"{min_step:,}–{max_step:,} passos"
    return f"""
    <section class="chart">
      <h2>{escaped_title}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escaped_title}">
        <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
        <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
        <line x1="{left}" y1="{top + plot_height / 2:.1f}" x2="{left + plot_width}" y2="{top + plot_height / 2:.1f}" class="grid"/>
        <polyline points="{' '.join(points)}" class="curve"/>
        <text x="8" y="{top + 5}" class="label">{format_value(max_value)}</text>
        <text x="8" y="{top + plot_height}" class="label">{format_value(min_value)}</text>
        <text x="{left}" y="{height - 10}" class="label">{x_label}</text>
        <text x="{left + plot_width}" y="{height - 10}" text-anchor="end" class="label">etapa global</text>
      </svg>
      <p class="tag">{html.escape(tag)} · {len(finite_values)} amostras</p>
    </section>
    """


def build_html(
    run_dir: Path,
    output: Path,
    scalars: dict[str, list[tuple[int, float]]],
    elapsed_seconds: float,
    total_timesteps: int,
    refresh_seconds: int | None,
) -> str:
    latest_step = max(
        (step for values in scalars.values() for step, _ in values),
        default=0,
    )
    progress = min(100.0, 100.0 * latest_step / total_timesteps)
    latest_fps = scalars.get("time/fps", [(0, 0.0)])[-1][1]
    remaining = max(total_timesteps - latest_step, 0)
    eta = remaining / latest_fps if latest_fps > 0 else math.nan
    latest_reward = scalars.get("rollout/ep_rew_mean", [(0, math.nan)])[-1][1]
    latest_error = scalars.get("custom/tracking_error_m", [(0, math.nan)])[-1][1]
    latest_success = scalars.get("rollout/success_rate", [(0, math.nan)])[-1][1]

    charts = "".join(
        make_chart(
            title,
            tag,
            scalars.get(tag, []),
        )
        for tag, title in (
            ("rollout/ep_rew_mean", "Recompensa média por episódio"),
            ("custom/tracking_error_m", "Erro médio de trajetória"),
            ("rollout/success_rate", "Taxa de sucesso"),
            ("time/fps", "Velocidade de treinamento"),
        )
    )

    refresh = (
        f'<meta http-equiv="refresh" content="{refresh_seconds}">'
        if refresh_seconds is not None
        else ""
    )
    elapsed_minutes = elapsed_seconds / 60.0
    eta_text = f"{eta / 60:.1f} min" if math.isfinite(eta) else "n/d"
    success_text = f"{100 * latest_success:.1f}%" if math.isfinite(latest_success) else "n/d"

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh}
  <title>Progresso do PPO</title>
  <style>
    body {{ margin: 0; padding: 24px; color: #17212b; background: #f3f6f8;
      font: 15px system-ui, sans-serif; }}
    main {{ max-width: 1200px; margin: auto; }}
    h1 {{ margin-bottom: 4px; }}
    .subtle, .tag {{ color: #617282; font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr));
      gap: 12px; margin: 22px 0; }}
    .card, .chart {{ padding: 16px; border-radius: 10px; background: white;
      box-shadow: 0 1px 5px #1c344011; }}
    .card span {{ display: block; color: #617282; font-size: 13px; }}
    .card strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    .progress {{ height: 12px; overflow: hidden; border-radius: 8px; background: #dce5eb; }}
    .progress div {{ height: 100%; width: {progress:.2f}%; background: #25855a; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr));
      gap: 14px; }}
    .chart h2 {{ margin: 0 0 8px; font-size: 17px; }}
    svg {{ width: 100%; height: auto; }}
    .axis {{ stroke: #71808c; stroke-width: 1.2; }}
    .grid {{ stroke: #dce5eb; stroke-width: 1; }}
    .curve {{ fill: none; stroke: #25855a; stroke-width: 2.5;
      stroke-linecap: round; stroke-linejoin: round; }}
    .label {{ fill: #617282; font-size: 12px; }}
    .empty {{ min-height: 100px; color: #617282; }}
    @media (max-width: 520px) {{ body {{ padding: 12px; }} .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Progresso do treinamento PPO</h1>
    <div class="subtle">Run: {html.escape(str(run_dir))} · Atualizado: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
    <div class="progress" aria-label="Progresso estimado"><div></div></div>
    <div class="cards">
      <div class="card"><span>Timesteps</span><strong>{latest_step:,} / {total_timesteps:,}</strong></div>
      <div class="card"><span>Progresso</span><strong>{progress:.1f}%</strong></div>
      <div class="card"><span>Recompensa média</span><strong>{format_value(latest_reward)}</strong></div>
      <div class="card"><span>Erro médio</span><strong>{format_value(latest_error)} m</strong></div>
      <div class="card"><span>Taxa de sucesso</span><strong>{success_text}</strong></div>
      <div class="card"><span>FPS / ETA</span><strong>{format_value(latest_fps)} / {eta_text}</strong></div>
      <div class="card"><span>Tempo observado</span><strong>{elapsed_minutes:.1f} min</strong></div>
    </div>
    <div class="charts">{charts}</div>
    <p class="subtle">Arquivo: {html.escape(str(output))}</p>
  </main>
</body>
</html>
"""


def update_dashboard(
    run_dir: Path,
    output: Path,
    total_timesteps: int,
    refresh_seconds: int | None,
) -> None:
    scalars, elapsed_seconds = read_scalars(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_html(
            run_dir,
            output,
            scalars,
            elapsed_seconds,
            total_timesteps,
            refresh_seconds,
        ),
        encoding="utf-8",
    )
    latest_step = max(
        (step for values in scalars.values() for step, _ in values),
        default=0,
    )
    print(f"[OK] {latest_step:,}/{total_timesteps:,} passos · {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um dashboard HTML com o progresso do treinamento PPO."
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Diretório do TensorBoard a monitorar; por padrão seleciona o run mais recente.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--interval", type=int, default=10, help="Segundos entre atualizações.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Gera o HTML uma vez e termina, sem acompanhar novas métricas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps deve ser maior que zero.")
    if args.interval <= 0:
        raise ValueError("--interval deve ser maior que zero.")

    run_dir = args.run_dir or find_latest_run(args.log_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Diretório de run não encontrado: {run_dir}")

    while True:
        update_dashboard(
            run_dir,
            args.output,
            args.total_timesteps,
            None if args.once else args.interval,
        )
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
