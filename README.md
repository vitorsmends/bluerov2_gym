# BlueROV2 Gymnasium

Ambiente de reinforcement learning do BlueROV2 com dinâmica de seis graus de
liberdade, distúrbios JONSWAP e avaliação de PID, SMC, NMPC e PPO.

## Instalação

Na raiz do repositório, em um ambiente Python 3.10 ou superior compatível com
PyTorch e suas dependências:

```bash
python -m pip install -e .
# Opcional: NMPC e otimização de hiperparâmetros
python -m pip install -e '.[control,optimize]'
```

O comando principal é `bluerov`. A alternativa equivalente, sem depender do
executável no PATH, é `python -m bluerov2_gym.workflows`.

## Treinar

```bash
bluerov train --config configs/training/curriculum.yaml
# Treinamentos independentes:
bluerov train --config configs/training/curriculum.yaml --seeds 42 43 44 45 46
```

Cada treinamento salva seus modelos, VecNormalize, checkpoints e TensorBoard
em uma pasta própria sob `runs/training/`. [Guia de treinamento](docs/training.md).

## Avaliar o agente atual em um setpoint

```bash
bluerov inspect --config configs/evaluation/stationkeeping_single.yaml
bluerov evaluate --config configs/evaluation/stationkeeping_single.yaml
```

Essa configuração usa `bluerov_ppo.zip` e `bluerov_vec_normalize.pkl` da raiz,
setpoint `(0, 0, -2 m)`, 100 segundos e mecânica corrigida. Não retreina o agente.
O comando imprime o diretório da execução. Gere os gráficos a partir dele:

```bash
bluerov plot --run runs/evaluation/SEU_RUN --suite single_policy
```

## Comparar controladores

```bash
bluerov inspect --config configs/evaluation/trajectory_comparison.yaml
bluerov evaluate --config configs/evaluation/trajectory_comparison.yaml
bluerov plot --run runs/evaluation/SEU_RUN --suite trajectories
bluerov plot --run runs/evaluation/SEU_RUN --suite article
```

A lista `controllers` no YAML define os controladores da avaliação. Use
`enabled: false` para excluir um controlador. O `inspect` mostra a seleção,
parâmetros, caminhos da política/normalizador, cenários, seeds e total de casos
antes da simulação. Confira os caminhos dos modelos nos exemplos de campanhas;
os checkpoints históricos podem não estar disponíveis neste clone.

| Configuração | Finalidade |
|---|---|
| `configs/evaluation/stationkeeping_single.yaml` | Um agente PPO, um setpoint |
| `configs/evaluation/stationkeeping_comparison.yaml` | PID/PPO/SMC/NMPC, um setpoint, cenários do YAML |
| `configs/evaluation/trajectory_comparison.yaml` | Campanha geral de trajetórias |
| `configs/evaluation/article.yaml` | Campanha do artigo: 200 casos configurados |
| `configs/evaluation/ftc.yaml` | Campanha com a política FTC indicada |

[Guia de avaliação](docs/evaluation.md) · [Guia de plots](docs/plotting.md).

## Gráficos existentes e curvas de treino

Plotar não inicia avaliação e não carrega políticas:

```bash
bluerov list-plots
bluerov plot --config configs/plots/article_legacy.yaml
bluerov plot --config configs/plots/stationkeeping_legacy.yaml
bluerov plot-training --config configs/plots/training.yaml
```

As rotinas de desenho e cálculo foram preservadas. A classe `PlotManager`
organiza as suites e seus caminhos; cada família mantém suas métricas e estilos.
As entradas antigas de scripts foram removidas. Resultados históricos continuam
nos diretórios originais. Novas execuções não
sobrescrevem automaticamente figuras de `oceans2026/figs`.

## Organização

```text
bluerov2_gym/
  envs/                    # ambiente, dinâmica, recompensa e renderização
  workflows/
    cli.py                 # train, evaluate, inspect, plot, plot-training
    training/              # currículo, treino legado, otimização
    evaluation/            # runner, controladores, referências e trajetórias
    plotting/              # PlotManager e famílias de figuras
configs/                   # configuração principal, agrupada por tarefa
runs/                      # novas execuções; dados locais, ignorados pelo Git
control_design/            # projetos de controle e seu modelo próprio
scripts/                   # auditoria física específica
```

[Guia rápido](docs/quickstart.md) · [Mapa de migração](docs/migration.md) ·
[Dados históricos e Git LFS](DATA.md).

O Gym continua disponível diretamente:

```python
import gymnasium as gym
import bluerov2_gym

env = gym.make('BlueRov-v0')
obs, info = env.reset()
```

A seleção da mecânica e das ondas é independente. Consulte
[mechanics.md](docs/mechanics.md) e [disturbance_models.md](docs/disturbance_models.md).

## Verificação

```bash
python -m unittest discover -s tests -v
```

Licença MIT.
