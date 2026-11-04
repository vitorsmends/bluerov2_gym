# Gráficos e tabelas

`PlotManager` organiza as rotinas originais. Ele não unifica as métricas nem os
estilos. Cada rotina roda em processo separado para isolar estado do Matplotlib.
Importar o manager não carrega PPO/CasADi e não lê resultados nem cria pastas.

```bash
bluerov list-plots
bluerov plot --run artifacts/evaluation/SEU_RUN --suite article
bluerov plot --run artifacts/evaluation/SEU_RUN --plots tracking_error rmse energy
```

| Suite | Rotinas | Dados esperados |
|---|---|---|
| `single_policy` | Erros do diagnóstico de um setpoint | CSV e evaluation.yaml da avaliação isolada |
| `stationkeeping` | stationkeeping, station_cost | CSV por controlador com target_id e repetition_id |
| `trajectories` | Figuras espaciais originais | CSV por controlador/trajetória; escolhe repetição 0 |
| `article` | tracking_error, rmse, cpu_time, energy, pareto | CSVs de trajetórias agrupados por cenário |
| `legacy_path_tracking` | path_tracking, cost, cost_tradeoff, violin | CSV por controlador, como pid.csv |
| `training` | Curvas TensorBoard | Logs de treino |
| `optimization` | Resumo Optuna | Banco do estudo original |
| `ftc` | Comparação nominal/FTC | Dois CSVs explicitamente indicados |

A suite espacial percorre os cenários da campanha; `--scenario severe` limita a
seleção. `--repetition-id 1` seleciona outra repetição representativa. Alguns
plots 3D continuam mostrando somente NMPC/PPO, conforme a rotina original.
Os plots do artigo mantêm o layout Moderate/Severe original.

Figuras ficam em `run/figures/<familia>`, tabelas em `run/tables/<familia>` e logs
em `run/logs/plot_*.log`. Os nomes dos PNG/PDF/CSV dentro dessas pastas são os
mesmos de antes. Os gráficos usam dados salvos, sem repetir a simulação.

## Dados históricos

Os YAMLs em `configs/plots/*_legacy.yaml` apontam para os dados importados em
`artifacts/evaluation/imported/` e
escrevem em destinos separados sob `artifacts/plots/`. Ajuste os caminhos aos dados
presentes no clone. Os CSVs históricos foram movidos sem recalcular seu conteúdo.

```bash
bluerov plot --config configs/plots/article_legacy.yaml
bluerov plot --config configs/plots/trajectories_legacy.yaml
bluerov plot --config configs/plots/stationkeeping_legacy.yaml
```

O perfil `legacy_path_tracking` espera o formato por controlador, e não deve
ser aplicado a CSVs individuais de trajetórias. Há pré-validação de arquivos
e colunas para detectar seleção de suite incompatível.

A energia do artigo continua usando interpolação da curva T200. A energia das
outras famílias continua usando suas fórmulas/colunas anteriores. As regras de
agrupamento, interpolação temporal, tratamento de repetições e bandas também
não foram alteradas.

## Python

```python
from bluerov2_gym.workflows.plotting import PlotManager

plots = PlotManager('artifacts/evaluation/SEU_RUN')
plots.generate_suite('article')
plots.generate(['tracking_error', 'rmse'])
```

Também é possível fornecer `config` com `input_dir`, `output_dir`, `table_dir`
e, no artigo, `scenarios: {Moderate: caminho, Severe: caminho}`. No uso por Python,
forneça caminhos absolutos para evitar dependência do diretório atual.

## Publicação explícita

```bash
bluerov export-figures --run artifacts/evaluation/SEU_RUN \
  --folder article --destination oceans2026/figs/nova_comparacao
```

A exportação copia figuras, preserva nomes e recusa sobrescrever arquivos.
Nenhuma campanha publica automaticamente nas figuras do artigo.
