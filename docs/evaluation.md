# Avaliação

A CLI separa inspeção, simulação e plots. Use primeiro:

```bash
bluerov inspect --config configs/evaluation/trajectory_comparison.yaml
```

O plano mostra a tarefa, controladores habilitados e parâmetros, caminhos PPO,
condições, repetições, seeds e orçamento. `dt` do benchmark e `integration_dt`
da dinâmica aparecem separados para tornar divergências preexistentes visíveis.
A seleção de figuras não altera a seleção de controladores.

```yaml
controllers:
  - name: pid
    enabled: true
    params: {}
  - name: ppo
    enabled: true
    params:
      model_path: ../../artifacts/models/current/bluerov_ppo.zip
      vecnormalize_path: ../../artifacts/models/current/bluerov_vec_normalize.pkl
      env_config_path: ../dynamics/blue_rov_config.yaml
      dynamics_config_path: ../dynamics/dynamics_config.yaml
  - name: nmpc
    enabled: false
```

Modelos e normalizadores precisam existir nos caminhos explícitos. A nova CLI
não escolhe um checkpoint automaticamente. Os controladores são importados
somente quando selecionados; instalar CasADi é necessário apenas para NMPC.

## Um agente, um setpoint

`stationkeeping_single.yaml` usa a avaliação diagnóstica já existente:
setpoint `(0,0,-2)`, offset `(0.2,-0.15,0.08)`, yaw inicial 10°, yaw alvo 0°,
100 segundos e seed 42. A política é determinística, e o normalizador fica
congelado. Os erros e a transformação das observações são os mesmos de antes.

## Campanhas

`stationkeeping_comparison.yaml` usa um alvo e os controladores habilitados.
O runner histórico de station keeping percorre **todos** os itens de
`disturbance.scenarios`; `default_scenario` não restringe esse modo. Para limitar
os cenários, forneça somente os cenários desejados no YAML (sem carregar uma
lista externa maior). O `inspect` mostra exatamente essa seleção.

`trajectory_comparison.yaml` e `article.yaml` selecionam os cenários por
`scenario_names`/`default_scenario`, como o runner original. Repetições,
trajetórias, ordem de execução e estado inicial foram preservados.

## Artefatos

```text
artifacts/evaluation/<run_id>/
  config.requested.yaml
  config.resolved.yaml
  plan.yaml
  manifest.yaml
  data/                         # CSVs originais; trajetória: subpasta por cenário
  figures/
  tables/
  logs/evaluation.log
  logs/episodes.yaml             # campanhas: seed, caso, passos e parada
```

A avaliação isolada salva também `data/evaluation.yaml`, com configuração
física, hashes, resumo e flags de encerramento. Os CSVs e seus cabeçalhos não
foram alterados para acomodar metadados: eles ficam nos arquivos auxiliares.

Os parâmetros da planta ficam em `environment.dynamics_config`. Os caminhos de
configuração em `controllers[].params` pertencem ao controlador e não trocam a
planta por efeito colateral. A mecânica e o preset de ondas continuam separados.

Não se pressupõe equivalência entre a avaliação isolada, o runner de campanhas
e `workflows/evaluation/legacy_stationkeeping.py`: cada um mantém suas condições
e forma de registrar amostras. As entradas antigas da raiz foram removidas;
use a CLI ou execute explicitamente o módulo canônico desejado.

## Validar com a mesma dinâmica do treinamento

```yaml
environment:
  training_run: ../../artifacts/training/my_train
```

O resolver carrega `config.resolved.yaml` desse treino e herda sua configuração
de dinâmica e ambiente (limites, passo, etc.). Se overrides alterarem esses
parâmetros, a execução falha antes de simular. Para uma ablação intencional,
remova o vínculo e configure explicitamente as diferenças. A política e o
normalizador continuam sendo selecionados por caminhos explícitos nos parâmetros
PPO: use o par salvo na pasta `models` do mesmo treinamento.

O preset mecânico e o preset de ondas são independentes. O treino `my_train`
utilizou `mechanics_model: corrected` e ondas `std`; ele não habilitou as
correções opcionais de JONSWAP. Condições do mar podem variar entre as etapas do
currículo e os cenários de robustez da avaliação sem mudar as equações do modelo.

## Comparar o treino fixo `stationkeeping_restored` com PID e SMC

Essas configurações apontam explicitamente ao modelo e normalizador de
`artifacts/training/stationkeeping_restored/models/` e herdam a dinâmica do
snapshot desse treino. Incluem apenas PPO, PID e SMC. O cenário `training_sea`
reproduz os parâmetros de mar desse treino fixo (Hs=2 m, Tp=12 s), com três
repetições pareadas, seeds 42/43/44 e 100 segundos por execução.

```bash
source .venv-ppo-reference/bin/activate
bluerov evaluate --config configs/evaluation/trajectory_stationkeeping_restored.yaml --output artifacts/evaluation/stationkeeping_restored_traj
bluerov plot --run artifacts/evaluation/stationkeeping_restored_traj --suite trajectories
bluerov plot --run artifacts/evaluation/stationkeeping_restored_traj --suite article
bluerov evaluate --config configs/evaluation/stationkeeping_restored.yaml --output artifacts/evaluation/stationkeeping_restored_fixed
bluerov plot --run artifacts/evaluation/stationkeeping_restored_fixed --suite stationkeeping
```

Trajetórias: quadrado, oito com profundidade e círculo. A suíte `trajectories`
produz os plots espaciais, enquanto `article` produz as comparações de erro,
RMSE, CPU e energia. Station keeping: um alvo `[0,0,-0.8]`, yaw zero, com
perturbação inicial de posição/orientação definida pelo avaliador; a suíte
`stationkeeping` produz os gráficos e tabelas próprios desse benchmark.
Os resultados ficam nas pastas de avaliação indicadas, com figuras em `figures/`.
Use pastas de saída novas a cada execução. Edite os YAMLs para outros alvos,
trajetórias ou cenários; mudar o mar é um teste de robustez além do cenário de treino.
O runner de station keeping percorre todos os cenários presentes em `scenarios`.
Essa campanha avalia o agente existente sem retreiná-lo.

## Convenção das observações do PPO

O controlador PPO agora usa `params.observation_adapter: goal_frame` por padrão.
O adaptador transforma o estado para um mundo virtual ancorado na posição/yaw
do alvo e desconta as três velocidades de referência no BODY correto antes da
normalização. `legacy` reproduz a transformação antiga. As configurações com
snapshot de treinamento selecionam `goal_frame` explicitamente; as campanhas
antigas sem snapshot mantêm `legacy` para rastreabilidade.
Veja a [derivação matemática e as limitações para trajetórias móveis](ppo_observation_convention.md).
Essa mudança corrige a convenção de entrada, sem alterar treinamento, recompensa,
física ou normalizador e sem garantir, por si só, desempenho equivalente no tracking.
