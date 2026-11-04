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
      model_path: ../../bluerov_ppo.zip
      vecnormalize_path: ../../bluerov_vec_normalize.pkl
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
runs/evaluation/<run_id>/
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
