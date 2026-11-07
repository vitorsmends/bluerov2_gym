# Currículo de trajetórias com progressão do mar

Modo opcional: `training.task: trajectory_curriculum_legacy`, implementado em
`workflows/training/trajectory_curriculum_legacy.py`. O treino de referência 889a766,
o currículo de station keeping e o experimento fixo permanecem separados.
Não foram modificados o simulador, o modelo de distúrbio ou suas equações.
O modo utiliza a API existente para mudar os parâmetros de mar a cada nível.

## Executar

```bash
bluerov train --config configs/training/trajectory_curriculum_legacy.yaml \
  --output artifacts/training/trajectory_curriculum_legacy
```

Para exigir GPU, acrescente `--device cuda`. Para múltiplas seeds, use `--seeds`;
como nos outros treinadores, cada seed recebe sua própria subpasta.

```bash
python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/trajectory_curriculum_legacy
```

O painel mostra nível, mar, trajetória atual e sucessos por trajetória.
Reward e erro euclidiano continuam sendo registrados; o reward por passo é bruto,
anterior à normalização.

## Níveis iniciais configurados

| Nível | Mar | Trajetórias disponíveis |
|---|---|---|
| 1 | Sem ondas | Figura oito de 889a766 |
| 2 | Calm | Figura oito original e círculo |
| 3 | Moderate | Anteriores e quadrado |
| 4 | Severe | Anteriores e oito com profundidade |
| 5 | Extreme | Anteriores e estrela |

Cada nível define `jonswap` e uma lista de `trajectories`, com `name`, `params`
e um `id` opcional. IDs devem ser únicos dentro do nível. A figura oito original
continua nos níveis seguintes para manter exposição à habilidade inicial.

A seleção é **cíclica**, em blocos definidos por
`trajectory_selection.episodes_per_trajectory` (padrão 3). Por exemplo, no nível
Calm são três episódios de figura oito, três de círculo, e o ciclo se repete.
Não há sorteio de tipo de trajetória que possa deixar um caso sem exposição.
O início temporal e a perturbação de posição continuam aleatórios, como na
receita 889a766.

## Episódios e critério de avanço

Este modo tem episódios de 1.000 passos = 100 s, configurados explicitamente
em `environment.max_episode_steps`. Essa duração serve para encerrar uma amostra,
medir sucesso e trocar a trajetória; **não é um limite de passos por fase**.
O perfil de referência 889a766 continua sem esse TimeLimit implícito.

A tolerância é medida na distância real à trajetória, em metros, antes da
normalização. Com a configuração inicial, um episódio só é aprovado se:

1. Completar os 100 s sem terminação por distância ou falha física.
2. Após os 10 s de acomodação, ficar abaixo de 0,20 m em pelo menos 90% dos passos.
3. Ter RMSE de posição de no máximo 0,20 m nesse mesmo trecho.

```yaml
trajectory_selection:
  episodes_per_trajectory: 3
curriculum_transition:
  mode: success
  settling_time_seconds: 10.0
  position_tolerance_m: 0.2
  min_tracking_fraction: 0.9
  max_rmse_m: 0.2
  consecutive_successes_per_trajectory: 3
```

O avanço exige três episódios consecutivos aprovados **de cada trajetória do
nível**. Uma falha zera a sequência daquela trajetória. As sequências são
contabilizadas separadamente por tipo, mesmo que episódios de outros tipos
ocorram entre suas execuções. Não basta dominar apenas a trajetória mais fácil.
Todos os contadores são reiniciados no próximo nível, já com o novo mar.
O critério é medido na coleta de treinamento, com exploração: não substitui a
avaliação determinística nem prova retenção em todos os mares anteriores.

Não há step budget por fase nem por treinamento neste modo. Se o nível não for
dominado, ele continua treinando nele até uma interrupção manual. Uma única
política, otimizador e normalizador são mantidos durante todos os níveis.
Na transição, o cache de observação do SB3 é invalidado para que a próxima chamada
learn receba um reset do novo cenário, sem reaproveitar observação da fase anterior.
A arquitetura, a recompensa, os seis comandos de força, o clip ±40 N e o feedback
virtual de 889a766 são reutilizados, sem novos canais de mar/trajetória na entrada
da rede. O nome da trajetória e do nível só aparece no info para monitoramento.

A progressão do distúrbio depende dos parâmetros efetivos do modelo atual:
scale, limite de velocidade da água, ganhos/limites de força e direção, entre
outros. No preset std, não se deve interpretar apenas Hs como energia imposta.
As flags e equações do distúrbio não são alteradas pelo treinador.

## Salvamento e avaliação

Há checkpoints periódicos pareados com normalizadores, além de um par
`_stage_N` ao concluir cada nível. O estado do currículo é salvo em
`models/curriculum_status.yaml`, incluindo cobertura e métricas do último episódio.
Ctrl+C salva também um par `_interrupted`; isso preserva os pesos e estatísticas,
mas não implementa retomada automática do estado interno completo da coleta.

Arquivos finais:

- `models/ppo_trajectory_curriculum.zip`.
- `models/vec_normalize_trajectory_curriculum.pkl`.

```bash
bluerov evaluate --config configs/evaluation/trajectory_curriculum_legacy.yaml \
  --output artifacts/evaluation/trajectory_curriculum_calm
bluerov plot --run artifacts/evaluation/trajectory_curriculum_calm --suite trajectories
bluerov plot --run artifacts/evaluation/trajectory_curriculum_calm --suite article
```

O YAML de avaliação seleciona **calm** por padrão, compara PPO/PID/SMC e inclui
as cinco trajetórias. Os cenários de mar foram preenchidos com os mesmos parâmetros
dos níveis deste currículo. Para outros níveis, altere `default_scenario` ou
selecione `scenario_names` no YAML. Se editar os parâmetros de mar do treino,
mantenha o YAML de avaliação correspondente coerente com a nova execução.
A física é herdada do snapshot do treinamento e o adaptador PPO é legacy.

## Escopo da revisão

A implementação foi revisada estaticamente. Não foram executados testes,
treinamentos, inferências ou simulações, conforme a restrição do usuário.
Os limiares propostos são configuráveis e não foram calibrados empiricamente.
