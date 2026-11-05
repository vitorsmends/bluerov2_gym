# Treinamento

## Treinamento validado: station keeping por currículo

Referência: `training/train_curriculum.py` do commit
`0436ce3482e95105cb2376681b2bd6cbec72d99c`, fornecido pelo usuário.
A implementação usa a organização atual do pacote e a física corrigida.

A receita preservada tem `n_steps=4096`, `batch_size=256`, redes separadas
`pi=[256,256]` e `vf=[256,256]`, normalização de observações/reward e cinco fases:

| Fase | Limite de passos por fase |
|---|---:|
| No Waves | 200.000 |
| Calm Sea | 400.000 |
| Moderate Sea | 600.000 |
| Severe Sea | 800.000 |
| Extreme Sea | 1.000.000 |

A mesma política, otimizador e `VecNormalize` continuam entre fases. A referência
não fixa seed do PPO: `seed: null` mantém esse comportamento; `--seeds 42`
é uma escolha explícita para controlar a aleatoriedade. A seed padrão do mar
continua 42. A recompensa e as observações permanecem de station keeping na origem.
O avanço agora depende do objetivo, conforme `curriculum_transition.mode: success`.
O agente deve terminar três episódios consecutivos mantendo erro de posição abaixo
do raio de sucesso do ambiente (0,20 m) em todos os passos dos 10 segundos finais,
sem terminação por falha física. O critério usa episódios da coleta de treinamento,
com exploração do PPO. `hold_seconds` e `consecutive_episodes` são configuráveis;
`radius_m`, se informado, substitui o raio do objetivo.

Os passos da tabela são limites de segurança, não uma autorização de avanço.
Se a fase esgotar o orçamento sem cumprir o objetivo, o treino para, salva pesos
e normalizador com sufixo `_phase_N_incomplete` e marca a execução como falha;
não passa ao próximo mar. O orçamento pode ser ultrapassado até completar o rollout.
Para reproduzir a regra antiga, selecione `curriculum_transition.mode: timesteps`:
nesse modo, 3.000.000 passos solicitados resultam em 3.010.560 efetivos.
O painel usa `models/curriculum_status.yaml` para mostrar a fase realmente ativa,
em vez de inferir a fase pelos antigos limites de passos.

```bash
bluerov train --config configs/training/curriculum.yaml \
  --output artifacts/training/curriculum_restored
```

Use uma pasta de saída nova. Esse comando não passa `--seeds`, portanto os arquivos
finais são `models/bluerov_ppo_curriculum.zip` e
`models/bluerov_vec_normalize_curriculum.pkl`, sem sufixo de seed.

```bash
bluerov evaluate --config configs/evaluation/trajectory_curriculum_restored.yaml \
  --output artifacts/evaluation/curriculum_restored_trajectories
```

Essa avaliação herda a física do snapshot do novo treino. A configuração
`trajectory_comparison.yaml` existente mantém a seleção anterior de `my_train`.
Para observar no navegador:

```bash
.venv-ppo-reference/bin/python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/curriculum_restored
```

**O treinamento padrão é o com currículo**, em `configs/training/curriculum.yaml`.
O experimento anterior de cenário fixo foi preservado separadamente em
`workflows/training/no_curriculum.py` e `configs/training/no_curriculum.yaml`.
Ele só é selecionado explicitamente, para comparações futuras:

```bash
bluerov train --config configs/training/no_curriculum.yaml \
  --output artifacts/training/sem_curriculo
```

| Propriedade | Padrão: com currículo | Experimento preservado: sem currículo |
|---|---|---|
| Configuração | `curriculum.yaml` | `no_curriculum.yaml` |
| Tarefa | `stationkeeping_curriculum` | `stationkeeping_no_curriculum` |
| Mar | Cinco fases progressivas | Fixo, parâmetros JONSWAP da configuração física |
| `n_steps` / `batch_size` | 4096 / 256 | 2048 / 64 |
| Rede | `pi`/`vf`: `[256,256]` | `[256,256]` |
| Passos solicitados | 3.000.000 | 1.000.000 |
| Limite efetivo de episódio | Registro do ambiente (5.000 neste pacote) | 1.000 explícitos |

Ambos usam a física corrigida, observações/reward de station keeping, checkpoints
com normalizador, métricas euclidianas e progresso. O cenário fixo mantém a receita
que produziu o experimento anterior; portanto, comparar diretamente os dois não
isola apenas o efeito do currículo. Para esse estudo, alinhe hiperparâmetros,
seed, orçamento e limite de episódio. A opção já existente `--ablation no_curriculum`
sobre `curriculum.yaml` é outro experimento: mantém a receita curricular e usa
seu mar final desde o início; não substitui o cenário fixo preservado.

A visualização no navegador funciona para ambos; passe o diretório de saída
correspondente em `--run`. Cada treino usa uma pasta nova e seus próprios pesos
e normalizador. Execuções já iniciadas não mudam de cenário ao editar o código.
O limite de episódio mantém a regra da referência: o registro preexistente do
ambiente é preservado (5.000 passos neste pacote), apesar do campo nominal 1.000
no YAML. Essa regra não foi alterada silenciosamente nesta restauração.

Implementação principal: `bluerov2_gym/workflows/training/curriculum.py`.
Configuração: `configs/training/curriculum.yaml`.

```bash
bluerov train --config configs/training/curriculum.yaml
bluerov train --config configs/training/curriculum.yaml --seeds 42 43 44 45 46
bluerov train --config configs/training/curriculum.yaml \
  --seeds 42 43 44 45 46 --ablation no_curriculum
```

O algoritmo PPO, arquitetura, recompensas, cronograma do currículo e ablações
continuam sendo os da implementação existente. A nova interface muda a
organização das saídas, sem lançar avaliação ao final.

Cada seed recebe uma execução exclusiva com `models/`, `logs/tensorboard/`,
`logs/training.log`, configurações solicitada/efetiva e `manifest.yaml`.
Com `--output artifacts/training/estudo --seeds 42 43`, as pastas serão
`artifacts/training/estudo/seed_42` e `seed_43`. Os arquivos de modelo mantêm seus
nomes anteriores, incluindo sufixos por seed quando solicitados.

O manifesto registra commit, estado local do Git, versões e seed. O manifesto
`models/*_config.yaml` produzido pelo treinador permanece disponível, inclusive
as flags efetivas da dinâmica. Nenhum modelo antigo é movido.

## Curvas

```bash
bluerov plot-training --run artifacts/training/SEU_RUN
```

Para TensorBoard histórico, edite `configs/plots/training.yaml` e execute:

```bash
bluerov plot-training --config configs/plots/training.yaml
```

Use `run: NOME_DO_LOG` para um diagnóstico individual ou `groups` para agrupar
pelo menos cinco treinamentos independentes, com os padrões já suportados:

```yaml
suite: training
log_dir: ../../artifacts/logs/tensorboard
output_dir: ../../artifacts/plots/comparacao_treinos
min_seeds: 5
groups:
  - baseline=PPO_BlueROV2_Curriculum_seed_*
  - no curriculum=PPO_BlueROV2_Curriculum_ablation_no_curriculum_seed_*
```

Cinco repetições do benchmark não são cinco treinamentos independentes.

Os fluxos de treino anteriores com comportamentos próprios permanecem acessíveis
somente pelos módulos canônicos, sem duplicatas na raiz:

```bash
python -m bluerov2_gym.workflows.training.legacy_train
python -m bluerov2_gym.workflows.training.optimize
```

Eles mantêm seus parâmetros, nomes de estudos e destinos anteriores; não são
substituídos silenciosamente pelo treino com currículo. A otimização requer
as dependências opcionais `.[optimize]`.

## Métricas físicas de aprendizado

`PositionErrorCallback` registra, sem alterar ações ou recompensas:

- `rollout/position_error_mean_m`: média dos erros observados no rollout;
- `rollout/position_error_rmse_m`: raiz da média dos erros ao quadrado;
- `rollout/episode_position_error_mean_m`: média do erro médio dos últimos
  100 episódios completos.

Valores são metros, provenientes do `info` físico do ambiente, antes de
normalizações de observação/reward. Episódios de ambientes vetorizados são
acumulados separadamente. Dados ausentes não são convertidos em erro zero.
A coleta só afeta treinamentos realizados após a inclusão do callback.

O gráfico `training_overview` faz parte da suite `training` e indica
explicitamente métricas ausentes. Os arquivos escalares exportados ficam em
`tables/training_overview/training_scalars.csv`. A duração maior de um episódio
pode mudar a soma do reward; interprete reward junto com duração e erro.

Para treinamentos antigos, `bluerov evaluate-checkpoints --training-run RUN`
mede checkpoints em um desafio fixo. O resultado é retrospectivo, determinístico,
com uma única seed por execução; para estimar variabilidade, repita com outras
seeds em diretórios separados. Esse teste não recupera o histórico amostral do
treinamento original. O modelo final também é incluído, além dos checkpoints.

## Acompanhar reward e distância euclidiana ao vivo

Em outro terminal, com o mesmo ambiente Python ativado:

```bash
python -m bluerov2_gym.workflows.plotting.training.live \
  --run artifacts/training/curriculum_restored
```

Abre dois gráficos e atualiza a cada 5 segundos, sem modificar o treino ou os logs.
O primeiro mostra reward médio por episódio; o segundo mostra erro euclidiano médio e RMSE, em metros.
A janela pode ser aberta antes dos primeiros registros. As métricas só mudam quando o treinador
grava outro rollout (4.096 passos nessa configuração), não a cada época interna do PPO.
A legenda mostra o último valor bruto; as linhas fortes usam média móvel de 5 registros.
Use `--window 1` para exibir sem suavização e `--interval 2` para consultar a cada 2 segundos.
Fechar a janela não interrompe o treinamento.

Para apenas salvar uma imagem, inclusive sem interface gráfica:

```bash
MPLBACKEND=Agg python -m bluerov2_gym.workflows.plotting.training.live \
  --run artifacts/training/curriculum_restored --once
```

A imagem fica em `figures/live/training_live.png` dentro da execução.

## Monitorar o treinamento pelo navegador

Em outro terminal, na raiz do repositório:

```bash
.venv-ppo-reference/bin/python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/curriculum_restored
```

Abre automaticamente `http://127.0.0.1:8765` e atualiza a cada 5 segundos.
Mostra reward, erro euclidiano médio/RMSE, fase ou cenário fixo e parâmetros do mar.
A página permite alterar a suavização. Lê somente os logs; não altera o treinamento.
Use `--port 8766` caso a porta esteja ocupada, `--interval 2` para atualizar a cada 2 segundos
e `--no-open` para não abrir uma aba automaticamente. Encerre o servidor com Ctrl+C
nesse terminal; o treinamento continua no outro. O diretório em `--run` precisa ser
o mesmo passado como `--output` ao treinamento.

## Informações das ondas disponíveis aos controladores

PID e SMC usam estado do ROV e referência; o acesso à dinâmica fornece a matriz
de alocação dos propulsores. PPO usa observações do ROV e a referência pelo adaptador,
sem corrente/força de onda como entrada. NMPC conhece parâmetros do modelo e a
referência futura, porém `_forecast_disturbance_sequence` retorna zeros.
O `info` do simulador contém diagnósticos das ondas para registro, mas o runner
não o entrega como feedback aos controladores. Todos recebem o estado ideal do ROV,
sem ruído de sensores: os resultados atuais não modelam incerteza de estimação.
