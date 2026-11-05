# Treinamento

## Treinamento de trajetória recuperado de `889a766`

O caminho que replica o script `experiments_traj/train_ppo.py` fornecido pelo
usuário é `configs/training/trajectory_reference.yaml`. Ele usa a **arquitetura
e o treinamento de trajetória originais**, sobre a dinâmica e os distúrbios
atuais, sem alterar o simulador. É um treinador separado do station keeping.

```bash
bluerov train --config configs/training/trajectory_reference.yaml \
  --output artifacts/training/trajectory_reference
```

Para exigir GPU, acrescente `--device cuda`. O perfil preserva `n_steps=2048`,
`batch_size=64`, rede padrão da MultiInputPolicy, reset aleatório na figura oito,
observação virtual original, recompensa de tracking e um milhão de passos.
Não usa currículo nem a observação experimental goal_frame. Os pesos finais
ficam em `models/ppo_trajectory_final.zip`, com `models/vec_normalize.pkl` pareado.

Monitoramento:

```bash
python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/trajectory_reference
```

Avaliação contra PID e SMC, incluindo a referência analítica do próprio treino:

```bash
bluerov evaluate --config configs/evaluation/trajectory_reference.yaml \
  --output artifacts/evaluation/trajectory_reference
bluerov plot --run artifacts/evaluation/trajectory_reference --suite trajectories
bluerov plot --run artifacts/evaluation/trajectory_reference --suite article
```

Para station keeping desse agente, use
`configs/evaluation/stationkeeping_trajectory_reference.yaml` na avaliação e a
suíte `stationkeeping` nos plots. Use pastas novas a cada execução.

O tratamento original de feedback foi mantido inclusive nas suas particularidades:
reset absoluto, steps com observação virtual, correção de velocidade apenas em u
e ausência de limite temporal de episódio quando a subclasse é instanciada
diretamente. A explicação completa de fidelidade, diferenças permitidas e comandos
está em [Treinamento de trajetória de referência](trajectory_training_reference.md).

O currículo segue disponível como padrão **para station keeping**, assim como
o experimento sem currículo. Eles não são substitutos deste treino de trajetória.


## Treinamento validado: station keeping por currículo

Referência: `training/train_curriculum.py` do commit
`0436ce3482e95105cb2376681b2bd6cbec72d99c`, fornecido pelo usuário.
A implementação usa a organização atual do pacote e a física corrigida.

A receita preservada tem `n_steps=4096`, `batch_size=256`, redes separadas
`pi=[256,256]` e `vf=[256,256]`, normalização de observações/reward e cinco fases:
No Waves → Calm Sea → Moderate Sea → Severe Sea → Extreme Sea.

**Não existe limite de passos por fase no treinamento padrão.** A mesma política,
otimizador e `VecNormalize` continuam aprendendo na fase atual até cumprir o
objetivo: três episódios consecutivos mantendo erro de posição abaixo do raio
de sucesso do ambiente (0,20 m) durante todos os passos dos 10 segundos finais,
sem terminação por falha física. O critério usa episódios da coleta de treinamento,
com exploração do PPO. `hold_seconds` e `consecutive_episodes` são configuráveis;
`radius_m`, se informado, substitui o raio do objetivo.

O progresso mostra passos executados e a fase ativa, sem percentual ou ETA de um
total artificial. Os checkpoints e métricas continuam sendo salvos normalmente.
A referência não fixa seed do PPO: `seed: null` mantém esse comportamento;
`--seeds 42` controla explicitamente a aleatoriedade. A seed padrão do mar é 42.

O early stopping por estagnação é **opcional e desativado por padrão**:

```yaml
curriculum_transition:
  mode: success
  hold_seconds: 10.0
  consecutive_episodes: 3
  early_stopping:
    enabled: false
    window_episodes: 20
    patience_windows: 10
    min_delta_m: 0.01
    warmup_episodes: 40
```

Quando habilitado, após o aquecimento de 40 episódios, compara o erro euclidiano
médio em janelas não sobrepostas de 20 episódios. Uma melhora maior que 0,01 m
renova a paciência. Dez janelas consecutivas sem essa melhora encerram o treino
na fase atual, com status `early_stopped`, sem liberar o próximo mar. A paciência
é reiniciada a cada fase. Episódios sem métricas válidas não geram uma avaliação
de melhora. Cada melhora salva o par `_phase_N_best.zip` / normalizador
`_phase_N_best.pkl`; ao parar, o estado mais recente é salvo com sufixo
`_phase_N_incomplete`, sem substituir o modelo final de um currículo concluído.

A regra antiga por passos permanece somente em
`configs/training/curriculum_timesteps.yaml`, para comparações históricas.
Ela não é usada pelo comando de treinamento padrão.
O painel e os plots usam `models/curriculum_status.yaml` para mostrar as mudanças
de fase registradas, em vez de inferi-las por limites fixos de passos.

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
| Condição de parada | Objetivo por fase; sem limite de passos | 1.000.000 passos |
| Limite efetivo de episódio | Registro do ambiente (5.000 neste pacote) | 1.000 explícitos |

Ambos usam a física corrigida, observações/reward de station keeping, checkpoints
com normalizador, métricas euclidianas e progresso. O cenário fixo mantém a receita
que produziu o experimento anterior; portanto, comparar diretamente os dois não
isola apenas o efeito do currículo. Para esse estudo, alinhe hiperparâmetros,
seed, condições de parada e limite de episódio. A opção já existente `--ablation no_curriculum`
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

### GPU NVIDIA: detecção e uso no Dell G15

O setup agora usa `uv --torch-backend auto` para selecionar o backend do PyTorch
compatível com o hardware/driver. Se `nvidia-smi` encontra uma GPU e o PyTorch
existente não oferece CUDA, o setup reinstala o pacote `torch`, mantendo a versão
fixada. Ao final, mostra o dispositivo e testa uma operação real. Não instala o
driver NVIDIA do sistema; se esse driver não estiver disponível/compatível, será
necessário corrigir sua instalação. A seleção automática segue a
[integração oficial uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

Para reparar uma instalação existente e conferir a GPU:

```bash
bash scripts/setup.sh --reinstall-torch
source .venv-ppo-reference/bin/activate
python -m bluerov2_gym.workflows.training.device --device cuda --check
```

Os dois treinadores usam `training.device: auto`: CUDA quando o PyTorch consegue
acessá-la; CPU nos outros casos. Para exigir GPU e impedir fallback silencioso:

```bash
bluerov train --config configs/training/curriculum.yaml \
  --device cuda --output artifacts/training/com_curriculo_cuda
```

Também é possível selecionar `--device cpu` ou `--device auto`. Os logs mostram
`requested`, `selected`, versão/build CUDA do PyTorch e nome da GPU; o manifesto
do modelo registra `execution_device`. Para consultar a utilização no Dell,
execute `nvidia-smi -l 1` em outro terminal.

A GPU executa a rede PPO e sua otimização; a dinâmica NumPy do ROV continua na CPU.
A arquitetura, os hiperparâmetros e a física não são alterados pela seleção de
dispositivo. Utilização baixa de GPU durante a coleta de rollouts pode ser normal:
PPO com rede pequena e um único ambiente pode ser mais rápido em CPU, como observa a
[documentação do SB3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
Compare os passos por segundo antes de concluir que CUDA acelera esse experimento.

Para escolher explicitamente o backend no setup, use por exemplo
`bash scripts/setup.sh --torch-backend=cpu --reinstall-torch`; backends CUDA
específicos também podem ser passados nesse argumento, desde que tenham um wheel
compatível com a versão de PyTorch fixada e com o driver.
