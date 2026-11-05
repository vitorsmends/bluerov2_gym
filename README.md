# BlueROV2 Gymnasium

Ambiente de reinforcement learning do BlueROV2 com dinâmica de seis graus de
liberdade, distúrbios JONSWAP e avaliação de PID, SMC, NMPC e PPO.

## Setup antes do treinamento

Na raiz do repositório, execute no macOS ou Linux:

```bash
bash scripts/setup.sh
source .venv-ppo-reference/bin/activate
```

O script instala o `uv` se necessário, cria o ambiente com Python 3.10 quando
ele não existe, instala o pacote em modo editável e verifica as dependências.
Inclui treinamento, plots, PID/SMC, NMPC e otimização. Não precisa de `pip`
instalado dentro do ambiente. Se o ambiente já existe, ele é reutilizado.
A ativação com `source` deve ser feita no seu terminal, pois um script executado
com `bash` não consegue ativar o ambiente no processo do terminal que o chamou.

As versões centrais do PPO seguem `configs/setup/ppo_reference_constraints.txt`,
conferidas no ambiente usado nos testes: NumPy 2.2.6, SciPy 1.15.3, Gymnasium 1.2.3,
Stable-Baselines3 2.8.0 e PyTorch 2.11.0. O setup usa essas restrições com
`pyproject.toml`; não aplica as versões antigas do `uv.lock`.
O instalador de `uv` segue a [documentação oficial da Astral](https://docs.astral.sh/uv/getting-started/installation/).

Para instalar apenas as dependências de treinamento, plots e PID/SMC:

```bash
bash scripts/setup.sh --minimal
```

Para conferir o que seria instalado, sem instalar pacotes:

```bash
bash scripts/setup.sh --dry-run
```

Para criar outro ambiente (o caminho relativo é resolvido a partir da raiz do projeto):

```bash
BLUEROV_VENV=.venv-comparacao bash scripts/setup.sh
source .venv-comparacao/bin/activate
```

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

Depois do setup, execute o treinamento padrão com currículo e física nova:

```bash
bluerov train --config configs/training/curriculum.yaml \
  --output artifacts/training/com_curriculo
```

Use uma pasta de saída nova para cada execução. Em novos terminais, basta repetir
`source .venv-ppo-reference/bin/activate`; não é preciso reinstalar tudo.

O comando principal é `bluerov`. A alternativa equivalente, sem depender do
executável no PATH, é `python -m bluerov2_gym.workflows`.

## Passo a passo: treinamento, avaliação e plots

Execute os comandos abaixo na raiz do repositório, com o ambiente Python ativado.
Por exemplo, `source .venv/bin/activate` ou, neste workspace,
`source .venv-ppo-reference/bin/activate`. Se o comando `bluerov` ainda não
estiver disponível, execute `bash scripts/setup.sh` e ative o ambiente novamente.
O setup funciona mesmo quando o ambiente não possui `pip`.

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

### 1. Configurar e executar o currículo

Edite `configs/training/curriculum.yaml` para definir os parâmetros PPO,
as etapas do currículo e o orçamento de passos. A configuração física vem de
`configs/dynamics/dynamics_config.yaml`; alterações específicas do treino podem
ser colocadas em `environment.dynamics_config` no YAML de treinamento.

Confira a configuração e execute uma seed com um nome de saída explícito:

```bash
bluerov inspect --config configs/training/curriculum.yaml
bluerov train --config configs/training/curriculum.yaml \
  --seeds 42 --output artifacts/training/meu_treino
```

O diretório de saída precisa ser novo. Para repetir o teste, escolha outro nome.
Sem `--output`, o programa cria uma pasta com timestamp em `artifacts/training/`.

Com a configuração padrão e o comando acima, os arquivos principais serão:

```text
artifacts/training/meu_treino/
  models/bluerov_ppo_curriculum_seed_42.zip
  models/bluerov_vec_normalize_curriculum_seed_42.pkl
  models/                       # checkpoints e configuração do treinador
  logs/tensorboard/             # eventos para curvas e TensorBoard
  logs/training.log
  config.requested.yaml
  config.resolved.yaml
  manifest.yaml
```

### 2. Gerar as curvas do treinamento

Depois do treino:

```bash
bluerov plot-training --run artifacts/training/meu_treino
```

Os PDFs ficam em `artifacts/training/meu_treino/figures/training/`.
O comando lê os logs existentes; não inicia outro treinamento. Métricas que
não foram registradas são omitidas. Um teste muito curto pode não gerar dados
PPO suficientes para desenhar as curvas.

Para acompanhar o treinamento ao vivo:

```bash
tensorboard --logdir artifacts/training/meu_treino/logs/tensorboard
```

### Reward e erro euclidiano ao longo do treinamento

Para o treino atual:

```bash
bluerov plot-training --run artifacts/training/my_train
```

O painel `figures/training_overview/training_progress.png` mostra reward e duração
dos episódios, com as transições do currículo. Os gráficos individuais também
são exportados em PNG e PDF. O eixo X representa passos de interação com o
ambiente; `n_epochs` do PPO são passagens internas sobre um rollout, não esse eixo.

Nos próximos treinos, o erro euclidiano médio e RMS por rollout e a média por
episódio também serão registrados no TensorBoard. O treino `my_train` anterior
não continha essas métricas; não é possível recuperá-las exatamente a partir do
reward, que combina posição, velocidade, yaw, estabilidade e bônus.

Para avaliar retrospectivamente os checkpoints, sem retreiná-los:

```bash
bluerov evaluate-checkpoints --training-run artifacts/training/my_train --steps 500 --seed 42 --output artifacts/checkpoint_evaluation/novo_teste
bluerov plot --run artifacts/checkpoint_evaluation/novo_teste --suite training_checkpoints
```

O destino deve ser novo. O teste usa o cenário final do currículo para **todos**
os checkpoints, estado inicial igual e a mesma seed; `--stage 0` seleciona outra
etapa, e `--steps` controla o horizonte. Cada modelo é pareado com seu próprio
VecNormalize, com estatísticas congeladas. As medições e hashes ficam em `data/`;
o protocolo fica em `protocol.yaml`. Encerramentos antecipados são registrados.

Já gerei esse teste para `my_train` em
`artifacts/checkpoint_evaluation/my_train_fixed_final/`. Para regenerar somente
a figura existente:

```bash
bluerov plot --run artifacts/checkpoint_evaluation/my_train_fixed_final --suite training_checkpoints
```

A figura é uma **reavaliação de checkpoints**, não o histórico de erro medido
no treino. Ela testa estabilização na origem, objetivo do treinamento atual,
e não substitui a avaliação de seguimento de trajetória.

### 3. Avaliar o modelo treinado em um único setpoint

Crie uma cópia de `configs/evaluation/stationkeeping_single.yaml`, na mesma
pasta, chamada `meu_agente.yaml`. Ajuste nela os arquivos do modelo e do
normalizador, sempre usando o par da mesma execução:

```yaml
model: ../../artifacts/training/meu_treino/models/bluerov_ppo_curriculum_seed_42.zip
normalizer: ../../artifacts/training/meu_treino/models/bluerov_vec_normalize_curriculum_seed_42.pkl
target: [0.0, 0.0, -2.0]
duration: 100.0
seed: 42
```

Preserve as demais entradas da cópia, incluindo `experiment`, `dynamics` e
`offset`. Os caminhos no YAML são relativos ao próprio arquivo.

```bash
bluerov inspect --config configs/evaluation/meu_agente.yaml
bluerov evaluate --config configs/evaluation/meu_agente.yaml \
  --output artifacts/evaluation/meu_agente_setpoint
bluerov plot --run artifacts/evaluation/meu_agente_setpoint --suite single_policy
```

Abra `artifacts/evaluation/meu_agente_setpoint/figures/single_policy/errors.png`.
O CSV fica em `data/stationkeeping.csv`. Essa avaliação congela a normalização
e não modifica os pesos do agente.

### 4. Repetir com várias seeds

```bash
bluerov train --config configs/training/curriculum.yaml \
  --seeds 42 43 44 45 46 --output artifacts/training/meu_estudo
```

Cada seed terá sua própria pasta, por exemplo `artifacts/training/meu_estudo/seed_42/`.
Para plotar uma delas:

```bash
bluerov plot-training --run artifacts/training/meu_estudo/seed_42
```

Para curvas agregadas entre treinamentos, configure os grupos de logs conforme
[o guia de treinamento](docs/training.md). Repetições de avaliação não substituem
seeds independentes de treinamento.

## Avaliar o agente atual em um setpoint

```bash
bluerov inspect --config configs/evaluation/stationkeeping_single.yaml
bluerov evaluate --config configs/evaluation/stationkeeping_single.yaml
```

Essa configuração usa `bluerov_ppo.zip` e `bluerov_vec_normalize.pkl` de
`artifacts/models/current/`,
setpoint `(0, 0, -2 m)`, 100 segundos e mecânica corrigida. Não retreina o agente.
O comando imprime o diretório da execução. Gere os gráficos a partir dele:

```bash
bluerov plot --run artifacts/evaluation/SEU_RUN --suite single_policy
```

## Comparar controladores

```bash
bluerov inspect --config configs/evaluation/trajectory_comparison.yaml
bluerov evaluate --config configs/evaluation/trajectory_comparison.yaml
bluerov plot --run artifacts/evaluation/SEU_RUN --suite trajectories
bluerov plot --run artifacts/evaluation/SEU_RUN --suite article
```

A configuração `trajectory_comparison.yaml` usa o modelo final e o normalizador
correspondente do treino `artifacts/training/my_train` (seed 42). A entrada
`environment.training_run` herda a configuração física e do ambiente salva nesse
treino, em vez de depender dos defaults atuais. Overrides incompatíveis são
rejeitados; os cenários de ondas da campanha continuam definidos separadamente.
Para outro treino, ajuste `training_run`, `model_path` e `vecnormalize_path`.

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

A suite de plots deve corresponder aos dados da avaliação:

| Tipo de avaliação | Comando de plots |
|---|---|
| Um PPO e um setpoint | `bluerov plot --run CAMINHO_DO_RUN --suite single_policy` |
| Campanha de station keeping | `bluerov plot --run CAMINHO_DO_RUN --suite stationkeeping` |
| Trajetórias XY/3D | `bluerov plot --run CAMINHO_DO_RUN --suite trajectories` |
| Erro, RMSE, CPU, energia e Pareto das trajetórias | `bluerov plot --run CAMINHO_DO_RUN --suite article` |

Para uma campanha de station keeping:

```bash
bluerov inspect --config configs/evaluation/stationkeeping_comparison.yaml
bluerov evaluate --config configs/evaluation/stationkeeping_comparison.yaml \
  --output artifacts/evaluation/comparacao_stationkeeping
bluerov plot --run artifacts/evaluation/comparacao_stationkeeping --suite stationkeeping
```

Confira `controllers[].enabled` e os caminhos PPO antes de executar.
As figuras ficam em `figures/`, as tabelas em `tables/` e os registros em `logs/`
dentro do diretório da avaliação. Você pode repetir apenas o comando `plot`
sem rodar novamente a simulação.

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
As entradas antigas de scripts foram removidas. Resultados históricos foram
centralizados em `artifacts/`, preservando seu conteúdo. Novas execuções não
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
artifacts/                 # modelos, treinos, avaliações, logs, figuras e tabelas
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

Os dados antes espalhados na raiz foram movidos para `artifacts/`. Veja o
[mapa de dados](DATA.md) para localizar modelos, logs e resultados históricos.

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
