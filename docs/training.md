# Treinamento

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
Com `--output runs/training/estudo --seeds 42 43`, as pastas serão
`runs/training/estudo/seed_42` e `seed_43`. Os arquivos de modelo mantêm seus
nomes anteriores, incluindo sufixos por seed quando solicitados.

O manifesto registra commit, estado local do Git, versões e seed. O manifesto
`models/*_config.yaml` produzido pelo treinador permanece disponível, inclusive
as flags efetivas da dinâmica. Nenhum modelo antigo é movido.

## Curvas

```bash
bluerov plot-training --run runs/training/SEU_RUN
```

Para TensorBoard histórico, edite `configs/plots/training.yaml` e execute:

```bash
bluerov plot-training --config configs/plots/training.yaml
```

Use `run: NOME_DO_LOG` para um diagnóstico individual ou `groups` para agrupar
pelo menos cinco treinamentos independentes, com os padrões já suportados:

```yaml
suite: training
log_dir: ../../bluerov_tensorboard
output_dir: ../../runs/plots/comparacao_treinos
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
