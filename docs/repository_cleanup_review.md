# Proposta de enxugamento do repositório

Inventário de 19/11/2026. **Nenhum arquivo foi removido e nenhum comportamento foi alterado.** A lista indica candidatos para decisão; não é uma lista de arquivos comprovadamente mortos.

## Conclusões principais

- O artigo e o path tracking da tese usam o treino `progressive_wave_exposure`, com 16 estágios e filtros de ondas/ruído desligados em todos eles. A mecânica simplificada continua ativa em A/B1/B2.
- A tese mantém station keeping e sua formulação histórica com filtros, energia e comando direto de seis propulsores. Não tratar esses recursos como obsoletos apenas porque saíram do artigo.
- Existem três repositórios adicionais dentro do workspace: artigo, tese e referência BlueROV2. Sua limpeza e histórico são independentes do Git principal.
- Os nomes `legacy`, `frozen` e `finetune` não são critério suficiente para exclusão: há dependências do fluxo atual com esses nomes.

## O que preservar no fluxo científico atual

| Item | Motivo |
|---|---|
| `configs/training/progressive_wave_exposure.yaml` | Receita atual de 16 estágios. |
| `artifacts/training/progressive_wave_exposure/` | Modelo/normalizador final, logs de treino e avaliações de avanço. Selecionar checkpoints antigos é uma decisão separada. |
| `artifacts/evaluation/article_scale4_four_controllers/` | 36 registros pareados atuais; tabelas e gráficos com PPO, NMPC, PID e SMC. |
| `artifacts/evaluation/article_current_ppo_wave_seeds/` | 60 episódios atuais: 20 seeds, três trajetórias, W2. |
| `artifacts/evaluation/progressive_wave_article_10199040/inputs/` | Contém o modelo e normalizador apontados pelos YAMLs atuais; não apagar esta pasta pela idade do nome. |
| `scripts/evaluate_article_scale4.py`, `scripts/evaluate_current_ppo_seeds.py`, `scripts/update_thesis_wave_figures.py` | Avaliação e figuras atuais; primeiro script reutiliza campanhas anteriores. |
| `relative_tracking.py`, `path_tolerance.py`, `wave_article_results.py`, `training/`, `thesis_wave_field.py` em plotting | Métricas e figuras publicadas, incluindo visualizador de treinamento. |
| `envs/core/dynamics.py`, alocação, controladores, loader/runner/configuração | Mecânica e execução do benchmark. |
| Assets/figuras e fontes incluídos pelas compilações da tese/artigo | Lista exata registrada no JSON (`document_inputs`). |

## Grupos de decisão

| Grupo | Candidatos | Recomendação e dependências |
|---|---|
| A — descartáveis regeneráveis | `__pycache__`, `.pyc`, `.DS_Store`, `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, `.toc`, `.lof`, `.lot`, `.blg`, `bluerov2_gym.egg-info/` | Remover após seleção, preservando os PDFs entregáveis e fontes/figuras. `.bbl` é regenerável, mas convém preservar a versão usada para submissão. |
| B — exploração de FT encerrada | `training/finetune_square.py`, `training/finetune_polygon.py`; `evaluation/finetune_campaign.py`; `plotting/finetune_*.py`, `checkpoint_alternatives.py`, `checkpoint_panels.py`, `finetune_decision.py`; YAMLs `*finetune*` e documentação correspondente | Arquivar em bloco. Antes, retirar a importação de `ROOT` que liga `thesis_wave_field.py` a `finetune_controller_comparison.py`; atualizar CLI/configuração/registro/testes. |
| C — receitas substituídas e diagnósticos antigos | `trajectory_889_*`, `trajectory_curriculum_legacy`, `trajectory_curriculum_pre_walker`, bridge de filtros, `walker_ramp*`, `progressive_wave_height*`, investigações de commits/observações | Arquivar receitas e campanhas antigas; manter a base de código `trajectory_frozen_env.py` e `reference_sources/trajectory_889a766.py`, ainda herdada pelo currículo atual. |
| D — experiências opcionais fora dos resultados atuais | FTC, otimização Optuna, GUI 3D/Meshcat, treinamento sem currículo | Decisão separada. FTC/Optuna são candidatos. O treino sem currículo foi pedido anteriormente como comparação futura: manter até revogação explícita. Meshcat/mesh não podem ser removidos antes de desacoplar a importação e a resolução do asset na inicialização do env. |
| E — duplicação de dados e checkpoints | cópias em `workers/`, campanhas intermediárias, múltiplas exportações de figura e snapshots de modelos | Consolidar depois de proteger modelos/logs finais e atualizar scripts/proveniência/caminhos. A simples ausência de uma figura no PDF não prova que seus dados podem ser apagados. |
| F — backups editoriais | `documents/oceans2026/reference_sources/`, `documents/msc_thesis/archives/`, imagens/exportações sem uso na compilação atual | Mover para arquivo externo, preferencialmente uma cópia verificável. Não apagar de primeira: contêm textos e resultados retirados a pedido. |
| G — referências externas | `_references/bluerov2/`, códigos e PDFs de referência | Podem sair da distribuição mínima após preservar revisão Git, links e arquivos usados como fonte de parâmetros. Não são o simulador atual; ainda documentam a origem dos coeficientes. |
| H — histórico Git/LFS | `.git/lfs`, `.git/objects`, históricos dos repositórios de documentos | Não apagar manualmente. Tratar separadamente com procedimentos de Git/LFS, confirmação de cópias e disponibilidade remota. Apagar arquivos do checkout não elimina versões históricas. |

## Filtros: resposta específica

1. **Path tracking atual: não são usados.** Todos os 16 estágios gravados têm `wave_filter=false` e `noise_filter=false`; os testes W1–W3 e com seeds usam o mesmo desligamento.
2. **Station keeping na tese: ainda há uso histórico/documentação.** `configs/evaluation/stationkeeping_comparison.yaml` seleciona o preset `std`; no código esse preset liga os dois filtros. Capítulos 3–5 mantêm condições anteriores, inclusive ruído e filtros. Não confirmei reconstrução de cada campanha histórica, portanto a remoção global não pode ser declarada segura.
3. **Proposta:** tirar presets filtrados da experiência padrão de path tracking e arquivá-los em um perfil de compatibilidade. Remover de vez somente se você decidir abrir mão da reexecução histórica de station keeping ou atualizar explicitamente sua metodologia e dados.
4. `alpha_wave`, `alpha_noise`, sigma de ruído, kernel antigo e rotação desabilitada são opções separadas; remover as chaves sem migrar YAMLs antigos quebra seu carregamento ou muda seu significado.

## Dependências que impedem apagar por nome

- `curriculum_context.py` importa `trajectory_frozen_env.py`, que herda `reference_sources/trajectory_889a766.py`: parte do PPO atual.
- `trajectory_curriculum.py` importa `TensorboardCallback` de `trajectory.py`; esse módulo importa infraestrutura de treino e env antigos. Extrair callbacks antes de retirar receitas.
- A CLI importa `training/curriculum.py` mesmo para outros tipos de treino. Não remover o treinador de station keeping sem revisar esse encadeamento.
- `thesis_wave_field.py` importa `ROOT` de `finetune_controller_comparison.py`. O gráfico atual não precisa do experimento FT, mas precisa hoje dessa importação.
- `wave_article_results.py` ainda importa `article/plot_energy.py` e calcula energia nos CSVs internos, embora a análise tenha saído do artigo. O registro `suite=article` também ainda lista `energy`. Limpar esse fluxo não exige apagar a energia usada pela tese em station keeping.
- `BlueRov.__init__` resolve `assets/BlueRov2.dae` e importa o renderer mesmo sem renderização. Remover Meshcat/mesh requer primeiro tornar esse caminho opcional.
- A mecânica `std` é usada na preparação atual. O adapter `legacy_yaw_error` também é selecionado pelo PPO atual; o nome antigo não significa feedback dispensável.
- `control_design/lqr_design.py` não está no artigo, mas o LQR ainda é descrito na seção final de `chapters/40_path_tracking_protocol.tex` da tese. Não excluir como “fora dos dois textos” sem decidir sobre essa seção.

## Tamanho das áreas principais (tamanho lógico dos arquivos; não espaço garantido recuperável)

| Área | Tamanho |
|---|---|
| `artifacts/` | 3181.1 MiB |
| `artifacts/training/` | 1092.7 MiB |
| `artifacts/evaluation/` | 1716.8 MiB |
| `artifacts/models/` | 256.8 MiB |
| `documents/oceans2026/reference_sources/` | 56.2 MiB |
| `documents/msc_thesis/archives/` | 2.8 MiB |
| `bluerov2_gym/assets/` | 43.6 MiB |
| `_references/` | 133.3 MiB |

Foram encontrados **87 grupos de arquivos com conteúdo idêntico**, com 315.6 MiB de bytes logicamente repetidos em `artifacts/` (apenas arquivos ≥1 MiB). Isso não é economia garantida: cópias podem sustentar snapshots e caminhos de reprodução; APFS também pode compartilhar blocos.

### Maiores grupos idênticos

- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w3/nmpc_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_ppo_nmpc/workers/walker_w3/data/walker_w3/nmpc_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w3/nmpc_figure_eight_depth.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w1/nmpc_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w1/nmpc_figure_eight_depth.csv`, `artifacts/evaluation/trajectory_scale_controllers/scale_4/data/walker_w1/nmpc_figure_eight_depth.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w3/ppo_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_ppo_nmpc/workers/walker_w3/data/walker_w3/ppo_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w3/ppo_figure_eight_depth.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w1/ppo_figure_eight_depth.csv`, `artifacts/evaluation/trajectory_scale_exploration/scale_4/data/walker_w1/ppo_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w1/ppo_figure_eight_depth.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w2/nmpc_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_ppo_nmpc/workers/walker_w2/data/walker_w2/nmpc_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w2/nmpc_figure_eight_depth.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w2/ppo_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_ppo_nmpc/workers/walker_w2/data/walker_w2/ppo_figure_eight_depth.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w2/ppo_figure_eight_depth.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w3/nmpc_circle.csv`, `artifacts/evaluation/article_scale4_ppo_nmpc/workers/walker_w3/data/walker_w3/nmpc_circle.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w3/nmpc_circle.csv`
- 3 cópias × 2.7 MiB: `artifacts/evaluation/article_scale4_ppo_nmpc/benchmark/data/walker_w1/nmpc_circle.csv`, `artifacts/evaluation/article_scale4_four_controllers/benchmark/data/walker_w1/nmpc_circle.csv`, `artifacts/evaluation/trajectory_scale_controllers/scale_4/data/walker_w1/nmpc_circle.csv`

## Inventário de campanhas para seleção

Abaixo estão todos os diretórios de primeiro nível. “Atual” não significa que toda cópia interna seja necessária; “histórico” não autoriza remoção sem verificar as dependências descritas.

### `artifacts/training/`

| Diretório | Tamanho | Situação inicial |
|---|---|---|
| `com_curriculo/` | 7.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `corner_curriculum_finetune/` | 75.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `imported/` | 267.4 MiB | Histórico importado; verificar station keeping da tese |
| `my_train/` | 60.0 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum/` | 19.5 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_from_scratch/` | 40.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_unfiltered/` | 32.3 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_v2/` | 2.3 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_walker/` | 105.0 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_yaw/` | 33.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_article_walker_finetune/` | 110.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `progressive_wave_exposure/` | 170.7 MiB | Preservar: evidência atual |
| `progressive_wave_height/` | 4.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `progressive_wave_height_acquisition/` | 5.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `progressive_wave_height_from_scratch/` | 31.3 MiB | Candidato a arquivo histórico; verificar dependências |
| `stationkeeping_restored/` | 20.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_889_frozen/` | 7.3 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_889_std/` | 7.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_reference/` | 0.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_square_finetune/` | 2.3 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_star_finetune/` | 10.3 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_ramp/` | 47.4 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_ramp_progress/` | 7.5 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_ramp_window/` | 22.4 MiB | Candidato a arquivo histórico; verificar dependências |
### `artifacts/evaluation/`

| Diretório | Tamanho | Situação inicial |
|---|---|---|
| `article_current_ppo_wave_seeds/` | 159.8 MiB | Preservar: evidência atual |
| `article_ppo_three_paths/` | 34.7 MiB | Candidato a arquivo histórico; verificar dependências |
| `article_scale4_four_controllers/` | 150.1 MiB | Preservar: evidência atual |
| `article_scale4_ppo_nmpc/` | 92.4 MiB | Fonte reutilizada por scripts/proveniência atuais; consolidar antes |
| `corner_ft_comparison_2836480/` | 60.0 MiB | Candidato a arquivo histórico; verificar dependências |
| `corner_ft_current_comparison/` | 59.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `corner_ft_final_article_2897920/` | 63.5 MiB | Candidato a arquivo histórico; verificar dependências |
| `corner_ft_final_article_three_paths_2897920/` | 42.5 MiB | Candidato a arquivo histórico; verificar dependências |
| `nmpc_ppo_unfiltered_quick/` | 2.1 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_comparison/` | 33.0 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_from_scratch/` | 24.4 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_yaw_comparison/` | 32.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_yaw_sea_states/` | 91.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_yaw_seed_robustness/` | 121.5 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_yaw_seed_star/` | 13.5 MiB | Candidato a arquivo histórico; verificar dependências |
| `physics_wave_curriculum_yaw_star/` | 24.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_889_frozen_vs_references_square/` | 2.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_889_std_vs_corrected_square/` | 2.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_commit_models_vs_current_square/` | 2.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_february_vs_june_vs_current_square/` | 2.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_from_scratch_comparison_1556480/` | 38.1 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_old_vs_new_square/` | 1.6 MiB | Candidato a arquivo histórico; verificar dependências |
| `ppo_without_wave_filter/` | 5.1 MiB | Candidato a arquivo histórico; verificar dependências |
| `progressive_wave_article_10199040/` | 73.0 MiB | Preservar inputs usados pelos YAMLs; restante avaliável |
| `smc_unfiltered_quick/` | 0.7 MiB | Candidato a arquivo histórico; verificar dependências |
| `star_finetune_validation/` | 277.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `stationkeeping_restored_fixed/` | 11.4 MiB | Candidato a arquivo histórico; verificar dependências |
| `stationkeeping_restored_traj/` | 21.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `train_stable_traj_vs_good_square/` | 1.6 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_889_std/` | 1.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_reference_diagnostic/` | 29.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `trajectory_scale_controllers/` | 46.4 MiB | Fonte reutilizada por scripts/proveniência atuais; consolidar antes |
| `trajectory_scale_exploration/` | 14.8 MiB | Fonte reutilizada por scripts/proveniência atuais; consolidar antes |
| `trajetorias_my_train/` | 21.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_checkpoint_4136960/` | 8.7 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_eight/` | 26.8 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_final_article_unfiltered_single/` | 27.2 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_final_article_unfiltered_vs_nmpc/` | 39.9 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_final_article_vs_nmpc/` | 42.1 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_ramp_final_vs_nmpc_4505600/` | 5.7 MiB | Candidato a arquivo histórico; verificar dependências |
| `walker_ramp_vs_nmpc_4341760/` | 4.9 MiB | Candidato a arquivo histórico; verificar dependências |
### `artifacts/models/`

| Diretório | Tamanho | Situação inicial |
|---|---|---|
| `current/` | 1.6 MiB | Candidato a arquivo histórico; verificar dependências |
| `curriculum/` | 255.1 MiB | Candidato a arquivo histórico; verificar dependências |
| `extracted/` | 0.1 MiB | Candidato a arquivo histórico; verificar dependências |

## Receitas e documentação para revisão

Listas completas, sem remoção automática. O único YAML de treino usado pela política atual do artigo é `progressive_wave_exposure.yaml`; isso não autoriza remover treinadores ou presets da tese por associação de nome.

### `configs/training/`

- `configs/training/corner_curriculum_finetune.yaml`
- `configs/training/curriculum.yaml`
- `configs/training/curriculum_timesteps.yaml`
- `configs/training/lessons_legacy.yaml`
- `configs/training/no_curriculum.yaml`
- `configs/training/physics_wave_curriculum_filter_bridge.yaml`
- `configs/training/physics_wave_curriculum_unfiltered.yaml`
- `configs/training/physics_wave_curriculum_walker.yaml`
- `configs/training/ppo_article_walker_finetune.yaml`
- `configs/training/progressive_wave_exposure.yaml`
- `configs/training/progressive_wave_height.yaml`
- `configs/training/progressive_wave_height_acquisition.yaml`
- `configs/training/progressive_wave_height_from_scratch.yaml`
- `configs/training/trajectory_889_frozen.yaml`
- `configs/training/trajectory_889_std.yaml`
- `configs/training/trajectory_curriculum.yaml`
- `configs/training/trajectory_curriculum_legacy.yaml`
- `configs/training/trajectory_curriculum_pre_walker.yaml`
- `configs/training/trajectory_reference.yaml`
- `configs/training/trajectory_square_finetune.yaml`
- `configs/training/trajectory_star_finetune.yaml`
- `configs/training/walker_ramp.yaml`
- `configs/training/walker_ramp_progress.yaml`
- `configs/training/walker_ramp_window.yaml`
### `configs/evaluation/`

- `configs/evaluation/article.yaml`
- `configs/evaluation/finetune_article_campaign.yaml`
- `configs/evaluation/ftc.yaml`
- `configs/evaluation/nmpc_ppo_unfiltered_quick.yaml`
- `configs/evaluation/ocean.yaml`
- `configs/evaluation/physics_wave_curriculum_from_scratch.yaml`
- `configs/evaluation/physics_wave_curriculum_yaw.yaml`
- `configs/evaluation/physics_wave_curriculum_yaw_sea_states.yaml`
- `configs/evaluation/physics_wave_curriculum_yaw_seed_robustness.yaml`
- `configs/evaluation/physics_wave_curriculum_yaw_seed_star.yaml`
- `configs/evaluation/physics_wave_curriculum_yaw_star.yaml`
- `configs/evaluation/ppo_wave_filter_ablation.yaml`
- `configs/evaluation/smc_unfiltered_quick.yaml`
- `configs/evaluation/star_finetune_validation.yaml`
- `configs/evaluation/stationkeeping_com_curriculo.yaml`
- `configs/evaluation/stationkeeping_comparison.yaml`
- `configs/evaluation/stationkeeping_restored.yaml`
- `configs/evaluation/stationkeeping_single.yaml`
- `configs/evaluation/stationkeeping_trajectory_reference.yaml`
- `configs/evaluation/trajectory_889_frozen.yaml`
- `configs/evaluation/trajectory_889_std.yaml`
- `configs/evaluation/trajectory_889_std_current_physics.yaml`
- `configs/evaluation/trajectory_com_curriculo.yaml`
- `configs/evaluation/trajectory_comparison.yaml`
- `configs/evaluation/trajectory_curriculum.yaml`
- `configs/evaluation/trajectory_curriculum_legacy.yaml`
- `configs/evaluation/trajectory_curriculum_restored.yaml`
- `configs/evaluation/trajectory_reference.yaml`
- `configs/evaluation/trajectory_square_finetune_sea_states.yaml`
- `configs/evaluation/trajectory_square_finetune_seed_robustness.yaml`
- `configs/evaluation/trajectory_square_finetune_seed_star.yaml`
- `configs/evaluation/trajectory_square_finetune_star.yaml`
- `configs/evaluation/trajectory_star_finetune_sea_states.yaml`
- `configs/evaluation/trajectory_star_finetune_seed_robustness.yaml`
- `configs/evaluation/trajectory_star_finetune_seed_star.yaml`
- `configs/evaluation/trajectory_star_finetune_star.yaml`
- `configs/evaluation/trajectory_stationkeeping_restored.yaml`
### `docs/`

- `docs/README.md`
- `docs/corner_curriculum_finetune.md`
- `docs/disturbance_models.md`
- `docs/evaluation.md`
- `docs/finetune_evaluation.md`
- `docs/gazebo_dynamics_validation.md`
- `docs/historical_training_evaluation_comparison.md`
- `docs/mechanics.md`
- `docs/migration.md`
- `docs/physics_wave_curriculum_proposal.md`
- `docs/plotting.md`
- `docs/ppo_observation_convention.md`
- `docs/ppo_tracking_diagnosis.md`
- `docs/progressive_wave_exposure.md`
- `docs/quickstart.md`
- `docs/repository_refactor_proposal.md`
- `docs/training.md`
- `docs/trajectory_curriculum.md`
- `docs/trajectory_curriculum_filter_bridge.md`
- `docs/trajectory_curriculum_legacy.md`
- `docs/trajectory_curriculum_walker.md`
- `docs/trajectory_training_reference.md`
- `docs/validation_audit_0436.md`
- `docs/vessel_dynamics.pdf`
- `docs/walker_ramp.md`

## Ordem sugerida após sua escolha

1. Consolidar um conjunto mínimo de arquivos/caminhos e registrar hashes dos modelos e dados protegidos.
2. Remover descartáveis regeneráveis escolhidos.
3. Desacoplar dependências antigas dos fluxos atuais; revisar CLI, registro de plots, documentação e testes associados.
4. Arquivar campanhas e cópias aprovadas; atualizar referências de arquivos antes de apagar originais.
5. Manter filtros/energia/mecânica histórica num perfil explícito de station keeping enquanto necessário para a tese.
6. Validar apenas os fluxos preservados e as compilações dos documentos. Limpeza de Git/LFS fica para uma operação separada.

Não executei treino, avaliação ou exclusões. O JSON contém referências de compilação, importações estáticas e hashes dos duplicados para apoiar a seleção.
