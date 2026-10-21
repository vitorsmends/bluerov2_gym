# Organização dos dados de experimentos

Os arquivos pesados são versionados com Git LFS. Um clone pode mostrar os
caminhos dos arquivos mesmo quando o conteúdo ainda não foi baixado: nesse
caso, o arquivo local contém apenas um ponteiro LFS. Depois de clonar o
repositório, baixe os objetos com:

```bash
git lfs install
git lfs pull
```

## Estrutura

```text
models/
  bluerov_ppo_curriculum_seed_<seed>.zip
  bluerov_vec_normalize_curriculum_seed_<seed>.pkl
  bluerov_ppo_curriculum_checkpoint_seed_<seed>_<steps>_steps.zip
  ...arquivos auxiliares dos checkpoints...
  bluerov_ppo_curriculum_seed_<seed>_config.yaml
bluerov_tensorboard/
  PPO_BlueROV2_Curriculum_seed_<seed>_1/
  PPO_BlueROV2_Curriculum_<run>/
  PPO_BlueROV2_Optuna_Trial_<trial>_1/
  ...outros logs de treinamento...
results-moderate/
  trajectory_ftc/                 # avaliação FTC por controlador e trajetória
  tables_trajectory_experiments/   # tabela resumida
results-severe/
  path_tracking/                   # avaliação de seguimento de trajetória
  stationkeeping/                  # avaliação de manutenção de posição
  trajectory_experiments/          # avaliação por controlador e trajetória
  tables_*/                        # tabelas agregadas por experimento
training_runs/
  ppo_physics_v2_seed42/           # retreinamento Physics v2, seed 42
    models/                        # modelo final e checkpoints
    tensorboard/                   # logs TensorBoard do treinamento
    evaluation/                    # CSVs e manifesto da avaliação
```

### Modelos e seeds de treinamento

O commit inclui nomes de artefatos de treinamento para as seeds **42, 43, 44,
45 e 46**. Isso não garante que seus conteúdos estejam baixados em cada clone,
nem identifica por si só uma ablação. Quando os objetos estiverem disponíveis,
`models/` contém modelos PPO (`.zip`), estatísticas de normalização do ambiente
(`.pkl`), checkpoints intermediários e configurações (`.yaml`). Os checkpoints
incluem o número de passos no nome; use o modelo e o arquivo `VecNormalize` da
mesma seed ao avaliar uma política.

Não há evidência documentada de cinco seeds para a ablação `no_curriculum`.
No checkout em que este inventário foi elaborado, foi identificado um log
local dessa ablação para a seed 42; isso não confirma a disponibilidade dos
dados correspondentes no outro computador. Não presuma que as seeds 43–46
sejam dessa ablação sem conferir suas configurações e logs.

O pacote `training_runs/ppo_physics_v2_seed42/` é uma execução separada de
retreinamento com a dinâmica `zup-airy-coriolis-v2`, seed 42. Ele contém modelo
final, checkpoints, logs TensorBoard, configuração e CSVs de avaliação para
cenários moderado e severo. Não é uma seed adicional do baseline curriculum
42–46 nem evidência de cinco seeds para uma ablação.

Os caminhos dos logs do TensorBoard ficam em `bluerov_tensorboard/`. Os
diretórios com `Curriculum_seed_<seed>` correspondem às execuções identificadas
por seed. Os demais diretórios incluem outros treinamentos, estudos de Optuna
e runs anteriores; o nome do diretório não garante por si só uma seed
específica.

### Resultados de avaliação

Os nomes das pastas indicam o cenário ou tipo de avaliação; quando hidratados,
os arquivos CSV contêm as medições. Exemplos:

- `results-severe/path_tracking/`: CSV por controlador (`pid.csv`,
  `ppo.csv`, etc.). No checkout usado para elaborar este documento, os CSVs
  inspecionados tinham `repetition_id` e `seed`; confira os arquivos baixados
  no seu clone antes de depender dessas colunas.
- `results-severe/stationkeeping/`: CSV por controlador para manutenção de
  posição. Os CSVs inspecionados no checkout de referência tinham colunas de
  repetição e seed; valide os arquivos locais após o download.
- `results-severe/trajectory_experiments/`: CSV por controlador e trajetória,
  por exemplo `ppo_circle.csv`. Os arquivos atualmente armazenados são
  resultados legados de uma execução de 1000 passos: não têm `repetition_id`
  nem `seed`.
- `results-moderate/trajectory_ftc/`: resultados de avaliação FTC, separados
  por controlador e trajetória.
- `results-*/tables_*/`: tabelas resumidas ou agregadas, não substituem os
  CSVs brutos por execução.

**Seed de treinamento e seed de distúrbio são coisas diferentes.** A seed no
nome de um modelo identifica a execução de treinamento da política. Nos CSVs
de avaliação que têm as colunas `seed` e `repetition_id`, elas identificam a
realização do distúrbio e a repetição do benchmark. Os CSVs legados de
`trajectory_experiments/` não permitem recuperar essas informações.

Para conferir os arquivos LFS presentes no clone:

```bash
git lfs ls-files
```

Em geral, `*` na saída indica um objeto LFS baixado e `-` indica que o
conteúdo não está disponível localmente. Depois do `git lfs pull`, confira a
saída novamente e inspecione o cabeçalho dos CSVs relevantes, por exemplo:

```bash
git lfs pull
git lfs ls-files
head -n 1 results-severe/path_tracking/ppo.csv
head -n 1 results-severe/stationkeeping/ppo_stationkeeping.csv
head -n 1 results-severe/trajectory_experiments/ppo_circle.csv
```

Para confirmar uma seed de treinamento ou ablação, compare também os arquivos
de configuração em `models/` e os logs correspondentes do TensorBoard; a
existência do caminho no Git ou o nome da pasta não prova que o objeto foi
baixado nem que pertence a uma ablação específica.

Para entender como reproduzir os treinamentos e benchmarks, consulte
[experiments/README.md](./experiments/README.md).
