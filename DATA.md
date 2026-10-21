# Organização dos dados de experimentos

Os arquivos pesados são versionados com Git LFS. Depois de clonar o
repositório, baixe-os com:

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
```

### Modelos e seeds de treinamento

Há artefatos de treinamento para as seeds **42, 43, 44, 45 e 46**. Para cada
seed, `models/` contém o modelo PPO final (`.zip`), as estatísticas de
normalização do ambiente (`.pkl`), checkpoints intermediários e a configuração
efetiva (`.yaml`). Os checkpoints incluem o número de passos no nome; use o
modelo final e o arquivo `VecNormalize` da mesma seed ao avaliar uma política.

Os logs do TensorBoard ficam em `bluerov_tensorboard/`. Os diretórios com
`Curriculum_seed_<seed>` correspondem às execuções identificadas por seed.
Os demais diretórios incluem outros treinamentos, estudos de Optuna e runs
anteriores; o nome do diretório não garante por si só uma seed específica.

### Resultados de avaliação

Os nomes das pastas indicam o cenário ou tipo de avaliação; os arquivos CSV
contêm as medições. Exemplos:

- `results-severe/path_tracking/`: CSV por controlador (`pid.csv`,
  `ppo.csv`, etc.). Os resultados atuais têm `repetition_id` e `seed`.
- `results-severe/stationkeeping/`: CSV por controlador para manutenção de
  posição, também com repetições e seeds de distúrbio.
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

Para entender como reproduzir os treinamentos e benchmarks, consulte
[experiments/README.md](./experiments/README.md).
