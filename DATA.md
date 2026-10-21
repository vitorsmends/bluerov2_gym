# Dados atuais do artigo

A única campanha de avaliação mantida é:

```text
training_runs/ppo_physics_v2_seed42/
  models/             # modelo PPO seed 42, normalizador e checkpoints
  tensorboard/        # logs do treino no ambiente corrigido
  training_config.yaml
  evaluation/
    manifest.yaml
    moderate/trajectory_experiments/   # 20 CSVs
    severe/trajectory_experiments/     # 20 CSVs
    tables/                           # métricas derivadas
    figures/                          # gráficos derivados
```

São 40 simulações de 1000 passos, seed de distúrbio 42, com o PPO retreinado
na dinâmica `zup-airy-coriolis-v2`. A seed do treino e a seed do distúrbio
identificam processos diferentes. Há apenas uma política e uma realização
por condição; trajetórias diferentes não são repetições estocásticas.

As campanhas anteriores em `results-moderate`, `results-severe`,
`results-jonswap` e `results-physics-v2`, bem como os antigos exports
`article_plots_simple` e `article_tables_simple`, foram removidas para evitar
mistura de resultados. Modelos e logs históricos de treinamento foram
preservados, mas não integram a análise atual do artigo.

Para gerar os gráficos atuais, na raiz do repositório:

```bash
.venv/bin/python run_oceans.py --plots-only
```

Para sincronizar o artigo:

```bash
.venv/bin/python oceans2026/update_results.py
```

O artigo usa cópias dos gráficos em `oceans2026/figs/` e tabelas em
`oceans2026/tables/`. O script verifica os hashes dos artefatos contra o
manifesto antes de copiar os resultados.

Os CSVs, modelos e logs são armazenados com Git LFS. Após clonar:

```bash
git lfs install
git lfs pull
```
