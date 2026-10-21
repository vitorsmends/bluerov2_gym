# Comparação dos controladores: moderate e severe

Na raiz do repositório, execute:

```bash
.venv/bin/python run_oceans.py
```

O comando executa PID, SMC, NMPC e uma política PPO fixa nas cinco trajetórias,
para moderate e severe: 40 simulações de 1000 passos, dt=0.1 s. A configuração
está em `experiments/config/article_trajectories.yaml`; os parâmetros de mar são
lidos de `experiments/ocean_environment.yaml`. Uma realização por condição é o
padrão. A seed de distúrbio é a mesma entre controladores. Não se executam novos
modelos PPO nem a análise de sensibilidade às seeds de treinamento.

Por padrão, utiliza `ppo_trajectory_final.zip` e `vec_normalize.pkl`, o par
indicado na configuração legada de comparação. Isso não atribui retrospectivamente
os CSVs antigos a esse modelo. Para selecionar outra política fixa, informe o par:

```bash
.venv/bin/python run_oceans.py \
  --model models/bluerov_ppo_curriculum_seed_42.zip \
  --vecnormalize models/bluerov_vec_normalize_curriculum_seed_42.pkl \
  --results-root results-physics-v2-seed42
```

Saídas padrão:

- `results-physics-v2/moderate/trajectory_experiments/`: CSVs moderate.
- `results-physics-v2/severe/trajectory_experiments/`: CSVs severe.
- `results-physics-v2/manifest.yaml`: configuração, parâmetros de mar e hashes do
  modelo e normalizador.
- `results-physics-v2/figures/`: gráficos agregados e trajetórias dos dois cenários.
- `results-physics-v2/tables/`: tabelas RMSE e Pareto; tabelas espaciais também são
  salvas nas subpastas de cada cenário.

Uma pasta de resultados não vazia é recusada para preservar campanhas anteriores.
Use outro `--results-root` para uma nova execução. Os arquivos antigos em
`results-moderate`, `results-severe` e as figuras do artigo são preservados.

Para executar somente a coleta, adicione `--skip-plots`. Para gerar novamente
os gráficos de uma campanha concluída:

```bash
.venv/bin/python run_oceans.py --plots-only --results-root results-physics-v2
```

Teste curto de integração, separado dos dados científicos:

```bash
.venv/bin/python run_oceans.py --steps 4 --results-root /tmp/bluerov-smoke
.venv/bin/python -m unittest discover -s tests -v
```

## Espectro corrigido

O gerador usa JONSWAP parametrizado por Hs, Tp e gamma conforme Fossen MSS,
`waveSpectrum.m`, caso 7, com compensação DNV:
`alpha = (5/16) Hs² wp⁴/g²` e `A_gamma = 1 - 0.287 ln(gamma)`.
Fonte: https://github.com/cybergalactic/MSS/blob/master/LIBRARY/environment/waveSpectrum.m

`alpha_js` dos YAMLs antigos é aceito por compatibilidade, mas não controla mais
a escala. A altura espectral é aproximada devido à compensação analítica,
à discretização e ao corte em 3 wp; não há normalização numérica posterior.
`scale`, filtros, limites de corrente e ganhos de carga continuam ativos: Hs
parametriza o espectro de elevação, não uma força aplicada diretamente.

Essa mudança também afeta futuros treinamentos. Os modelos existentes foram
aprendidos no ambiente anterior; esta campanha mede sua resposta no ambiente
corrigido, sem retreiná-los. Ela não passa a representar um treino com as alturas
corrigidas. O campo continua avaliado em ponto fixo, mas as velocidades filtradas agora
são rotacionadas do referencial inercial para o corpo. Ambos usam Z positivo
para cima. A versão física também corrige Coriolis, inércia e restauração;
consulte `docs/physics_conventions.md`.

Os CSVs incluem seed de distúrbio, repetição, cenário e tempo de CPU por passo.
Com uma realização por condição, a dispersão dos gráficos representa variação
entre trajetórias/cenários, não incerteza de repetições independentes.
