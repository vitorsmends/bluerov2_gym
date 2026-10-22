# Comparação dos controladores: moderate e severe

## Retreinar o pipeline validado do commit 26857c4 na Physics 2 (seed 42)

```bash
.venv/bin/python train_ppo_seed42.py
```

Restaura o treinamento e a interface PPO de
`26857c4456e11c6173b04678b21b6806d77b5b4c`: mesma rede, recompensa,
normalização, hiperparâmetros, currículo, transições entre estágios e
observação usada na avaliação. A única diferença funcional intencional é a
dinâmica Physics 2 carregada de `config/dynamics_config.yaml`. O limite efetivo
permanece em 5000 passos e o orçamento é de 3 milhões. Os arquivos são salvos
separadamente em `training_runs/ppo_26857_physics2_seed42/`.
O script também exige as versões do ambiente do modelo validado, registradas em
`requirements-ppo-reference.txt`, para evitar alterar o PPO por atualização de
Python, Gymnasium, Stable-Baselines3, Torch, NumPy ou SciPy.


Para treinar e, ao terminar, executar a comparação dos quatro controladores
usando o novo modelo e gerar os gráficos, use em vez do comando anterior:

```bash
.venv/bin/python train_ppo_seed42.py --evaluate
```

Os resultados dessa avaliação ficam na subpasta `evaluation/` do treino.
Para treinar e executar somente o novo PPO, reutilizando os dados já salvos dos
outros três controladores e gerando a comparação completa:

```bash
python train_ppo_seed42.py --reuse-results training_runs/ppo_physics_v2_seed42/evaluation
```

O comando valida a física, configuração e integridade dos 30 CSVs convencionais
antes de treinar. Copia esses CSVs para a nova avaliação, executa os dez casos
PPO e gera gráficos e tabelas com os quatro controladores. Preserva os dados
originais e registra os hashes dos arquivos reutilizados no manifesto.

Se já existir uma execução nessa pasta, escolha outro destino com
`--output-dir training_runs/ppo_26857_physics2_seed42_run2`.
`--smoke-test` executa apenas 320 passos para verificar o pipeline e salva em
uma subpasta `smoke_test/`; esse modo não constitui um treinamento utilizável.

## Executar a comparação

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

Por padrão, utiliza o modelo retreinado e o normalizador de
`training_runs/ppo_physics_v2_seed42/models/`, identificados no manifesto atual. Para selecionar outra política fixa, informe o par:

```bash
.venv/bin/python run_oceans.py \
  --model models/bluerov_ppo_curriculum_seed_42.zip \
  --vecnormalize models/bluerov_vec_normalize_curriculum_seed_42.pkl \
  --results-root evaluation-new-run
```

Saídas padrão:

- `training_runs/ppo_physics_v2_seed42/evaluation/moderate/trajectory_experiments/`: CSVs moderate.
- `training_runs/ppo_physics_v2_seed42/evaluation/severe/trajectory_experiments/`: CSVs severe.
- `training_runs/ppo_physics_v2_seed42/evaluation/manifest.yaml`: configuração, parâmetros de mar e hashes do
  modelo e normalizador.
- `training_runs/ppo_physics_v2_seed42/evaluation/figures/`: gráficos agregados e trajetórias dos dois cenários.
- `training_runs/ppo_physics_v2_seed42/evaluation/tables/`: tabelas RMSE e Pareto; tabelas espaciais também são
  salvas nas subpastas de cada cenário.

Uma pasta de resultados não vazia é recusada para preservar campanhas anteriores.
Use outro `--results-root` para uma nova execução. As campanhas antigas foram removidas. A campanha atual e seus artefatos de
treinamento são preservados; o artigo recebe cópias dos gráficos atuais.

Para executar somente a coleta, adicione `--skip-plots`. Para gerar novamente
os gráficos de uma campanha concluída:

```bash
.venv/bin/python run_oceans.py --plots-only --results-root training_runs/ppo_physics_v2_seed42/evaluation
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
