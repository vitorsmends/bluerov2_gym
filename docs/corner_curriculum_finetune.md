# Fine-tuning curto em trajetórias com quinas

Configuração: `configs/training/corner_curriculum_finetune.yaml`.
Origem: modelo final de `progressive_wave_exposure`, após 10.199.040 passos.
O modelo original e sua normalização não são sobrescritos.

- Treinamento: quadrado e estrela, sorteados com probabilidade 50% por episódio.
- Três fases: 30%, 60% e 100% do Hs de W1/W2/W3. Caso de mar sorteado
  uniformemente por episódio; períodos e outros parâmetros permanecem fixos.
- Mecânica de validação, kernel Airy e rotação para o corpo; sem filtros nem
  ruído adicional. Alocação de cinco ações e recompensa de yaw preservadas.
- Profundidade −5 m; quadrado de lado 1,6 m/período 80 s; estrela de raios
  1,2/0,45 m/período 100 s, iguais aos casos da campanha comparativa.
- Episódios de 120 s. 75% dos resets de treinamento ficam 1–5 s antes de uma
  quina; os demais sorteiam a fase uniformemente. A política não recebe o
  instante da próxima quina nem preview da referência.
- Pesos, estado do otimizador e normalização carregados da origem. Overrides
  explícitos: learning_rate=3e−5, target_kl=0,01, ent_coef=0, n_epochs=5.
  Foram verificados no modelo salvo, e não apenas no YAML.

Antes de adaptar, avalia-se a política de origem nas três fases. As avaliações
posteriores ocorrem a cada 10.240 passos: 24 episódios, distribuídos igualmente
entre quatro geometrias (oito/quadrado/círculo/estrela) e três mares. Oito e
círculo são somente avaliações de retenção, sem atualizações de gradiente.
As avaliações têm 120 s, excluindo 10 s de acomodação. A janela de três lotes
exige aprovação global e por combinação mar/geometria. Cada episódio precisa
completar o horizonte, RMSE ≤0,2 m, pelo menos 90% do tempo dentro de 0,2 m,
yaw MAE ≤12 graus e, nos polígonos, RMSE ≤0,3 m nos 3 s após as quinas.
O protocolo identifica regressões de retenção, mas não garante sua ausência.

O early stopping de fase tem cinco avaliações de aquecimento e paciência de
15 avaliações sem melhora significativa em sucesso global, pior combinação,
RMSE, yaw MAE ou RMSE após as quinas. Os incrementos mínimos são 2,5 pontos
percentuais, 0,005 m e 0,25 grau. Melhora em qualquer indicador renova a
paciência; recuperar um valor antigo não conta como novo progresso.
Estagnação permite seguir à próxima intensidade sem declarar domínio.
A execução termina após a última fase, distinguindo domínio de mera exposição.
Os 500 mil passos por fase são somente uma previsão, não um orçamento rígido;
o treino pode terminar antes ou durar mais enquanto houver progresso.

```bash
bluerov train --config configs/training/corner_curriculum_finetune.yaml \
  --output artifacts/training/corner_curriculum_finetune

python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/corner_curriculum_finetune \
  --host 0.0.0.0 --port 51349 --no-open
```

Depois de terminar, a mesma campanha com PID/SMC/NMPC pode avaliar o FT:

```bash
python -m bluerov2_gym.workflows.evaluation.wave_article_campaign \
  --training-run artifacts/training/corner_curriculum_finetune \
  --output artifacts/evaluation/corner_curriculum_article

python -m bluerov2_gym.workflows.plotting.wave_article_results \
  --campaign artifacts/evaluation/corner_curriculum_article
```

Os CSVs do PPO de origem permanecem disponíveis para comparação pareada
antes/depois, sem escolher retrospectivamente a política pelos casos de teste.
