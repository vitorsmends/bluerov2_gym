Gráficos tradicionais da avaliação physics_wave_curriculum_comparison.

As figuras na raiz agregam três trajetórias e três repetições (nove execuções
por controlador). As subpastas por trajetória usam somente suas três repetições.
Curvas: média; sombras: mais/menos um desvio-padrão populacional (ddof=0), como
nas rotinas históricas. Não são intervalos de confiança nem variância entre seeds
de treinamento do PPO. Há um único agente PPO, avaliado em realizações distintas.

Boxplots: um RMSE por execução; três pontos por controlador em cada trajetória,
nove por controlador no resumo geral. Energia: estimada pela interpolação da
curva elétrica do T200 a 16 V, integrada no tempo; não é medição de bateria.
Pareto: RMSE versus tempo médio de CPU por ação. A comparação inclui PID, SMC,
NMPC e PPO, no cenário target da configuração resolvida da avaliação.
