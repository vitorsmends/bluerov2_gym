Figuras de uma execução de treinamento, seed 42; sem média entre seeds de PPO.

curriculum_learning: reward bruto e erros de posição por rollout. Traços claros:
valores registrados; traços fortes: média móvel de até 10 registros, separada
por fase. Fundos e pontilhados indicam mudanças de mecânica/estado do mar.

curriculum_validation: política determinística e normalização congelada em
episódios de 100 s. RMSE usa o intervalo após 10 s de acomodação. Estrelas:
baterias aprovadas pelo critério completo, incluindo grupos. Os CSVs também
contêm RMSE incluindo o transitório. Sucesso não é uma estimativa entre agentes.

curriculum_sample_cost: passos de atualização e de validação por fase, separados.
A contagem global do PPO não inclui as interações adicionais de validação.
