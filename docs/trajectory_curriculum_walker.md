# Curriculum PPO com casos Walker

Novo perfil: configs/training/physics_wave_curriculum_walker.yaml (também padrão em trajectory_curriculum.yaml). Modelo iniciado do zero em artifacts/training/physics_wave_curriculum_walker; anterior interrompido e preservado em physics_wave_curriculum_unfiltered.

| Fase | Mecânica | Ondas |
|---|---|---|
| A | std | Sem ondas |
| B | std | W1 |
| C1 | alvo | W1: Hs=2.78 m, Tp=7.1 s |
| C2 | alvo | W2: Hs=3.47 m, Tp=9.5 s |
| C3 | alvo | W3: Hs=3.24 m, Tp=11.1 s |
| C4 | alvo | Mistura uniforme W1/W2/W3 |

A fase final avalia separadamente os três casos (24 episódios; oito por grupo). Mantidos gates de posição, três batches consecutivos nas fases alvo, normalizador congelado durante avaliações e revisão de fases alvo anteriores com probabilidade 0.3. Não há orçamento rígido de passos. Preparações exigem dois batches.

PPO, hiperparâmetros e recompensa permanecem os do artigo, incluindo yaw relativo com peso 0.5. Reset e feedback do código 889 são mantidos; uma referência configurável mantém o oito a -5 m sem rampa de profundidade. A fonte histórica permanece intacta.

Ondas: 54 m de lâmina d'água, amostragem fixa em (0,0,-5) m, normalização de Hs, gamma=3.3, unidirecionais, kernel Airy, rotação para o corpo, escala 1, sem filtros nem ruído adicional. max_current=1e6 representa cap não ativo. Mantidos os ganhos/caps de carga empírica e a mecânica do veículo. Isso não replica exatamente espectros de boia, carregamento experimental, nem o NMPC com previsão de Walker; ver docs/analysis/walker_wave_scenarios.md.

O treinamento anterior do artigo continua documentado e reproduzível via trajectory_curriculum_pre_walker.yaml e sua configuração arquivada. Os resultados publicados não foram substituídos pelos deste treinamento novo.
