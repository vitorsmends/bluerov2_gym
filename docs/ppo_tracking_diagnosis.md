# Diagnóstico do PPO em trajetórias — treino my_train

**Referência de restauração corrigida pelo usuário:** seguir o treinador curricular do commit `0436ce3482e95105cb2376681b2bd6cbec72d99c`, com PPO 4096/256 e redes pi/vf [256,256]. Os ensaios com ZIPs abaixo são diagnósticos históricos e não definem a receita de treinamento a restaurar. A sugestão de substituir o currículo por cenário fixo foi retirada. Consulte [treinamento](training.md).

**Revisão após auditoria histórica:** as limitações do adaptador descritas abaixo já existiam no commit 26857c4 e não explicam sozinhas a regressão. A política histórica reproduziu bom tracking com esse mesmo adaptador. Consulte a [comparação histórica controlada](historical_training_evaluation_comparison.md) para as diferenças de pesos, receita de treino e física.

O agente aprendeu estabilização na origem, mas a avaliação de trajetórias usa uma transformação de observações que não foi usada nesse treinamento. Há também uma inconsistência comprovada nas velocidades fornecidas ao PPO. O diagnóstico não alterou o comportamento dos controladores nem os arquivos do modelo.

## Compatibilidade verificada

Modelo: `artifacts/training/my_train/models/bluerov_ppo_curriculum_seed_42.zip`, 3.010.560 passos, com seu `bluerov_vec_normalize_curriculum_seed_42.pkl`. O treinamento e a avaliação usam `mechanics_model: corrected` e os mesmos parâmetros de dinâmica do snapshot. Distúrbios usam preset `std`; o cenário de mar pode diferir do último estágio do currículo, o que foi isolado nos testes abaixo. A normalização fica congelada durante a avaliação; nenhuma observação do teste original atingiu o limite de clipping.

O teste dos checkpoints mediu estabilização na origem por 50 segundos. A avaliação mede trajetórias móveis por 100 segundos, incluindo aproximação inicial. Uma curva boa no primeiro teste não demonstra aprendizado do segundo objetivo.

## Problemas encontrados

1. O treinamento recebe posições globais, velocidades BODY e orientação absoluta, e recompensa posição/velocidade/yaw próximos de zero. Não recebe trajetória ou orientação desejadas. O adaptador de avaliação transforma posição em erro no referencial de yaw atual; essa mudança de significado não foi treinada.
2. `build_tracking_observation` subtrai somente a primeira componente de velocidade da referência. Subtrai velocidade WORLD de velocidade BODY e mantém v/w absolutas. A variante experimental mantém o resto do adaptador e calcula os três canais como `nu_linear_body - R_body_to_world.T @ velocity_reference_world`.
3. O adaptador mantém yaw absoluto e ignora yaw desejado. PID/SMC usam a referência de yaw. O PPO tenta manter yaw perto de zero enquanto a referência gira. Apenas subtrair yaw ou trocar referencial de posição não resolveu nos testes: essas alterações também mudam a distribuição das observações e não equivalem a treinamento de tracking.
4. A coluna reward da avaliação continua sendo a recompensa de estabilização na origem, não uma recompensa de tracking. Ela não deve ser interpretada como desempenho de trajetória. Isso não altera as ações determinísticas do PPO.
5. Há inconsistências secundárias: PID/SMC também misturam referenciais no cálculo de velocidade, a métrica legada de erro de velocidade mistura WORLD/BODY, e as referências angulares das trajetórias são zero mesmo quando o yaw desejado muda. O RMSE de posição apresentado aqui é válido e não depende dessas métricas de velocidade. Os outros controladores foram mantidos como estavam para a comparação pareada.

## Experimento pareado

Mesmos pesos, normalizador, dinâmica, trajetórias, estado inicial, mar severe, dt=0,1 s, 1.000 passos e seeds 42/43/44. Números são a média do RMSE por execução, não RMSE agregado de todas as amostras.

| Trajetória | PPO atual | PPO com velocidades relativas | PID | SMC |
|---|---:|---:|---:|---:|
| Quadrado | 0,856 m | 0,683 m | 0,746 m | 0,233 m |
| Oito com profundidade | 0,488 m | 0,307 m | 0,804 m | 0,230 m |
| Círculo | 1,075 m | 0,430 m | 0,917 m | 0,254 m |

A alteração isolada reduz o erro em aproximadamente 20%, 37% e 60%, respectivamente. Ainda não supera o SMC. Isso evidencia um problema na interface, mas não sustenta prometer superioridade do PPO após uma correção simples.

No círculo/seed 42, o adaptador original reproduziu o RMSE da avaliação existente: 1,148 m. No mesmo mar, estabilização na origem com observação original apresentou 0,146 m. Sem ondas/ruído, o círculo original continuou ruim (1,081 m), enquanto a variante com posição global e velocidades relativas apresentou 0,434 m. Usando o mar do estágio final do treino, o círculo original também ficou ruim (1,085 m). Portanto, incompatibilidade do mar não explica sozinha a falha.

Outras ablações no círculo severe/seed 42: mudar apenas o erro de posição para WORLD resultou em 1,160 m; WORLD com velocidades relativas, 0,470 m; manter posição do adaptador e corrigir velocidades, 0,453 m; WORLD com velocidades relativas e yaw relativo, 0,780 m. São testes de uma seed para localizar causas, não campanhas suficientes para escolher uma interface definitiva.

## Artefatos e reprodução

- Comparação: `artifacts/diagnostics/ppo_tracking/my_train_report/figures/paired_rmse.png`.
- Erro, XY e yaw no círculo: `artifacts/diagnostics/ppo_tracking/my_train_report/figures/circle_diagnosis.png`.
- Métricas por seed: `artifacts/diagnostics/ppo_tracking/my_train_report/paired_metrics.csv`.
- Ablações, protocolos e hashes dos modelos: `artifacts/diagnostics/ppo_tracking/my_train_adapter_ablation/`.
- Campanha de velocidades relativas: `artifacts/diagnostics/ppo_tracking/my_train_paired_repetitions/`.

```bash
python scripts/diagnose_ppo_tracking.py --campaign --adapters production body_relative --output artifacts/diagnostics/ppo_tracking/reproduction
```

O diretório de saída precisa ser novo. O script carrega a configuração atual; reproduzir exige preservar configuração, modelos e versões. Os protocolos salvos registram os parâmetros efetivamente usados e hashes dos modelos.

## Encaminhamento

Disponibilizar um adaptador selecionável por configuração, preservando o atual para rastreabilidade, e repetir a comparação com velocidades relativas. Para treinar tracking de fato, alinhar observação e recompensa de treino/avaliação com referências móveis, referenciais coerentes e uma decisão explícita sobre seguir yaw ou apenas posição. O treino deve variar velocidades e trajetórias, e os checkpoints devem ser avaliados nessa mesma tarefa. A transformação de observação isolada não torna o problema de trajetória equivalente ao de estabilização: faltam acelerações/feedforward e a dinâmica em movimento não coincide com a de repouso.
