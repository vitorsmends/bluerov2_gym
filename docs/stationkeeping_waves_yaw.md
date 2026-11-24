# Station keeping com o currículo de física e ondas do artigo

Esta campanha é exclusiva da tese. Usa o mesmo motor de currículo de `progressive_wave_exposure`, com 16 estágios, avaliações independentes, detecção de progresso e avanço para exposição após estagnação. Substitui o perfil anterior de 19 etapas com aprovação por episódios de treino; esse perfil foi arquivado em `_references/stationkeeping_episode_curriculum/`.

## O que permanece específico de station keeping

- PPO do zero, seed 42, ator e crítico 256×256, ações diretas nos seis propulsores, ±40 N; hiperparâmetros e VecNormalize da receita antiga de station keeping (rollouts 4096, batch 256, 10 epochs).
- Alvo fixo `(0,0,-5)` m, yaw 0°; início com desvios uniformes de até ±(0.3,0.3,0.15) m e yaw até ±30°. Episódios de treino duram até 100 s ou falha física.
- Recompensa de regulação preservada e termo de yaw igual ao da política recente de trajetórias: `-0.5 * abs(wrap(yaw - yaw_ref))`, em radianos. Não usa o reward de trajetória.
- Observações: erros de posição inerciais, yaw relativo com wrap, roll/pitch e velocidades no corpo. A avaliação usa o mesmo adaptador `stationkeeping` desde o primeiro instante.
- O motor compartilhado usa internamente nomes de métricas de trajetória para compatibilidade; nesta campanha a referência é estritamente fixa.

## Currículo idêntico ao artigo

| Estágios | Mecânica | Ondas |
|---|---|---|
| A | Simplificada | Sem ondas |
| B1–B2 | Simplificada | W1–W3 com 5% e 10% de Hs |
| C | Completa | Sem ondas, para transferência |
| D1–D12 | Completa | W1–W3 com 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% de Hs |

Os estágios físicos e todos os parâmetros de avaliação, avanço e estagnação são copiados do YAML do artigo. Nos estágios com ondas, W1/W2/W3 são amostrados uniformemente no treinamento e avaliados separadamente. Mantêm-se normalização espectral, Airy em profundidade finita (54 m), amostragem fixa a −5 m, rotação para o corpo, ausência de filtros e ruído adicional. A preparação emprega exatamente a aproximação mecânica e os coeficientes de amortecimento usados no artigo.

- Avaliações com ações determinísticas e normalização congelada, em ambientes separados, por 100 s; primeiros 10 s excluídos das métricas de domínio.
- Aprovação de episódio: completar o horizonte sem falha, RMSE de posição ≤0.20 m e pelo menos 90% dos pontos avaliados dentro de 0.20 m.
- Cada avaliação usa 12 episódios sem ondas ou 24 com ondas (8 por W). Janelas de três avaliações exigem ≥90% de sucesso global e por caso; são necessárias três janelas aprovadas sem sobreposição, como no artigo.
- Yaw entra na recompensa, nos logs e nas estatísticas das avaliações; **não foi acrescentado um requisito angular ao critério de aprovação**, que permanece igual ao artigo.
- Avaliações após blocos nominais de 10.000 passos sem ondas ou 20.000 com ondas, arredondados para rollouts completos: 12.288 e 20.480 passos nesta receita de PPO.
- Estagnação considera sucesso global, pior caso e RMSE, com deltas de 0.025 e 0.005 m, aquecimento de 10 verificações e paciência de 30. Aquisição inicial exige 50% de sucesso para ativar a paciência; fases com ondas podem ativá-la com sucesso zero, como no artigo.
- Estagnação avança para exposição ao estágio seguinte, sem marcar domínio. O encerramento distingue `completed` e `exposure_completed`. Não há limite obrigatório de passos por fase: os 3 milhões são uma referência de planejamento, não um orçamento de parada.
- As fases das ondas são renovadas por episódio. Seeds de treinamento e avaliação independente ocupam metades distintas do espaço uint32. Política, otimizador e normalização são mantidos entre estágios.

## Early skipping protegido

A partir da fase 5, após pelo menos 12 avaliações independentes, a fase pode avançar para exposição se houver 6 verificações consecutivas sem melhora significativa. O detector de progresso usa os mesmos melhores valores de sucesso global, pior caso e RMSE; retornar a um nível antigo não reinicia a paciência. A regra não atua nas quatro fases iniciais, não atua enquanto houver janela aprovada ou progresso na sequência de aprovações, e requer RMSE finito. A regra original de estagnação com paciência 30 permanece como alternativa. Nenhuma saída antecipada certifica domínio.

Configuração em `early_phase_skipping`: `first_stage: 5`, `min_evaluations: 12`, `patience_evaluations: 6`. Pode ser desativada. Nas fases 5–16, o aquecimento do detector é de 5 verificações, mantendo o aquecimento e a paciência originais nas quatro fases iniciais. O currículo de trajetórias do artigo não foi alterado.

## Continuação iniciada do checkpoint protegido

A execução anterior foi interrompida com salvamento. A continuação parte explicitamente do checkpoint final da fase C (fase 4, 2.211.840 passos), com pesos, otimizador, normalizador e estados de RNG registrados, começando na fase 5. Trata-se de uma continuação com episódio novo, não de um treino do zero nem de reprodução bit a bit da sequência interrompida.

```bash
bluerov train --config configs/training/stationkeeping_waves_yaw_resume.yaml \
  --output artifacts/training/stationkeeping_physics_curriculum_early_skip
```

Não execute o comando acima enquanto a continuação atual estiver ativa. Visualizador já iniciado em `http://127.0.0.1:8769`. A configuração de avaliação agora aponta para essa continuação. O histórico descartado após a fase 4 fica preservado no diretório anterior, mas não faz parte da nova sequência.

## Execução do zero

Para um novo treinamento completo, utilize outra pasta de saída:

```bash
bluerov train --config configs/training/stationkeeping_waves_yaw.yaml \
  --output artifacts/training/stationkeeping_physics_curriculum_from_scratch
```

Não execute novamente enquanto essa execução estiver ativa. Para uma repetição independente, escolha outra pasta de saída e seed conforme necessário.

Visualizador:

```bash
python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/stationkeeping_physics_curriculum_from_scratch \
  --port 8768 --no-open
```

O visualizador histórico da execução interrompida permanece em `http://localhost:8768`; a continuação está na porta 8769. Para acesso na rede, inicie em outra porta com `--host 0.0.0.0` e utilize o IP deste computador.

## Avaliar após o fim

```bash
bluerov inspect --config configs/evaluation/stationkeeping_waves_yaw.yaml
bluerov evaluate --config configs/evaluation/stationkeeping_waves_yaw.yaml \
  --output artifacts/evaluation/stationkeeping_physics_curriculum_early_skip
bluerov plot --run artifacts/evaluation/stationkeeping_physics_curriculum_early_skip \
  --suite stationkeeping
```

São 4 controladores × 3 mares × 5 alvos × 3 realizações = 180 episódios de até 100 s. Todos começam no alvo mais `(0.20,-0.15,0.08)` m e yaw de referência +10°. As posições variam em XY, todas a −5 m e yaw de referência 0°; isso não demonstra generalização a profundidades ou headings diferentes. Seeds de teste 3.000.000.000–3.000.000.002 são emparelhadas e ficam fora da faixa de seeds de ondas de treinamento. Não é uma avaliação de múltiplas seeds de treinamento.

O YAML herda a dinâmica completa do novo treino e carrega o par `ppo_stationkeeping_physics_curriculum.zip` e `vec_normalize_stationkeeping_physics_curriculum.pkl`. Um par final também é salvo após exposição completa, mesmo sem domínio de todos os estágios; consultar o status antes de interpretar os resultados.

Figuras e tabelas incluem posição, yaw (MAE/RMSE em graus), esforço, energia e custo computacional. Sombras representam desvio padrão populacional entre alvos e realizações, não intervalos de confiança. Términos antecipados estão registrados no manifesto e devem ser considerados nas discussões. Os resultados antigos da tese serão substituídos apenas após a nova avaliação; não foram inseridos resultados ainda não obtidos.

## Continuação com menor espera por estagnação

A continuação mais recente usa `configs/training/stationkeeping_waves_yaw_fast_resume.yaml`, que seleciona explicitamente o último checkpoint completo de avaliação da execução `stationkeeping_physics_curriculum_early_skip`. O normalizador, otimizador, estágio, RNG registrado e melhores indicadores de progresso são preservados; as janelas independentes de domínio são preenchidas novamente, como nas demais retomadas. Não se retorna às fases fáceis.

Saída: `artifacts/training/stationkeeping_physics_curriculum_fast_resume`. Visualizador: `http://127.0.0.1:8770`. As avaliações finais continuam com 24 episódios de 100 s e os mesmos limites de aprovação. A aceleração vem de menor espera por estagnação, não de maior velocidade de aprendizado por passo nem de uma garantia de melhor desempenho.

## Histórico total reconstruído

Os logs da linhagem foram consolidados em `artifacts/analysis/stationkeeping_training_history`, uma visualização derivada (não outro treino nem um diretório de modelos):

1. Execução original: até o checkpoint de 2.211.840 passos (final da fase C).
2. Primeira continuação: acima de 2.211.840 até o checkpoint de 3.338.240 passos.
3. Continuação final: acima de 3.338.240 até 6.328.320 passos.

Os passos já eram globais; não foram deslocados nem somados novamente. Dados posteriores ao checkpoint escolhido em cada execução ancestral são excluídos. O total retido é 6.328.320 passos; custo de tentativas descartadas e interrupções não faz parte desse eixo e deve ser contabilizado à parte se discutido. As duas retomadas e alterações dos critérios de progressão devem ser declaradas: não é uma execução sem interrupções.

Gerar a reconstrução novamente exige uma pasta nova:

```bash
python -m bluerov2_gym.workflows.plotting.training.lineage \
  --run artifacts/training/stationkeeping_physics_curriculum_fast_resume \
  --output artifacts/analysis/stationkeeping_training_history_v2 --plots
```

Para replotar a reconstrução já criada:

```bash
bluerov plot-training --run artifacts/analysis/stationkeeping_training_history \
  --suite training_thesis
```

Em `figures/training_thesis/`: `curriculum_learning`, `curriculum_validation`, `curriculum_yaw` e `curriculum_sample_cost` (PDF/PNG). R1 e R2 marcam as retomadas. A suavização dos plots de tese não cruza esses pontos ou as mudanças de fase. O histórico também contém figuras individuais em `figures/training/` e a visão geral em `figures/training_overview/`.

`retained_scalars.csv` identifica a execução de origem de cada escalar. `lineage.json` contém os cortes, contagens e hashes dos arquivos de entrada; `tables/training_thesis/` contém os dados exportados das figuras e avaliações. A reconstrução preserva os timestamps originais, sem interpolação ou alteração dos resultados.

## English plots included in the thesis

```bash
python scripts/plot_stationkeeping_training_thesis.py
```

This renders the reconstructed history in English and exports the PDFs/PNGs to `documents/msc_thesis/images/stationkeeping/training_current/`. The station-keeping training results are included through `chapters/50_stationkeeping_training.tex` in Chapter 5. Trajectory training and its independent wave-seed diagnostics remain in Chapter 6. Chapter 3 contains protocols and architectural definitions, not the observed learning curves. Historical controller-comparison results remain clearly separated pending replacement by the new W1–W3 campaign.

## Atualizar as figuras e tabelas da comparação na tese

Após a avaliação e os plots padrão estarem concluídos:

```bash
python scripts/update_stationkeeping_thesis_results.py
```

O script usa somente os CSVs concluídos, não executa novas simulações. Exporta sete figuras em inglês e a tabela por cenário para `documents/msc_thesis/images/stationkeeping/validation_current/` e `documents/msc_thesis/tables/stationkeeping/validation_current/`. A seção atual é `chapters/50_stationkeeping_comparison.tex`, com protocolo em `chapters/40_stationkeeping_protocol.tex`. Os textos e resultados antigos foram preservados em `archives/before_current_stationkeeping_comparison/`.
