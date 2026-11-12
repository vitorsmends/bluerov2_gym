# Avaliações após o fine-tuning exclusivo na estrela

Execute na raiz do repositório, com o venv ativado. A rotina não treina agentes,
não reexecuta PID/SMC/NMPC e não sobrescreve o artigo ou as avaliações antigas.
Exige que o fine-tuning tenha `status: completed` e verifica a linhagem do modelo
original por hashes de checkpoint e normalizador.

## Preparar e conferir

```bash
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase plan
```

Configuração: `configs/evaluation/finetune_article_campaign.yaml`. Saídas padrão:
`artifacts/evaluation/finetune_article_campaign/`. Use `--output` com outra pasta
para uma campanha distinta. Repetir uma fase reaproveita avaliações completas
compatíveis, conferindo os hashes. Uma execução incompleta não é silenciosamente
sobrescrita.

## 1. Comparar os quatro checkpoints

```bash
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase checkpoints
```

Modelos, cada qual com seu próprio normalizador:

1. `ppo_trajectory_curriculum_stage_1.zip`: preparação A, sem ondas (seu “stage 0”).
2. `ppo_trajectory_curriculum_stage_2.zip`: preparação B, com a dinâmica simplificada
   e o mar daquela fase.
3. `ppo_trajectory_curriculum.zip`: final do currículo treinado no oito.
4. `ppo_trajectory_finetuned.zip`: final do fine-tuning somente na estrela.

**Todos são avaliados na mesma física-alvo atual**, em Moderate, com seed 42,
100 s e as referências oito, quadrado, círculo e estrela. Não avaliamos cada rede
na física em que foi treinada: isso confundiria a comparação. São 16 episódios.
O gráfico compacto apresenta RMSE de posição e MAE de yaw em matrizes 4×4.
Episódios interrompidos mostram sua duração e não recebem RMSE de horizonte
completo. CSVs e hashes são preservados. É uma comparação descritiva de checkpoints,
não uma ablação causal do currículo nem uma avaliação de várias seeds de treino.

Figura: `checkpoint_comparison/checkpoint_comparison.pdf` e `.png`.

## 2. Gerar somente os novos CSVs PPO e combinar com os outros controladores

```bash
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase controllers
```

Executa 27 episódios PPO (três mares × três trajetórias × três repetições), mais
três episódios PPO na estrela em Moderate. Reutiliza os CSVs arquivados de PID,
SMC e NMPC das campanhas do artigo. As pastas `combined_benchmark` e `combined_star`
contêm as quatro estratégias, com o PPO antigo substituído **somente nessas novas
pastas**. Os arquivos originais não são alterados.

Antes de reutilizar os resultados, verifica configuração física, horizonte/dt,
cenários selecionados, referências/parametrização, seeds por repetição, séries de
referência nos CSVs, condições iniciais, dados finitos e horizontes completos.
Os arquivos copiados ficam identificados por hashes em `reuse_provenance.yaml`.
Não se garante identidade de fontes antigas que não foram arquivadas com hashes.
Os tempos de CPU dos controladores reutilizados são da execução anterior; o novo
PPO tem tempos medidos agora. A análise de precisão pode ser pareada, mas uma nova
alegação de velocidade relativa exige considerar hardware/software e campanhas.

## 3. Refazer a avaliação de seeds para a política final

```bash
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase seeds
```

Somente PPO: 20 realizações de ondas em Moderate para oito, quadrado, círculo e
estrela (80 episódios). As seeds são 3000000000–3000000019, fora do intervalo de
ondas de treino, e são confrontadas com as avaliações dos dois treinamentos.
Nenhum controlador clássico nem checkpoint inicial é reavaliado nas 20 seeds.
Isso mede sensibilidade ambiental de uma política, não variabilidade de agentes
independentemente treinados.

## 4. Plots e material para o artigo

```bash
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase plots
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase article
```

Gera novamente os plots tradicionais de erro/yaw com sombras, boxplots, energia,
CPU, Pareto e trajetórias; a comparação dos checkpoints usa sua figura própria.
`article_candidate/` reúne as figuras para o artigo, incluindo a figura espacial
em Moderate e o novo diagnóstico de seeds. `checkpoint_comparison.tex` apresenta
a comparação dos quatro modelos **antes** da avaliação de múltiplas seeds.

O candidato não substitui automaticamente o manuscrito: as informações do
fine-tuning, hashes e valores numéricos precisam acompanhar as novas figuras.
Estrela passa a ser geometria treinada; círculo e quadrado continuam testes de
transferência. A avaliação original e a tese permanecem intactas.

## Executar tudo em sequência

```bash
python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase all
```

Este comando executa 126 episódios novos, incluindo os 16 de checkpoints, e
reutiliza os demais controladores. Não modifica a rede nem inicia outro treino.

O candidato também inclui `tables/benchmark_results.tex`, resumos por mar e
pooled, e métricas por execução. Eles são calculados a partir dos novos conjuntos
combinados, preservando a interpolação T200 de 16 V e o peso igual de cada run.
A preparação dos scripts foi conferida com quatro testes sintéticos de integridade
e com os contratos/CSVs arquivados; as campanhas completas não são executadas
pela fase `plan`.
