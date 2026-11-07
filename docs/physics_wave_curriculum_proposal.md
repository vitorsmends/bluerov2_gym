# Proposta: currículo de mecânica e perturbações para PPO de trajetórias

Proposta de método. A implementação padrão foi adicionada em 07/11/2026;
veja trajectory_curriculum.md e configs/training/trajectory_curriculum.yaml.
Os baselines e a versão anterior do currículo permanecem separados.
Referência empírica: treino std e treino corrected com seed 42 e mesma receita
889a766; o primeiro transferiu melhor no quadrado na avaliação atual. Um par de
seeds não prova superioridade estatística nem identifica a correção responsável.

## Fundamentação e hipótese

Curriculum Learning organiza tarefas intermediárias para facilitar uma tarefa
alvo. A formulação de famílias de tarefas e transferência está sistematizada em
[Narvekar et al., JMLR 2020](https://www.jmlr.org/papers/v21/20-212.html).
[Klink et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)
propõem adaptar a distribuição de tarefas à competência do agente aproximando-a
progressivamente da distribuição alvo. A ADR amplia distribuições de ambientes
randomizados durante a aprendizagem em
[Akkaya et al., 2019](https://arxiv.org/abs/1910.07113).
[Portelas et al., CoRL/PMLR 2020](https://proceedings.mlr.press/v100/portelas20a.html)
propõem ALP-GMM, com seleção de contextos por progresso absoluto de aprendizagem.

Aqui propomos uma sequência discreta desenhada manualmente com progressão
controlada por avaliações independentes. É inspirada nesses princípios; não
implementa formalmente SPDR, ADR ou ALP-GMM. Proporções e limiares abaixo são
hipóteses de engenharia a validar, não valores prescritos pelos artigos.

Hipótese: a pré-aprendizagem de tracking em mecânica std permite inicializar uma
política que se adapta à mecânica corrected e suporta perturbações crescentes,
com maior sucesso ou menor custo amostral que aprender diretamente no destino.
O currículo também pode causar transferência negativa; isso deve ser medido.

Família de tarefas: M(theta), theta=(mecânica, intensidade de perturbação,
Tp, direção, espalhamento, realização aleatória, fase inicial). Objetivo final:
maximizar sucesso na física corrected e distribuição alvo de ondas especificada
ANTES dos experimentos. Não é maximizar sucesso médio misturando física antiga
e nova. Na versão atual da proposta, o destino usa espectro normalizado para Hs
controlar energia; essa mudança é explicitamente registrada na transição B→C1.

## Sequência recomendada

Atualização de 07/11/2026: duas etapas preparatórias seguidas de quatro níveis
de mar na física atual, seis fases no total. O distúrbio dos quatro níveis finais
usará normalização de espectro explícita para que Hs controle sua energia.

| Etapa | Mecânica | Perturbação | Objetivo |
|---|---|---|---|
| A | std, coeficientes train_stable_traj | Sem ondas/ruído/corrente | Aprender tracking básico |
| B | std, mesmos coeficientes | Configuração train_stable_traj | Consolidar a fonte validada |
| C1 | corrected atual | Mar baixo, espectro normalizado | Adaptar ao destino |
| C2 | corrected atual | Mar intermediário | Ampliar domínio |
| C3 | corrected atual | Mar elevado | Robustez |
| C4 | corrected atual | Distribuição operacional alvo | Consolidação final |

B preserva um episódio-âncora com os parâmetros e seed 42 da branch: Hs=2,
Tp=12, scale=0.5, max_current=0.7, noise_std=0.01, ganhos e limites históricos.
Os demais episódios podem usar novas realizações de ondas com parâmetros
físicos iguais. O episódio-âncora serve à rastreabilidade; não deve ser todo o
conjunto de treino/validação. Em C1, reduzir o mar ao mudar a física evita mudar
dois fatores difíceis simultaneamente.

Não interpolar arbitrariamente forças restauradoras e operadores de Coriolis
entre versões para a primeira implementação. A física std/corrected é uma
escolha por estágio, aplicada só em reset entre episódios e após concluir ou
descartar corretamente o rollout anterior. O modelo de massa/dinâmica deve ser
reconstruído consistentemente; atribuir somente dynamics.mechanics_model não
recalcula matrizes de massa já construídas nem os coeficientes.

Evitar inicialmente misturar episódios de std e corrected no mesmo estágio.
O feedback atual não identifica a mecânica; a mesma observação pode exigir
ações diferentes, inclusive por causa do sinal histórico de empuxo. Na etapa
corrected, revisitar condições de mar fáceis USANDO corrected. Regressar a
condições fáceis não exige voltar a otimizar a física antiga.

## Intensidade das ondas: particularidade indispensável do código

A escolha atual para C1–C4 é normalize_spectrum=true, com os demais flags de
ondas fixados explicitamente para a distribuição alvo. A alternativa beta abaixo
fica registrada como opção para preservar ondas std; não é a escolha principal
do currículo atualizado. A normalização é uma mudança deliberada na transição
B→C1 e deve constar no protocolo; baixar Hs em C1 reduz a perturbação durante
a adaptação, mas não transforma essa transição em uma ablação de fator único.

Com disturbance_model.preset=std, o espectro usa alpha_js fixo e não usa Hs
para normalizar energia. Aumentar somente Hs não implementa a dificuldade que
seu rótulo sugere. Hs=0 sozinho também não garante perturbação nula.

Definir um envelope de perturbação beta em [0,1]. Para o algoritmo atual, um
primeiro mapeamento controlado pode ser:
- scale(beta)=beta*0.5;
- max_current(beta)=beta*0.7;
- noise_std(beta)=beta*0.01;
- ganho, Tp, direção média, filtros e limites de força mantidos;
- beta=0: enable_wave_force=false, scale=max_current=noise_std=0;
- beta=1: configuração operacional histórica de referência.

Esse beta mede intensidade de perturbação no simulador, NÃO altura significativa
física. O mapeamento muda deliberadamente a água/ruído de maneira conjunta e
deve ser documentado. Um estudo posterior pode separar ondas e ruído em eixos.
Registrar RMS e quantis de nu_current e tau_wave, além das frações de saturação,
para verificar a intensidade efetiva. Saturação e não linearidades impedem
assumir dificuldade estritamente proporcional a beta.

Níveis iniciais possíveis: 0, 0.25, 0.5, 0.75 e 1.0. Limites adicionais acima de
1 devem ser definidos a partir do domínio operacional, não da palavra storm.
Primeiro aumentar amplitude com Tp=12 e direção fixos. Só depois ampliar Tp,
direções e espalhamento. Tp menor e outras direções não são necessariamente
mais difíceis para toda política; medir a dificuldade, não presumir ordenação.
Se o objetivo futuro for Hs fisicamente consistente, ativar/validar a normalização
do espectro como experimento explícito, e não alterar esse modelo silenciosamente.

## Critério de domínio e controle do currículo

A política é avaliada deterministicamente em ambiente separado, com uma cópia
congelada do VecNormalize corrente. Nenhuma avaliação atualiza PPO ou estatísticas.
Um piloto dimensiona os limiares para um requisito físico definido; o piloto não
usa o conjunto de teste final. Números abaixo são um ponto de partida explícito:

- Avaliar a cada 20 mil interações globais aproximadamente, em fronteira de rollout.
  Esse intervalo não é um limite de passos da fase.
- Episódios de AVALIAÇÃO de 100 s, com 10 s iniciais separados como transitório.
  Oito de treino tem período de aproximadamente 41.9 s; 100 s cobrem cerca de
  2.4 voltas. O aumento solicitado refere-se ao orçamento de interações do
  treinamento, não à duração dos episódios.
- Episódio aprovado: completar a duração sem falha física; RMSE de tracking <=0.20 m
  e >=90% das amostras de tracking com erro <=0.20 m após o transitório.
- Exigir >=90% dos episódios aprovados em cada grupo da validação, em três baterias
  consecutivas de 20 episódios com realizações distintas. Distribuir casos para
  que uma direção ou fase fácil não esconda falha em outro grupo.
- Registrar também RMSE total (incluindo transitório), erro de profundidade/yaw,
  esforço, saturação e taxa de falha. O transitório não desaparece dos relatórios.
- Avançar somente após domínio; se falhar, continuar no nível. Opcionalmente
  usar histerese para reduzir a intensidade de ondas após queda persistente,
  com regra fixa e registrada. Não avançar automaticamente por orçamento.
- Horizonte planejado de 3.000.000 de passos por fase (antes: 1.000.000 por
  treino de referência). É um valor proposto para planejamento, não um gatilho
  de avanço nem um mínimo obrigatório. Se a fase não for dominada nesse ponto,
  pode continuar; a configuração operacional deverá explicitar essa extensão.
  Não há teto rígido nesta proposta nem avanço automático pelo contador de passos.
- Early stopping da FASE por domínio: encerra a fase atual, salva PPO/normalizador,
  marca mastered e continua treinando o mesmo agente na fase seguinte. Na última
  fase, encerra o treino com completed. A sequência de aprovações zera se uma
  bateria falhar; não se acumulam sucessos antigos de políticas distintas.
- Early stopping por estagnação é outro recurso, inicialmente DESATIVADO.
  Se habilitado e acionado, marca incomplete e não aprova nem pula a fase.
  Não se reduz a exigência automaticamente para evitar uma fase longa.

Os limiares de 20 cm podem ser exigentes. Se o piloto mostrar que não correspondem
ao requisito atingível, a revisão deve ser justificada e fixada antes do estudo
principal, nunca relaxada automaticamente para fabricar progresso.

Ao desbloquear um nível de ondas, uma proposta simples é 70% de episódios no
nível novo e 30% em níveis anteriores, sempre com a mesma mecânica do estágio.
São episódios NOVOS com a política corrente, não replay de transições antigas.
O PPO coleta novas interações para atualizar sua política, conforme
[Schulman et al., 2017](https://arxiv.org/abs/1707.06347).
A proporção deve ser configurável e pode ser objeto de ablação.

## O que manter invariável na primeira versão

Mesma rede 64/64, PPO 889a766, ação em N, dt, limites, figura oito, reward,
transformação de observações e distribuição de reset. Não adicionar contexto
privilegiado de ondas ou mecânica à observação. Preservar pesos do ator/crítico,
otimizador, contagem global de passos e VecNormalize entre estágios. A avaliação
congela uma cópia da normalização e a política; o treino pode continuar atualizando
suas estatísticas. Não reinicializar tudo ao trocar a fase.

Há uma mudança de protocolo a assumir explicitamente: episódios temporais de
100 s tornam o domínio por episódio bem definido, mas o original 889a766 não
aplicava TimeLimit. Os baselines usados para avaliar o efeito do currículo devem
receber o mesmo horizonte, wrapper e tratamento de truncamento. Separar truncamento
por tempo de término por falha para não alterar incorretamente o bootstrap do PPO.
Se se desejar preservar treinamento sem TimeLimit, usar rollouts de avaliação
finitos para decidir domínio, em vez de impor horizonte novo só ao currículo.

Para máxima continuidade com os testes atuais, a PRIMEIRA implementação pode
manter o treino sem TimeLimit e usar avaliação de 100 s como gate. A mudança de
mecânica ocorrerá em um reset explícito na fronteira de rollout, e será registrada.
Isso evita confundir currículo com uma nova regra de truncamento de episódios.

Não introduzir inicialmente rotação de trajetórias: o treino fica na figura oito.
As demais trajetórias ajudam a medir generalização. Um currículo de trajetórias
pode ser adicionado depois como fator independente.

## Desenho experimental

Comparações mínimas:
1. Treino direto em corrected com mar alvo, sem currículo.
2. Treino std e transferência sem adaptação (baseline atual).
3. Corrected desde o começo, apenas currículo de ondas.
4. Proposta completa: std sem ondas -> std com ondas -> corrected com ondas baixas
   -> corrected com ondas crescentes.

Ablação útil: remover A, ou remover a repetição de níveis anteriores. Comparar
pelo menos cinco seeds de PPO pareadas entre métodos, preferencialmente dez
quando houver recursos. Todas recebem o mesmo protocolo de avaliação. Para
comparar custo amostral, contar TODAS as interações do pré-treino e adaptação e
comparar as curvas nas mesmas contagens globais; o currículo não ganha passos
extras ocultos. Isso não impõe avanço por orçamento no treinamento operacional.

Separar:
- treino: episódios de atualização;
- validação/gate: decide avanço e seleção de checkpoint;
- teste final: sementes/casos reservados, nunca usados para decisões.

Um mesmo conjunto fixo de validação pode ser sobreajustado por consultas repetidas;
usar baterias de validação rotativas e manter um teste realmente reservado. O
quadrado já examinado muitas vezes é diagnóstico/validação, não teste final intocado.
A política final deve ser avaliada exclusivamente no simulador corrected do alvo,
com métricas por trajetória, mar/direção e seed, intervalos de incerteza sobre seeds
de TREINO, taxa de conclusão do currículo e tempo/interações até o domínio.
Não tratar 1000 instantes de uma trajetória como 1000 execuções independentes.
Não excluir seeds que estagnaram ou não chegaram ao último estágio.

## Estado do código

O trajectory_curriculum atual já tem seleção de trajetórias, níveis de mar e
critério por episódios do TREINO. Ele não troca a mecânica entre estágios, nem
usa um gate de avaliação independente, nem distribui episódios entre níveis
anteriores. Os níveis Hs da configuração não garantem aumento de energia no
preset std. Portanto não basta renomear fases do YAML existente.

A proposta exige um agendador de contexto, fábrica de dinâmica por estágio,
avaliação independente, registro das distribuições/critério e reprodução completa
(checkpoint do PPO, normalizador, etapa, RNG e parâmetros efetivos). Tudo deve ficar
em perfil novo para preservar os baselines já testados.

## Maior orçamento de passos e avanço antecipado por domínio

Correção de entendimento: manter o horizonte de avaliação em 100 s. O aumento
solicitado é no número de interações do PPO, total_timesteps, e não no horizonte
do episódio nem no parâmetro n_epochs. Este último permanece 10: dez passagens
de otimização pelos dados de cada rollout de 2048 passos, com batches de 64.

A proposta passa a planejar 3 milhões de passos por fase. Com seis fases, são
18 milhões de passos como referência de planejamento; o custo real pode ser
menor pelo avanço por domínio ou maior se uma fase precisar continuar. Isso
não configura um limite global obrigatório. Preserva-se a continuidade da rede,
otimizador, normalização e contador global entre fases. A última fase encerra
com completed apenas ao atingir o critério físico de domínio.

O painel deve mostrar fase atual, passos acumulados na fase, referência de
planejamento, tempo gasto e sequências de avaliação aprovadas (por exemplo, 2/3).
Um percentual relativo ao horizonte planejado não representa percentual de
domínio; se o treino puder ultrapassá-lo, uma ETA deve ser identificada como
previsão do orçamento, não como previsão de aprovação.

O protocolo está implementado no perfil trajectory_curriculum.yaml. Os treinadores
de referência permanecem separados. Esta implementação não inicia um treino longo
automaticamente; os comandos estão em trajectory_curriculum.md.
