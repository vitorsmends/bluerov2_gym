# Consolidação e exposição progressiva a ondas

Perfil: `configs/training/progressive_wave_exposure.yaml`. Inicialização do zero,
sem mudanças na arquitetura, recompensa, observações, alocação ou parâmetros
físicos dos 16 estágios do currículo anterior.

## Domínio e experiência são registros distintos

As avaliações mantêm os limites de 0,2 m para RMSE e tolerância de posição,
90% do tempo dentro da tolerância e 90% de episódios aprovados globalmente e
por caso. Oito episódios por caso são avaliados em cada lote. As três avaliações
de uma janela fornecem 24 episódios por caso; a janela exige pelo menos 22/24.

Agora são necessárias **três janelas consecutivas sem sobreposição**, isto é,
nove lotes, para dominar a fase. Janelas intermediárias continuam sendo
mostradas, mas só os lotes 3, 6, 9 etc. atualizam a sequência de domínio.
Um bloco reprovado reinicia essa sequência. Os episódios usam seeds distintas;
as políticas mudam durante o aprendizado, portanto isso avalia estabilidade
operacional, sem constituir uma prova estatística de generalização.

## Estagnação avança a exposição

A paciência é de 30 avaliações sem progresso significativo, após dez avaliações
de aquecimento, monitorando melhor sucesso global, melhor sucesso do pior caso
e melhor RMSE. Os incrementos mínimos continuam sendo 2,5 pontos percentuais
e 0,005 m. Não há teto de passos nem duração mínima fixa por fase.

A primeira aquisição de controle mantém a ativação em 50% de sucesso com RMSE
finito, evitando interromper uma política antes de aprender o básico. No estágio
de transferência sem ondas, a mesma proteção é mantida. Nos estágios com ondas,
a ativação não exige sucesso nem RMSE finito: deterioração em ondas fortes não
impede o monitor de detectar estagnação e liberar a exposição ao próximo nível.

Ao atingir a paciência, a execução não para: a fase recebe
`advanced_on_plateau`, com `mastered: false`, e a política segue para a próxima.
Assim, não dominar 70% não impede contato com 80%, 90% e 100%. Instabilidade
numérica ou falha de execução continuam sendo erros, não motivos para declarar
uma fase aprovada. Se o agente nunca adquirir o controle básico na primeira
fase, a proteção de aquisição pode mantê-lo nela e é necessário acompanhamento.

Ao terminar a última fase, os estados possíveis são:

- `completed`: todas as fases foram dominadas.
- `exposure_completed`: todos os níveis foram visitados, mas há fases não
  dominadas, listadas em `unmastered_stages` e no painel.

O modelo final é salvo nos dois casos. Os checkpoints de cada avaliação e os
motivos de transição são preservados. Exposição adicional não garante melhora
nem impede esquecimento de casos mais fáceis; isso deve ser avaliado depois.
Os perfis anteriores mantêm seu comportamento, pois o avanço por estagnação
e as janelas não sobrepostas são opções explícitas desta configuração.
