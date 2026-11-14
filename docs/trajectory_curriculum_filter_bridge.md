# Currículo com preparação simplificada e retirada do filtro

Configuração: `configs/training/physics_wave_curriculum_filter_bridge.yaml`.
Também é o padrão em `configs/training/trajectory_curriculum.yaml`.
Esta é uma nova proposta experimental; sua convergência ainda precisa ser medida.

## Preparação preservada

As fases A, B e C são as fases A, B e C1 do treinamento
`physics_wave_curriculum_yaw`, incluindo os parâmetros físicos e de ondas:

1. A: mecânica simplificada sem ondas.
2. B: mecânica simplificada, distúrbio filtrado de `train_stable_traj`.
3. C: mecânica de validação, Hs entre 0,2 e 0,5 m, ainda filtrada.

Essas fases mantêm a referência original, com rampa até −0,5 m, incluindo o
ruído artificial da receita original quando aplicável. A mecânica de validação
não é usada para iniciar o aprendizado. A arquitetura, normalização,
recompensa de posição/yaw e hiperparâmetros do PPO são os do artigo.

## Ponte e remoção do filtro

D usa a mecânica de validação, referência a −5 m, kernel Airy, rotação para o
corpo e os parâmetros de W1, mas com Hs entre 0,2 e 0,5 m e filtro α=0,02.
Não há ruído gaussiano adicional. Esta fase separa a transferência de mecânica
da adoção da nova configuração de ondas/profundidade.

E introduz W1, W2 e W3 completos, amostrados uniformemente por episódio, ainda
com α=0,02. F e G mantêm exatamente esses casos e aumentam α para 0,1 e 0,4.
H desliga explicitamente o filtro. A equação aplicada enquanto ele está ativo é

`v_f[k] = (1 − α) v_f[k−1] + α v_wave[k]`.

Logo, aumentar α reduz a suavização. Os valores intermediários são escolhas
experimentais configuráveis, não parâmetros publicados por Walker. Os casos
W1–W3 não constituem uma ordem monotônica de dificuldade. E ainda contém um
salto de intensidade em relação a D; se houver estagnação, uma fase intermediária
de Hs pode ser inserida sem alterar o PPO ou o simulador.

Nas fases D–H usam-se d=54 m, escala unitária, limite de velocidade não
vinculante e os mesmos ganhos/limites empíricos de força e momento. Desligar o
filtro não transforma esse modelo em uma reprodução exata do experimento de
Walker. O modelo dinâmico e o gerador de distúrbios não foram modificados.

## Critérios e reprodução

Cada avaliação dura 100 s e ignora os primeiros 10 s. Para um episódio passar,
ele precisa completar o horizonte, apresentar RMSE de posição ≤0,2 m e
permanecer dentro de 0,2 m em pelo menos 90% das amostras consideradas.
Aprovação exige sucesso de pelo menos 90% globalmente e em cada grupo, por
duas avaliações consecutivas em A/B e três nas demais fases. E–H usam 24
episódios, oito por caso. O critério herdado é de posição; yaw continua na
recompensa e nas métricas, mas não é um critério adicional de transição.

Os 3 milhões de passos planejados por fase não são um orçamento obrigatório
nem um teto. O perfil não mistura episódios de fases anteriores, evitando
reintroduzir filtros fortes durante a retirada. Não há retomada automática de
pesos. Use uma pasta de saída nova para cada execução.

```bash
bluerov train --config configs/training/physics_wave_curriculum_filter_bridge.yaml \
  --output artifacts/training/physics_wave_curriculum_filter_bridge

python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/physics_wave_curriculum_filter_bridge
```

O suporte de referência por fase é aplicado no reset e registrado no contexto
de avaliação. Treinamentos sem essa opção preservam o comportamento anterior.
As receitas anteriores e os dados já gerados permanecem disponíveis.

## Experimento rápido a partir do PPO do artigo

`configs/training/ppo_article_walker_finetune.yaml` carrega o par final de
`physics_wave_curriculum_yaw` (antes do FT de estrela), preservando também
otimizador e normalização. Executa somente E–H, com o oito a −5 m e os três
casos Walker. Avalia a política de origem em cada fase antes da adaptação.
Os passos registrados são interações adicionais, sem contar o treino original.
É um teste de transferência; não substitui uma futura execução do zero.

```bash
bluerov train --config configs/training/ppo_article_walker_finetune.yaml \
  --output artifacts/training/ppo_article_walker_finetune
```
