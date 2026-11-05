# Convenção de observações do PPO: treino e avaliação

Esta correção afeta somente a preparação das observações antes da normalização
na avaliação. O treinamento curricular, a recompensa, a física e os critérios
de avanço permanecem com seu comportamento atual. Não foram lidos resultados,
logs, snapshots de execuções ou pesos, nem executados testes ou simulações nesta alteração.

## Entrada do treinamento

O ambiente fornece `x,y,z` no mundo W; `roll,pitch,yaw` são os ângulos do corpo
em relação a W; `u,v,w` e `p,q,r` são velocidades linear e angular no corpo B.
A rotação da cinemática é R_WB = Rz(yaw) Ry(pitch) Rx(roll), de B para W:

    p_dot_W = R_WB v_B

A política aprende a estabilizar posição zero, orientação nivelada com yaw zero
e velocidades zero. A entrada é o estado absoluto, e o VecNormalize aplica
as médias/variâncias correspondentes a esses canais. Não há rotação da posição
para os eixos do corpo no treinamento.

## Transformação antiga: legacy

O adaptador antigo recebia uma referência de posição p_d e velocidade v_dW e fazia:

    posição enviada = Rz(yaw_atual)^T (p_W - p_d)
    u enviado = u_B - v_dW.x
    v,w enviados = v_B.y, v_B.z
    yaw enviado = yaw_atual

São três inconsistências: a posição vira BODY enquanto o yaw continua em W;
a subtração de u mistura BODY e WORLD; e a velocidade de referência não é
descontada nos outros dois eixos. O yaw desejado também não é usado.

Exemplo: alvo na origem, ROV em [1,0,0], yaw de 90 graus e velocidades zero.
O treino apresenta posição [1,0,0] com yaw de 90 graus. O adaptador antigo
apresenta [0,-1,0] com o mesmo yaw: mudou a interpretação da posição sem mudar
a interpretação da orientação.

Para velocidade, com yaw de 90 graus, ROV parado e velocidade desejada WORLD
[1,0,0], a referência equivale a [0,-1,0] em BODY. O erro correto é [0,1,0],
enquanto o adaptador antigo entrega [-1,0,0].

## Transformação corrigida: goal_frame

Define-se um mundo virtual G com origem no alvo e eixos orientados pelo yaw
do alvo, psi_d. Para referências niveladas (roll_d=pitch_d=0):

    R_WG = Rz(psi_d)
    posição virtual_G = R_WG^T (p_W - p_d)
    roll_virtual = roll_atual
    pitch_virtual = pitch_atual
    yaw_virtual = wrap(yaw_atual - psi_d)
    velocidade virtual_B = v_B - R_WB^T v_dW
    velocidade angular virtual_B = omega_B - omega_dB

A posição está em um mundo virtual, não no corpo instantâneo. A orientação
é medida em relação a esse mesmo mundo virtual. Os canais de velocidade
permanecem no corpo real, como no treinamento. Para psi_d=0, a posição é
simplesmente erro no WORLD original. O código rejeita referências com roll/pitch
desejados não nulos, pois a regra acima não implementa a composição geral de
orientações em SO(3).

Com referência completa zero, a transformação se reduz à observação original
do ambiente, para qualquer yaw do veículo. Para um alvo fixo apenas deslocado,
com yaw desejado zero, muda somente a origem da posição. Para um alvo fixo com
yaw desejado diferente, posição e orientação são transformadas consistentemente
para os eixos fixos desse alvo. A convenção geométrica fica coerente; isso não
prova invariância do modelo de ondas ou generalização da política a todo alvo.

O vetor de referência mantém a convenção já usada no pacote: posição/velocidade
linear desejadas no WORLD, e velocidades angulares desejadas em BODY.
`StationKeepingReference` e `BaseTrajectory` atualmente produzem p/q/r desejados
iguais a zero. Nesses casos o adaptador preserva as velocidades angulares medidas.
A normalização é aplicada uma única vez, depois da transformação, usando o
normalizador pareado ao modelo, com atualização desativada.

## Limites no caso de trajetória móvel

A transformação corrigida não torna tracking equivalente a estabilização.
Quando psi_d varia, a derivada da posição virtual inclui um termo adicional:

    e_dot_G = R_WG^T (R_WB v_B - v_dW) - omega_GW_G × e_G

Também surgem efeitos de aceleração da referência nas velocidades relativas.
A política treinada para repouso não recebe explicitamente esses termos.
Os geradores atuais ainda fornecem taxas angulares desejadas zero mesmo quando
o yaw desejado varia. A presente alteração não modifica os geradores, não cria
feedforward de aceleração e não garante melhora em trajetórias móveis.

## Seleção e rastreabilidade

Na entrada PPO do YAML de avaliação:

```yaml
- name: ppo
  enabled: true
  params:
    observation_adapter: goal_frame
    model_path: ...
    vecnormalize_path: ...
```

`goal_frame` é o padrão do controlador e foi explicitado nas configurações
com `environment.training_run`, incluindo `trajectory_com_curriculo.yaml` e
`stationkeeping_com_curriculo.yaml`. A seleção aparece no inspect, no YAML
resolvido da execução e no log de criação do PPO.

Para reproduzir o comportamento anterior, selecione `observation_adapter: legacy`.
As configurações genéricas antigas sem snapshot de treino (`article.yaml`,
`stationkeeping_comparison.yaml`, `ftc.yaml`) mantêm `legacy` explicitamente,
para não alterar silenciosamente essas campanhas. Se forem usadas com um
agente de station keeping treinado na convenção do ambiente, selecione
`goal_frame` conscientemente.

A avaliação de um único PPO também aceita `--observation-adapter goal_frame`
ou `legacy`; no YAML de `stationkeeping_single` a opção fica na raiz.
O treino e a reavaliação de checkpoints na origem continuam usando observações
originais, sem adaptador, como antes. PID/SMC/NMPC e as métricas existentes não
foram modificados. A questão distinta do erro de velocidade legado da comparação
WORLD/BODY nas métricas permanece fora desta mudança.
