# WalkerRamp: currículo de ondas sem filtragem

Perfil experimental em `configs/training/walker_ramp.yaml`, na branch
`walker_disturbances`. Não altera o simulador nem os resultados publicados.
Sua eficácia ainda precisa ser medida; a alocação conhecida não garante
aprendizado mais rápido.

## Interface de controle opcional

O controle anterior já comandava forças em newtons nos seis propulsores.
Agora `environment.action_control.mode` seleciona:

- `thrusters`: comportamento anterior, seis forças diretas, em N.
- `body_wrench`: cinco ações normalizadas em [−1,1], convertidas em
  `[X,Y,Z,K,N]` no corpo: três forças (N) e momentos de roll/yaw (Nm).

Os limites configuráveis são `[80,80,60,8,20]`. A matriz B vem da própria
configuração de dinâmica, compartilhada com a simulação. Para as cinco linhas
controladas, calcula-se `f = pinv(B_controlada) @ tau_desejado`. Quando necessário,
um fator comum reduz f até respeitar os limites físicos de cada propulsor.
Isso preserva a direção do esforço nos cinco eixos selecionados; não é uma
otimização que maximiza o aproveitamento dos atuadores. Os limites por eixo
não são simultaneamente realizáveis em todas as combinações.

A simulação continua recebendo seis forças e aplicando B para obter o esforço
real em seis eixos. Pitch permanece acoplado; não há comando independente de
pitch. A recompensa de esforço continua calculada sobre as forças efetivamente
aplicadas, e a métrica de saturação usa esses comandos.

```yaml
environment:
  action_control:
    mode: body_wrench
    wrench_limits: [80, 80, 60, 8, 20]
```

A mesma conversão é usada pelo controlador PPO na avaliação. Com
`environment.training_run`, a configuração de alocação é herdada automaticamente.
Sem essa referência, é necessário passá-la em `controllers[].params.action_control`.
O modelo com cinco ações falha explicitamente se for avaliado como seis forças.
Para comparação justa, use os mesmos parâmetros físicos de B no teste.

## Fases

Todas usam referência em oito a −5 m, dt=0,1 s e ondas sem filtro, sem ruído
gaussiano adicional. Não se mudou a geometria da trajetória nem o feedback
histórico de posição/velocidade e yaw relativo.

| Fase | Mecânica | Altura relativa dos três casos |
|---|---|---|
| A | Simplificada | Sem ondas |
| B1 | Simplificada | 5% |
| B2 | Simplificada | 10% |
| C | De validação | Sem ondas, para transferência de mecânica |
| D01–D12 | De validação | 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% |

Nas fases com ondas, o caso W1/W2/W3 é sorteado uniformemente por episódio.
Usa-se `Hs = lambda * Hs_Walker`, mantendo Tp de cada caso. A normalização
espectral está ativa: a variância da elevação varia como lambda². Os ganhos
e limites empíricos de força/momento permanecem, portanto a carga total no
veículo não necessariamente segue essa mesma proporcionalidade.

W1=(2,78 m; 7,1 s), W2=(3,47 m; 9,5 s), W3=(3,24 m; 11,1 s).
Profundidade de água 54 m; kernel Airy, rotação para o corpo, escala unitária
e limite de velocidade não vinculante. São cenários inspirados em Walker,
não uma reprodução exata do espectro medido, veículo ou controlador daquele estudo.

## Otimização e avaliação

O treinamento inicia do zero, com rede 64/64 preservada e cinco saídas. Este
perfil usa learning_rate=1e−4, ent_coef=0, target_kl=0,02, n_epochs=5 e
log_std_init=−1,5. São escolhas experimentais para reduzir atualizações grandes
e exploração excessiva, não novos valores padrão dos treinamentos anteriores.
Não há carregamento do PPO do artigo nem de seu normalizador.

A aprovação mantém os critérios de posição: RMSE ≤0,2 m e pelo menos 90% das
amostras dentro de 0,2 m, após 10 s de acomodação em episódios de 100 s.
Exige sucesso de 90% globalmente e por caso por três avaliações consecutivas.
Com ondas são 24 episódios, oito por caso, com seeds separadas do treinamento;
sem ondas são 12. Yaw permanece penalizado e registrado, mas não é um critério
adicional de transição. A aprovação de uma fase não garante bom yaw.

Os 3 milhões de passos por fase são apenas previsão de progresso. Não há
teto de passos nem avanço por orçamento. O early stopping de estagnação
encerra a execução se houver 30 avaliações sem melhoria de pelo menos
0,005 m no melhor RMSE médio, após o aquecimento configurado; não pula uma fase
não dominada. Todos os pares de checkpoints de avaliação são preservados.

```bash
bluerov train --config configs/training/walker_ramp.yaml \
  --output artifacts/training/walker_ramp

python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/walker_ramp --host 0.0.0.0 --port 51342 --no-open
```

Para uma ablação com controle direto, em uma pasta nova:

```bash
bluerov train --config configs/training/walker_ramp.yaml \
  --action-control thrusters --output artifacts/training/walker_ramp_direct
```

A ablação muda também o espaço de ações e sua escala inicial de exploração;
isso deve ser considerado ao interpretar diferenças de velocidade de aprendizado.
Não reutilize pesos entre os espaços de cinco e seis ações.

## Continuação com aprovação por janela

`configs/training/walker_ramp_window.yaml` retoma o checkpoint interrompido
de `walker_ramp` em outra pasta, preservando política, otimizador, normalização,
seed e fases físicas. A retomada começa em um novo episódio de simulação.
As fases já dominadas são preservadas no histórico.

O parâmetro `curriculum_transition.rolling_batches: 3` agrega os episódios das
três avaliações mais recentes da fase atual. São 24 episódios por caso nas
fases com ondas; 22/24 satisfazem a exigência de 90%, enquanto 21/24 não.
Cada episódio continua sujeito aos mesmos limites de RMSE e tempo dentro da
tolerância. Exige-se uma janela completa aprovada (`consecutive_batches: 1`),
evitando exigir três janelas sobrepostas como evidências independentes.
Ao retomar ou trocar de fase, a janela é preenchida com novas avaliações;
não se reutilizam retrospectivamente avaliações selecionadas para aprovação.
O log preserva o resultado do lote atual e o resumo `mastery_window`.
Os perfis anteriores continuam com a regra original por lote.

### Parada por ausência de progresso

Os perfis WalkerRamp agora usam `plateau_early_stopping.monitor: mastery_progress`.
Após preencher a janela e aquecer por cinco avaliações, a paciência é de 12
avaliações sem melhora significativa em nenhum dos três indicadores:
sucesso global, sucesso do pior caso e RMSE médio. Sucesso precisa melhorar
ao menos 2,5 pontos percentuais; RMSE, ao menos 0,005 m. Cada indicador mantém
seu próprio melhor valor significativo. Oscilar e recuperar um valor antigo
não reinicia a paciência. As janelas sobrepostas são um monitor operacional,
não amostras independentes para inferência estatística.

Com a frequência atual, 12 avaliações correspondem a aproximadamente 246 mil
interações de treino, além das avaliações. Não há orçamento rígido de passos.
O status registra os indicadores em `progress_monitor` e o motivo de parada.
A fase só avança por domínio; estagnação encerra e preserva os checkpoints.
Retomadas iniciam um novo monitor e uma nova janela de avaliação.
Perfis sem esse monitor mantêm a parada anterior por RMSE.
Este ajuste não modifica retroativamente as execuções salvas nem reinicia
automaticamente uma execução encerrada.
