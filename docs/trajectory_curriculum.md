# Currículo de mecânica e ondas

Perfil padrão de currículo de trajetórias: configs/training/trajectory_curriculum.yaml.
Executa a receita PPO e a figura oito de 889a766. O modo anterior foi preservado
em trajectory_curriculum_legacy.yaml e trajectory_curriculum_legacy.py.

```bash
bluerov train --config configs/training/trajectory_curriculum.yaml \
  --output artifacts/training/physics_wave_curriculum
```

Para exigir GPU, acrescente `--device cuda`. Para várias seeds, use `--seeds 42 43 44`;
nesse caso cada seed recebe uma subpasta e os caminhos de avaliação devem apontar
à seed escolhida. O comando não foi iniciado automaticamente nesta alteração.

## Fases

| Fase | Mecânica | Ondas |
|---|---|---|
| A | std histórica | Sem ondas, corrente ou ruído |
| B | std histórica | Configuração da train_stable_traj |
| C1 | corrected | Hs 0.2–0.5 m; Tp 10–12 s; direção 30–60° |
| C2 | corrected | Hs 0.5–1.25 m; Tp 9–12 s; direção 0–90° |
| C3 | corrected | Hs 1.25–2.5 m; Tp 8–12 s; direção 0–180° |
| C4 | corrected | Hs 2.5–4.5 m; Tp 8–12 s; direção 0–360° |

As faixas são valores iniciais de projeto, não limites certificados ou calibrados
por um estudo de desempenho. Gamma e espalhamento também estão no YAML. Nos
quatro níveis finais, normalize_spectrum=true impõe m0=(Hs/4)^2. finite_depth e
body_frame permanecem false, wave_filter/noise_filter true; não foi ativado o
preset completo corrected de distúrbios. A mecânica, separadamente, é corrected.
A velocidade da água e as forças continuam sujeitas aos limites configurados.

A fase B tem todos os parâmetros físicos históricos, com amortecimento quadrático
[33.8,54.26875,73.37135,40,40,40]. Seu primeiro reset usa a seed de ondas 42 como
âncora; os resets seguintes mudam a realização, preservando os parâmetros. Não
promete reproduzir a sequência aleatória inteira do treinamento antigo.

Cada reset sorteia um contexto e reconstrói Dynamics integralmente, recalculando
matrizes de massa, Coriolis e coeficientes. A mudança de estágio ocorre após a
última otimização de um rollout completo e força um reset antes da próxima coleta.
Não se altera somente uma flag numa instância com matrizes antigas em cache.

Em C2–C4, 30% dos resets sorteiam níveis de mar anteriores da mesma mecânica e
mesmo modelo de distúrbio. A distribuição da validação sempre é o nível atual,
sem essa mistura. Não há troca de trajetória: quadrado e círculo ficam na avaliação.

## Tempo disponível e domínio

A referência é **3.000.000 de passos por fase**. Não é limite, mínimo obrigatório
ou critério de aprovação. O total de seis referências é 18 milhões de passos;
o treinamento real pode acabar antes ou ultrapassar esse número. n_epochs=10
continua significando dez passagens pelos dados de cada rollout, não milhões de épocas.

O PPO treina em blocos completos de aproximadamente 20 mil passos (20.480 com
rollout 2.048). Depois de otimizar o bloco, uma validação determinística independente
usa 20 episódios de 100 s, normalizador congelado e seeds separadas do treino.
A avaliação não altera RNG de treino, estatísticas de normalização ou pesos.
O treino continua sem TimeLimit de episódio, como na receita histórica; o horizonte
finito pertence somente à validação. Isso preserva o tratamento original de término.

Cada episódio de avaliação deve:

1. Completar 100 s sem falha física, distância excessiva ou valores inválidos.
2. Após os 10 s de acomodação, ter RMSE <=0.20 m.
3. Manter erro <=0.20 m em pelo menos 90% das amostras após a acomodação.

A bateria deve aprovar >=90% dos episódios e >=90% em cada grupo. São quatro
grupos equilibrados; nos níveis C1–C4, cada grupo cobre um quarto da faixa de Hs.
**Com 20 episódios, há cinco por grupo, então 90% por grupo exige 5/5.** Os limiares
e a quantidade de grupos são configuráveis; os valores iniciais são exigentes e
precisam de piloto. Não se relaxam automaticamente para forçar avanço.

Nas fases A/B, um único grupo exige >=90% dos 20 episódios e duas baterias
consecutivas aprovadas. São avaliadas a cada ~10 mil passos. Os quatro grupos
de Hs e três baterias permanecem nos níveis C1–C4. Todas as fases mantêm 10 s
de acomodação e os mesmos limites de erro. A aprovação encerra a fase e libera a
seguinte. Uma bateria reprovada zera a sequência. A última fase aprovada encerra
o treino. São preservados ator, crítico, otimizador, normalização e contador global.

O early stopping por estagnação é independente e fica desativado. Se habilitado
em plateau_early_stopping, estagnação prolongada encerra como incompleto e salva
um par _incomplete; não avança nem declara sucesso. Não confundir esse evento
com o encerramento antecipado de uma fase por domínio.

## Monitoramento e artefatos

```bash
python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/physics_wave_curriculum
```

Para acessar de outro dispositivo na mesma rede Wi-Fi, inicie o visualizador com:

```bash
python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/physics_wave_curriculum --host 0.0.0.0 --no-open
```

No outro dispositivo, abra `http://IP_DO_COMPUTADOR:8765`, usando o IPv4 do
computador que executa o visualizador. `0.0.0.0` é o endereço de escuta, não o
endereço que deve ser digitado no outro dispositivo. O padrão continua local.

O painel mostra fase, mecânica, mar amostrado, sucesso da última bateria, sequência
de baterias aprovadas, tempo da fase e passos relativos à referência de 3 milhões.
A ETA refere-se a alcançar essa referência, não a dominar o cenário. Não há uma
previsão confiável da duração total de um currículo que avança por competência.
Os gráficos de reward/erro são do treino; os resultados da validação ficam separados.

- logs/curriculum_evaluations/: métricas por episódio, contextos/seed, checkpoint
  exato avaliado, fração dentro da tolerância e RMSE com/sem o transitório.
- models/curriculum_status.yaml: contexto efetivo, sequência e histórico das fases.
- models/*_stage_N_eval_STEPS.zip e normalizador pareado: política de cada validação.
- models/*_stage_N.zip e normalizador: domínio de cada fase.
- Checkpoints periódicos a cada 50 mil passos.
- Par final: models/ppo_trajectory_curriculum.zip e
  models/vec_normalize_trajectory_curriculum.pkl.

Snapshots adicionais _curriculum_state.pkl guardam RNG e estado do agendador para
rastreabilidade. A opção --resume-run preserva pesos, otimizador, normalizador, contador de
passos, RNG e fases concluídas. Ela começa um novo episódio físico, sem retomar
o rollout interrompido exatamente. Mudanças nos critérios zeram a sequência
de validações aprovadas da fase atual. Ctrl+C salva um par
_interrupted; falha salva _failed quando possível. Não usar esses pares como prova
de conclusão do currículo.

## Avaliação final

```bash
bluerov evaluate --config configs/evaluation/trajectory_curriculum.yaml \
  --output artifacts/evaluation/physics_wave_curriculum
bluerov plot --run artifacts/evaluation/physics_wave_curriculum --suite trajectories
bluerov plot --run artifacts/evaluation/physics_wave_curriculum --suite article
```

Os caminhos apontam ao run documentado acima. A física final corrigida e a
normalização espectral são herdadas do snapshot. O benchmark compara PPO/PID/SMC
na figura oito, quadrado e círculo. O cenário nominal é Hs=3.5 m, Tp=10 s; um
benchmark não substitui testes de generalização com várias realizações reservadas.
Se mudar o diretório de treino, atualizar training_run e os dois caminhos PPO.

## Verificação da implementação

Os testes automatizados cobrem reconstrução mecânica, energia espectral,
retenção de contextos fáceis sem retorno à mecânica std, isolamento de RNG e
normalização durante avaliação, reprovação por grupo/falha física, aprendizado
contínuo nas seis fases e estagnação sem avanço. A execução reduzida de teste
usa poucos passos e limiares deliberadamente permissivos: verifica integração,
não demonstra desempenho ou aprendizagem do currículo completo.

## Retomar sem perder o treinamento

Interrompa o treino antigo com Ctrl+C e aguarde o checkpoint _interrupted terminar
de ser salvo. Depois execute com os critérios atualizados:

```bash
bluerov train --config configs/training/trajectory_curriculum.yaml \
  --resume-run artifacts/training/physics_wave_curriculum \
  --output artifacts/training/physics_wave_curriculum_v2
```

O comando recusa uma origem ainda marcada como ativa e mantém o diretório
anterior intacto. Não permite mudar a física, a arquitetura ou a seed na retomada.
Ajustes de avaliação/transição são permitidos; a política precisa ser reavaliada
sob os novos critérios. Para avaliar o resultado v2, atualize training_run e os
caminhos de modelo/normalizador no YAML de avaliação para esse novo diretório.

### Rastreamento de yaw

O perfil `configs/training/trajectory_curriculum.yaml` adiciona ao reward histórico
`-weight * abs(wrap(yaw - yaw_ref))`, com erro em radianos e `weight: 0.5`.
Configure em `environment.yaw_tracking`:

```yaml
yaw_tracking:
  weight: 0.5
  relative_observation: true
```

O canal `yaw` passa a conter o erro angular em `[-pi, pi]`; os demais canais
mantêm a convenção histórica. No reset deste perfil, os canais de rastreamento
já recebem a mesma transformação usada nos passos e na avaliação.
Sem essa seção, o comportamento histórico é preservado. Para treinar sem
penalização mantendo o feedback relativo, use `weight: 0.0`.

Esta alteração exige um novo treinamento e uma nova pasta de saída; não retome
checkpoints com observações absolutas. Não altera a física nem os distúrbios.
O painel HTML mostra média absoluta e RMSE de yaw em graus por rollout, gravados
no TensorBoard como `rollout/yaw_error_mean_deg` e `rollout/yaw_error_rmse_deg`.
Logs antigos sem esses campos não permitem reconstruir essa curva.

Ao avaliar, `environment.training_run` herda automaticamente o adaptador
`legacy_yaw_error` do treino. Sem `training_run`, selecione explicitamente esse
adaptador nos parâmetros do controlador PPO. Modelos históricos continuam usando
`legacy`. Os critérios de promoção do currículo continuam baseados em posição;
a penalização de yaw faz parte da otimização, mas não é um novo critério de promoção.
