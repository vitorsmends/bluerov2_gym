# Treinador de trajetória de 889a766 no pipeline atual

Referência fornecida pelo usuário:
`889a766e4615098659a2f923782a1a56f921edd3:experiments_traj/train_ppo.py`.
A integração recupera o treinamento, o feedback, o reset, a recompensa e a
aplicação das ações. **Não restaura o simulador, as equações dinâmicas ou o
modelo de distúrbio daquele commit.** A configuração física continua vindo do
mesmo `configs/dynamics/dynamics_config.yaml` atual, sem alterações.

## Comandos

Após o setup e a ativação do ambiente:

```bash
bluerov train --config configs/training/trajectory_reference.yaml \
  --output artifacts/training/trajectory_reference
```

Para exigir GPU, acrescente `--device cuda`. Por padrão o dispositivo é auto,
como no PPO original. O perfil usa seed nula, como a chamada original ao PPO;
`--seeds 42` é uma opção explícita para controlar a aleatoriedade. Cada execução
precisa de uma pasta nova. Com múltiplas seeds, use a subpasta `seed_N` da seed
que deseja avaliar nos caminhos do YAML de avaliação.

Acompanhar o treinamento no navegador:

```bash
python -m bluerov2_gym.workflows.plotting.training.browser \
  --run artifacts/training/trajectory_reference
```

Gerar as curvas registradas:

```bash
bluerov plot-training --run artifacts/training/trajectory_reference --suite training_overview
```

Avaliar o novo agente contra PID/SMC e gerar os plots:

```bash
bluerov evaluate --config configs/evaluation/trajectory_reference.yaml \
  --output artifacts/evaluation/trajectory_reference
bluerov plot --run artifacts/evaluation/trajectory_reference --suite trajectories
bluerov plot --run artifacts/evaluation/trajectory_reference --suite article
```

O primeiro caso é `reference_889_figure_eight`, com o **mesmo gerador analítico**
usado pelo treino. Os demais casos são quadrado, oito com profundidade e círculo;
esses casos avaliam transferência para referências diferentes da referência
aprendida. O PPO usa explicitamente `observation_adapter: legacy`, e o ambiente
da avaliação é o BlueRov base: o controlador aplica a transformação uma única
vez, antes do normalizador. Não se usa TrajectoryTrackingEnv como ambiente de
avaliação para evitar aplicar a observação virtual duas vezes.

Station keeping, como tarefa de transferência desse agente de trajetória:

```bash
bluerov evaluate --config configs/evaluation/stationkeeping_trajectory_reference.yaml \
  --output artifacts/evaluation/trajectory_reference_stationkeeping
bluerov plot --run artifacts/evaluation/trajectory_reference_stationkeeping --suite stationkeeping
```

Ambas as configurações de avaliação herdam a física do snapshot do novo treino.
O cenário training_sea foi preenchido com os parâmetros da configuração física
atual. Caso altere o mar para um novo treino, ajuste também o cenário no YAML da
avaliação; o vínculo training_run herda a dinâmica, mas não substitui o cenário
explicitamente selecionado na avaliação.

## Receita preservada

- `MultiInputPolicy`, sem substituir a rede padrão por redes de 256 neurônios.
  `policy_kwargs: {}` mantém o comportamento do script, com SB3 fixado pelo setup.
- `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `gamma=0.99`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`.
- Defaults do SB3 explicitados: `n_epochs=10`, `vf_coef=0.5`, `max_grad_norm=0.5`.
- `Monitor → DummyVecEnv → VecNormalize`, normalizando observações e recompensa,
  com `clip_obs=10`. Os demais parâmetros do normalizador usam os defaults do SB3,
  como na fonte.
- Uma figura oito com raio 1, parâmetro de velocidade 0.15 e profundidade -0.5 m.
  A profundidade desce em rampa nos primeiros 10 segundos da referência.
- No reset, o instante é sorteado em `[0,50)`; a posição inicial recebe ruído
  uniforme `[-0.2,0.2]` por eixo e o yaw inicial é o da referência. As velocidades
  começam zeradas. O sorteio usa `np.random`, como o script original.
- Cada step avança primeiro o tempo da referência em 0.1 s, envia seis forças
  de propulsores em N, com clip ±40, e deixa a dinâmica atual evoluir. Depois
  calcula o erro relativo ao alvo nesse instante e entrega a observação virtual.
- O reset retorna observação **absoluta**, e os steps retornam observação virtual.
  Essa assimetria foi mantida deliberadamente, porque consta do código fornecido.
- A posição virtual é o erro rotacionado pelo yaw atual; apenas u é substituído
  pelo erro de velocidade da primeira componente. Os outros canais são mantidos,
  incluindo yaw absoluto. As misturas WORLD/BODY da receita não foram corrigidas.
- O gerador mantém `vz_ref=0` mesmo durante a rampa de profundidade, como a fonte.
- Não há currículo nem early stopping nesse perfil. São solicitados 1.000.000
  passos totais, que o PPO arredonda para 1.001.472. Isso reproduz a duração do
  script fornecido, não introduz limite de passos por fase no currículo existente.

A recompensa original de tracking é preservada, substituindo a recompensa base:

    reward = 1 - 2*||p-p_ref|| - 0.1*||v-v_ref||
               - 0.001*mean((action/40)^2)

Se o erro de posição exceder 3 m, encerra o episódio e desconta mais 10. O sucesso
é erro abaixo de 0.2 m. Não há penalidade explícita de yaw nessa recompensa.
As terminações físicas do BlueRov base também permanecem ativas.

O script instanciava a subclasse diretamente, por isso o `max_episode_steps=2000`
da chamada de registro não era aplicado. A configuração nova usa
`max_episode_steps: null` para preservar esse comportamento efetivo. Um limite
só será aplicado se for explicitamente configurado no novo perfil.

## Armazenamento e observabilidade

Modelo final: `models/ppo_trajectory_final.zip`.
Normalizador pareado: `models/vec_normalize.pkl`.
Checkpoints a cada 50 mil passos, agora com cópia do normalizador; todos ficam
na pasta da execução. O nome do prefixo inclui checkpoint para integração com
o descobridor de checkpoints do pacote. Esse salvamento adicional não muda o PPO.
O manifesto registra a referência de código, configuração efetiva, dispositivo e
a arquitetura efetivamente construída pelo SB3. O dispatcher chama o treinador
de trajetória diretamente, sem mesclar os defaults de rede/currículo de
`load_training_config` do treinador de station keeping.

Os tags custom/tracking_error_m, custom/velocity_error e custom/is_success
foram preservados. As métricas adicionais de erro euclidiano usam distância à
trajetória, substituindo a distância à origem no info usado pelos callbacks.
Os plots não devem confundir essas duas medidas.

Como os episódios não têm limite temporal, a média de reward por episódio pode
ficar sem novos registros durante bastante tempo. Este treinador também registra
`rollout/reward_mean_per_step`, a média do reward bruto por passo do rollout,
antes da normalização. O painel e o plot de overview mostram esse tag quando
presente e o rotulam como reward por passo, sem tratá-lo como retorno de episódio.
A adição de métricas e tqdm não modifica recompensa, ações ou RNG.

## O que foi observado no ambiente e dynamics antigos

A inspeção de `bluerov_env.py`, `core/rewards.py` e `core/dynamics.py` daquele
commit confirmou seis comandos diretos de força, limites ±40, dt=0.1, observações
float32 de 12 canais, velocidades BODY e as terminações físicas conhecidas.
A integração mantém essa interface usando a classe BlueRov atual.

A dinâmica antiga e seus distúrbios não foram portados. A inspeção encontrou:
mar padrão Hs=1/Tp=6; JONSWAP horizontal com alpha_js=0.076*Hs²; ruído vertical
zerado; ausência de tau_wave separado. O modelo atual tem outras opções de
espectro/profundidade/direção, componentes verticais, força de onda separada,
outros coeficientes de damping e correções mecânicas. O preset std atual não é
uma reprodução integral de 889a766.

Essas diferenças foram identificadas para explicitar o escopo, não para revertê-las.
O perfil **não altera nem Hs/Tp da configuração atual**, nem seus coeficientes,
modelos ou flags. Replica a receita do agente sobre o simulador atual, conforme
solicitado. Fidelidade ao treinador não é uma promessa de reproduzir numericamente
os pesos ou as curvas antigas no novo simulador.

## Revisão realizada

Integração revisada estaticamente contra o código versionado. Não foram
executados testes, treinamento, simulações ou avaliação de modelos nesta tarefa,
conforme a restrição do usuário.
