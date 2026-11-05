# Auditoria estática da validação desde 0436ce3

Referência: `0436ce3482e95105cb2376681b2bd6cbec72d99c`.
Entrada confirmada pelo usuário: `experiments/run_experiment.py`, comparação
entre controladores. Esta auditoria usa somente fontes, YAMLs versionados e
diffs do Git. Não leu resultados, logs, snapshots de treinos ou arquivos de
modelo/normalizador, nem executou testes, inferências ou simulações.

## Regressão identificada na mudança mais recente

O commit `de23e72` introduziu `goal_frame` como padrão do PPO e nos YAMLs
vinculados a treinos. Isso alterou efetivamente a função aplicada à entrada
antes da rede; não foi uma reorganização de código.

Em 0436ce3:

    e_pos = Rz(yaw_atual)^T (p - p_ref)
    observação de yaw = yaw_atual
    observação de u = u_BODY - v_ref_WORLD.x
    demais canais = estado original

Em goal_frame:

    e_pos = Rz(yaw_ref)^T (p - p_ref)
    observação de yaw = wrap(yaw_atual - yaw_ref)
    observação de velocidade = v_BODY - R_WB^T v_ref_WORLD

O adaptador antigo tem inconsistências de referenciais, mas já era o adaptador
com o qual o usuário validava a política. Corrigir expressões geometricamente
não garante compatibilidade comportamental com uma política congelada. Em
particular, descontar yaw_ref introduz uma meta de orientação que a validação
antiga não impunha: ela mantinha o yaw absoluto no canal esperado pela política.
No círculo inicial, cujo yaw_ref é aproximadamente 90 graus, o yaw observado
muda de zero para aproximadamente -90 graus antes da primeira ação. A posição
virtual também passa a usar outros eixos. Isso é uma alteração substancial.

O relato do usuário de piora após essa mudança é consistente com a mudança
de entradas, mas esta auditoria não mede sua contribuição numérica. Foi um erro
tratar goal_frame como uma correção retrocompatível já validada.

Correção aplicada agora: `legacy` volta a ser o padrão no helper, controlador,
avaliação single-policy, CLI e YAMLs. Sua expressão é a de 0436ce3. `goal_frame`
continua disponível somente mediante seleção explícita, como experimento.
A física corrigida, os pesos e o normalizador não foram revertidos/modificados.

## Comparação do caminho que produz a ação

    observação bruta → adaptador → VecNormalize.normalize_obs
    → model.predict(deterministic=True) → float32/reshape → clip [-40,40]
    → env.step

| Componente | 0436ce3 | Atual, após restaurar legacy |
|---|---|---|
| Seleção da entrada | `experiments/run_experiment.py` | `bluerov evaluate` |
| Adaptador PPO | BODY posição, u-only, yaw absoluto | Mesmo cálculo, com seleção explícita legacy |
| Normalização | `VecNormalize.load`; training=False; norm_reward=False; normalize_obs uma vez | Mesma lógica |
| Inferência | predict determinístico | Mesma lógica |
| Comandos | float32, vetor de 6, clip ±40 | Mesma lógica |
| Observações físicas | 12 canais, posições WORLD, velocidades BODY | Mesmos canais; env ganhou metadados em info |
| Recompensa | Station keeping na origem | Mesmo código da recompensa |
| Trajetórias | `experiments/trajectories.py` | Conteúdo idêntico no módulo movido |
| PID/SMC | Implementações do commit | Conteúdo idêntico nos módulos movidos |
| Configuração de mar | `experiments/ocean_environment.yaml` | Conteúdo idêntico em `configs/evaluation/ocean.yaml` |

Não foi encontrada alteração de escala de ação, rotação adicional dos comandos,
segunda normalização das observações ou substituição do cálculo de PID/SMC.
Isso descreve as fontes comparadas, não verifica os arquivos executados no outro PC.

## Diferenças de protocolo e configuração

### Modelos selecionados

O YAML padrão de 0436ce3 carregava `ppo_trajectory_final` e `vec_normalize.pkl`.
O treinador curricular daquele commit salvava `bluerov_ppo_curriculum` e
`bluerov_vec_normalize_curriculum.pkl`. Ele não encaminhava automaticamente o
resultado do treino para a configuração de avaliação.

Os YAMLs atuais de com_curriculo apontam aos nomes curriculares dentro de
`artifacts/training/com_curriculo/models`.

Nomes/caminhos diferentes não provam pesos diferentes, porque pode ter havido
cópia/renomeação manual. Contudo, o código não prova identidade dos modelos.
Não se deve deduzir a tarefa de treino pelo nome `ppo_trajectory_final`.

### Cenário selecionado

O YAML padrão antigo selecionava `moderate`. Os YAMLs de com_curriculo criados
neste fluxo selecionam `severe`, salvo edição explícita do usuário no outro PC.
Executar aqueles comandos sem mudar essa seleção não é avaliar Calm Sea.
A hipótese de regressão deve separar seleção do cenário e alteração do simulador.

O arquivo ocean_environment/ocean.yaml em si não mudou. As diferenças entre os
parâmetros Calm do treino e Calm da validação já existiam no commit de referência;
não podem ser apresentadas como uma regressão introduzida pela migração.

### Repetições e seeds

A função antiga de trajetórias não usava `experiment.repetitions`, mesmo com
valor 5 no YAML: fazia uma execução por controlador/trajetória. Os parâmetros
JONSWAP carregavam seed 42. A campanha passou a aplicar repetições e seeds
42+repetition_id em `a198880`. Os YAMLs atuais de com_curriculo usam 3 repetições.
Isso muda a amostra do experimento, não a fórmula de ação da primeira repetição.

O caminho de station keeping já tinha repetições e offset de posição
[0.20,-0.15,0.08] e yaw +10 graus no commit original; esses offsets não foram
introduzidos na refatoração.

### Ambiente e física

Em 0436ce3, o runner construía `gym.make` sem dynamics_config e usava os
parâmetros internos de Dynamics. O YAML de dinâmica era carregado no ambiente
auxiliar usado para anexar o VecNormalize, não no ambiente que simulava o ROV.
Esse ambiente auxiliar não executa os passos da campanha PPO.

`83f9850` introduziu o envio da configuração de dinâmica ao ambiente de avaliação.
O fluxo atual com training_run herda a física do snapshot do treino, o que é
compatível com o pedido de treinar/avaliar na física nova, mas difere da montagem
original. A existência desse vínculo no código não confirma qual YAML foi usado
no outro PC.

Em `a198880`, o damping quadrático do YAML foi alterado de
[33.8,54.26875,73.37135,40,40,40] para
[17.77,125.9,72.36,0.1246,0.1246,0.1857]. Os parâmetros internos antigos ficaram
inalterados até o ramo mecânico corrigido. Isso criou, naquele período, uma
diferença entre treino com YAML e avaliação com defaults internos.

`83f9850` também mudou as equações de restauração, Coriolis e transferência de
inércia, e adicionou flags de distúrbio. O preset std preserva as fórmulas antigas
de JONSWAP, mas é independente do modo mecânico. Não é possível classificar o
impacto dinâmico dessas mudanças como pequeno ou grande só pelo diff. Esta
auditoria não reverteu a física nova nem atribui a ela a piora observada.

## Mudanças nos gráficos

O gráfico espacial atual seleciona repetition_id=0 por padrão, em vez de
concatenar trajetórias de diferentes repetições. Os gráficos da suíte article
fazem RMSE por execução e agregação de erros entre execuções; isso é distinto
de um gráfico de uma única trajetória/repetição. A coluna tracking_error_m
continua sendo a norma do erro de posição WORLD, calculada antes do passo.
A expansão de colunas do CSV e essas agregações não alteram as ações do PPO.
Nesta leitura não foi identificada troca de colunas de estado/referência no
CSV capaz de explicar por si só a perda de tracking.

## O que esta revisão conclui e o que não conclui

A regressão recente da interface está identificada e sua ativação padrão foi
retirada. A entrada da política voltou ao cálculo do commit de referência.
As diferenças do protocolo, do carregamento da física e dos caminhos de modelo
estão mapeadas. A migração e3f1d49 preservou o núcleo da inferência PPO; não foi
ela que introduziu goal_frame.

Isso não comprova que toda a piora anterior a goal_frame foi resolvida. Sem
executar a mesma política/normalizador em condições controladas, não se pode
atribuir causalmente a diferença numérica a um único item. Essa limitação é
respeitada aqui conforme a restrição de análise apenas do código. A arquitetura
ser igual não demonstra identidade de pesos, normalização, protocolo ou entradas.

A comparação histórica deve começar com legacy, um cenário explicitamente
selecionado, uma única repetição seed 42, mesma referência/estado inicial e a
física nova documentada. Assim, uma futura comparação autorizada separa o efeito
da adaptação, da campanha e da física. Não é necessário retreinar para desfazer
a mudança de adaptador.
