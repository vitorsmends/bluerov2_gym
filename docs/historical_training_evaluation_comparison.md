# Comparação com 26857c4456e11c6173b04678b21b6806d77b5b4c

**Referência de restauração corrigida pelo usuário:** seguir o treinador curricular do commit `0436ce3482e95105cb2376681b2bd6cbec72d99c`, com PPO 4096/256 e redes pi/vf [256,256]. Os ensaios com ZIPs abaixo são diagnósticos históricos e não definem a receita de treinamento a restaurar. A sugestão de substituir o currículo por cenário fixo foi retirada. Consulte [treinamento](training.md).

A avaliação histórica boa foi reproduzida: o `ppo_trajectory_final.zip` contido nesse commit, com o seu `vec_normalize.pkl`, atingiu RMSE de posição de 0,168 m no círculo severe/seed 42. O diagnóstico anterior identificou limitações reais da interface, mas não demonstrava que elas causaram a regressão. A comparação histórica comprova que essas limitações já existiam quando esse agente funcionava bem.

**Limite de identificação:** o teste cruzado deste relatório usa `ppo_trajectory_final.zip`, não `bluerov_ppo.zip`. Ambos existem no commit, mas são agentes distintos. O segundo registra `policy_kwargs: {net_arch: [256, 256]}`, como o antigo treinador de station keeping em `examples/train.py`. Foi encontrado também um treinador de trajetória anterior ao commit (removido em `4e9967b`), com parâmetros compatíveis com `ppo_trajectory_final.zip`. Essa compatibilidade e a identidade dos blobs entre commits não provam qual tarefa gerou o ZIP. Não atribuir a regressão do agente de station keeping à ausência de treino de trajetória sem identificar o modelo efetivamente usado pelo usuário e sua procedência.

Nenhum controlador, configuração de usuário ou modelo foi alterado para esta auditoria. O código antigo e seus modelos foram extraídos com `git show` para diretórios isolados. Não houve checkout, reset ou retreinamento.

## Arquitetura e comportamento

| Componente | Commit histórico | Atual | Resultado da comparação |
|---|---|---|---|
| Entrada de treino | `run_oceans.py` chama `training/train_curriculum.py --config config/training_config.yaml --seeds 42 43 44 45 46` | CLI `bluerov train`, módulo `workflows/training/curriculum.py` | Encapsulamento com diretório de execução, logs e snapshots |
| Configuração de treino | `config/training_config.yaml` | `configs/training/curriculum.yaml` e snapshot `my_train` | PPO, VecNormalize, cinco estágios, mar inicial e defaults JONSWAP iguais semanticamente |
| Rede especificada no currículo | pi/vf `[256,256]` | pi/vf `[256,256]` | Igual no código/config do currículo |
| Observação de treino | Estado absoluto, velocidade BODY, orientação absoluta | Mesmos canais e referenciais | Igual |
| Recompensa e reset | Estabilização na origem, yaw zero; reset na origem | Mesmas regras | Arquivo de recompensa sem alteração; diferenças no env são metadados da dinâmica |
| Normalização | `Monitor → DummyVecEnv → VecNormalize` | Mesmo encadeamento | Mesma parametrização e atualização; inferência congelada |
| Currículo | Cinco estágios por orçamento de passos | Mesma sequência e orçamentos | Funções de atualização do mar e construção dos parâmetros iguais por AST |
| Entrada de avaliação | `experiments/runner.py` e YAML | CLI e `workflows/evaluation/runner.py` | Agora salva configurações e término de episódios; recebe configuração do ambiente/da dinâmica explicitamente |
| Observação virtual PPO | Erro de posição no yaw atual, só u descontado, yaw absoluto | Mesma transformação | AST idêntica, removendo docstring; não foi introduzida na refatoração |
| Inferência PPO | Normaliza observação virtual; `predict(deterministic=True)`; clip ±40 N | Mesmo cálculo | Alterações de caminhos de busca dos arquivos |
| Trajetórias | `experiments/trajectories.py` | `workflows/evaluation/trajectories.py` | Conteúdo idêntico |
| PID e SMC | `experiments/controllers/` | `workflows/evaluation/controllers/` | Conteúdo idêntico |
| Integração e tempo | Euler, dt=0,1 s, amostra pré-passo | Mesma integração e convenção | Mudanças estão nas equações/parametrização da física |

Não confundir `config/curriculum_config.yaml`, que descrevia outro formato de currículo legado, com `config/training_config.yaml`, realmente chamado por `run_oceans.py` naquele commit. Comparar o arquivo errado aparentaria mudanças de currículo que não ocorreram nesse caminho.

Existe uma inconsistência antiga do limite de episódio: o YAML informa 1.000 passos, mas o pacote registra `BlueRov-v0` com 5.000; o treinamento só registra se o ID ainda não existir. O registro de 5.000 e a condição já existiam no commit histórico. Isso merece correção explícita futura, mas não caracteriza uma regressão nova de refatoração.

## Mudanças efetivas na física e no carregamento

As alterações principais entraram no commit `83f9850`, anterior à migração de arquitetura `e3f1d49`:

- Equações mecânicas corrigidas: restauração com Z positivo para cima, matriz de Coriolis e transferência da inércia ao ponto de origem. No estado horizontal, `g_eta_z` passou de +4,524 N para -4,524 N; a equação subtrai esse vetor. A inércia total diagonal de roll/pitch passou de 5,3739/8,0620 para 5,5793/8,2674 kg m² com os parâmetros auditados.
- A avaliação histórica chamava `gym.make` sem `dynamics_config`. Portanto, não usava o YAML da dinâmica do treinamento. Seu damping quadrático interno era `[33.8,54.26875,73.37135,40,40,40]`; o treino com YAML usava `[17.77,125.9,72.36,0.1246,0.1246,0.1857]`. Hoje treino e avaliação de `my_train` usam o YAML resolvido comum.
- As flags de correção do distúrbio existem hoje, mas estão em `std` nessa execução: normalização espectral, profundidade finita corrigida e rotação BODY desabilitadas, filtros habilitados. Logo, não se deve atribuir esta regressão à ativação dessas correções JONSWAP.
- O caminho de busca dos modelos mudou e a configuração de avaliação passou a apontar ao modelo recém-treinado. A comparação anterior misturava a discussão do código curricular com um modelo histórico diferente, apenas pelo nome usado na avaliação.

Em um teste de 100 passos com ações aleatórias pareadas, `Dynamics` antigo e o atual com `mechanics_model: std`/distúrbio std produziram diferença máxima zero em todos os estados e diagnósticos retornados. Isso valida a reprodução do ramo legado nesse teste; não é prova de equivalência universal.

## O modelo histórico avaliado não corresponde ao treino curricular atual

A evidência vem dos próprios ZIPs, não de seus nomes:

| Metadado salvo | `ppo_trajectory_final.zip` em 26857c4 | `my_train` atual |
|---|---:|---:|
| Passos | 1.001.472 | 3.010.560 |
| Seed | Não registrada (`None`) | 42 |
| n_steps | 2.048 | 4.096 |
| batch_size | 64 | 256 |
| policy_kwargs | `{}`; rede padrão | pi/vf `[256,256]` |
| SB3 no arquivo | 2.8.0 | 2.8.0 |

A configuração curricular antiga já especificava 4.096/256/[256,256], assim como hoje. Portanto, o arquivo de modelo que sustentava a boa avaliação não foi produzido exatamente com aquela configuração, ou foi produzido por outro caminho/versão. Não foi localizado no commit um script que salve `ppo_trajectory_final.zip`; não há manifesto associado que permita afirmar seu cenário, recompensa e currículo de origem. O manifesto de `models/bluerov_ppo_curriculum_seed_42_config.yaml` pertence a outro nome de modelo e não comprova a procedência desse ZIP.

Os normalizadores também refletem distribuições de treino diferentes: desvio de v 0,106 versus 0,032 m/s, yaw 1,762 versus 0,637 rad e z 0,587 versus 4,854 m (histórico versus novo). São evidências de exposições diferentes, não prova isolada da causa ou de normalizador trocado.

## Teste cruzado controlado

Círculo padrão, severe, seed 42, estado inicial zero, 1.000 passos de 0,1 s. Cada política manteve seu próprio normalizador. Mesma referência e adaptador de produção, sem as correções experimentais de velocidade. O ambiente externo atual foi mantido: a diferença do código de ambiente para o antigo limita-se aos metadados; a classe de dinâmica histórica foi carregada diretamente do Git.

| Física/parametrização | PPO histórico | PPO novo |
|---|---:|---:|
| Avaliação histórica: equações antigas + parâmetros internos | **0,168 m** | 1,120 m |
| Treino histórico: equações antigas + YAML antigo | **0,165 m** | 1,173 m |
| Treino/avaliação atuais: equações corrigidas + YAML atual | **0,365 m** | 1,148 m |

O resultado novo/atual reproduz exatamente o círculo seed 42 da avaliação existente: 1,148002975585072 m. Todos os seis testes completaram os 1.000 passos.

Conclusões delimitadas por esses testes:

1. O resultado bom histórico é reproduzível; a interface legada conseguia funcionar bem com aquela política.
2. Mudar apenas a física degrada essa política de 0,168 para 0,365 m nesse círculo. Logo, há impacto real das correções físicas.
3. O modelo novo continua ruim mesmo nas duas configurações físicas antigas. Portanto, reverter apenas a física ou reorganização de pastas não recupera o comportamento antigo dessa política.
4. A diferença entre as políticas, incluindo suas arquiteturas, históricos de treinamento e normalizadores, é determinante nesse ensaio. Os testes não isolam qual hiperparâmetro/estágio causou a diferença; isso exige treino controlado.
5. As limitações de observação/reforço do diagnóstico anterior continuam válidas, mas são preexistentes. Não justificam afirmar que PPO precisa necessariamente perder, ou que tracking só pode funcionar após retreinamento: o agente histórico é um contraexemplo medido.

Esta é uma comparação causal local de uma trajetória/seed, não uma campanha estatística de todos os controladores. Não permite generalizar os valores a outras trajetórias ou afirmar superioridade geral da política histórica.

## Artefatos

Diretório: `artifacts/diagnostics/ppo_tracking/historical_26857c4/`.

- `figures/historical_comparison.png`: RMSE cruzado e trajetórias XY.
- `summary.json` e seis CSVs: resultados e amostras.
- `protocol.yaml`: commit, hashes de modelos/fontes e protocolo.
- `inputs/`: fontes e modelo/normalizador históricos extraídos do Git.
- `comparison.log`: resultados e verificação semântica das configurações.
- `reproduce.py`: reprodução a partir da raiz do workspace; usa configuração atual e sobrescreve somente seus próprios arquivos de diagnóstico.
- `equivalence_check.py`: verificação do ramo std e ASTs.

## Próximo experimento recomendado

Preservar a nova física e separar duas linhas explícitas: reprodução histórica com os pesos antigos e novo treino controlado. Antes de mais mudanças de observação/recompensa, comparar novo treino com hiperparâmetros históricos salvos (2.048/64/rede padrão), mantendo dinâmica, seed e cenário documentados; registrar métricas de trajetória durante o treino para selecionar checkpoints pela tarefa realmente desejada. Isso testa a hipótese da receita de treino sem misturar novas correções do adaptador. Não há base para restaurar toda a arquitetura antiga: a parte computacional equivalente já reproduziu a boa política histórica.
