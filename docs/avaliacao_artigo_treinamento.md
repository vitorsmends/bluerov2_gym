# Auditoria do pacote, treinamento e artigo

Data: 21/10/2026. Referência: `oceans2026/cas-dc-template.tex` no checkout atual.
Inspeção somente de leitura do código, configurações, metadados JSON dos modelos,
eventos TensorBoard e CSVs; nenhuma nova campanha de treinamento foi executada.

## Parecer

A descrição da recompensa, arquitetura PPO e currículo corresponde em grande
parte ao código e aos cinco modelos salvos. Entretanto, há erros no gerador de
distúrbios e divergências de configuração que impedem confirmar integralmente
o método físico e a reprodução dos resultados do artigo.

## Achados prioritários

1. **Hs não controla o espectro.** Em `envs/core/dynamics.py`, `_init_jonswap`
   armazena Hs, mas calcula S e A apenas a partir de alpha_js, Tp, gamma e N.
   Uma comparação executada com Hs=0 e Hs=4.5, mantendo os demais parâmetros,
   produziu diferença máxima de amplitude igual a zero. Portanto, os valores
   de altura significativa da tabela não são efetivamente impostos ao espectro.
   A primeira etapa continua sem excitação aplicada porque scale, max_current,
   noise_std e enable_wave_force a desligam. A severidade das outras etapas
   ainda muda por outros parâmetros. A correção deve normalizar a energia
   espectral para m0=(Hs/4)^2 e validar o efeito antes de novos treinos.

2. **Referenciais incompatíveis na corrente.** `_jonswap_disturbance` gera
   componentes com direções espaciais fixas, mas `step` as subtrai diretamente
   de velocidades do corpo, sem transformação pela atitude. O artigo define
   nu_c no referencial do corpo. É necessário transformar a velocidade do
   fluido do referencial espacial para o corpo antes da dinâmica e tratar
   consistentemente o referencial das cargas diretas.

3. **Campo avaliado em ponto fixo.** O gerador usa x_eval, y_eval e z_eval
   constantes, não a posição instantânea do veículo. O artigo afirma que as
   velocidades locais são avaliadas na posição do veículo. Deve-se implementar
   essa dependência ou declarar explicitamente a aproximação de ponto fixo.

4. **Episódios reais de 5000 passos.** Os manifestos registram 1000, mas a
   importação do pacote já registra BlueRov-v0 com 5000. O treinador só registra
   o ambiente se o identificador não existir. Os eventos confirmam
   rollout/ep_len_mean=5000 ao final dos cinco treinos. Isso altera retornos,
   frequência de resets e interpretação dos resultados de treinamento.

5. **Rastreabilidade do benchmark incompleta.** Os CSVs legados não identificam
   seed de treinamento, seed de distúrbio ou repetição. A configuração legada
   aponta para ppo_trajectory_final, cujo arquivo disponível tem 1.001.472
   passos, n_steps=2048, batch_size=64 e policy_kwargs vazio. É uma configuração
   diferente dos cinco modelos curriculares, não evidência de qual arquivo
   produziu cada CSV. A configuração article_trajectories aponta para um modelo
   curricular sem sufixo de seed que não foi encontrado em models. Vincular
   explicitamente cada avaliação ao modelo e VecNormalize correspondentes.

6. **Reprodução parcial das figuras.** results-moderate/trajectory_experiments
   está vazio neste checkout. Os 20 CSVs severe têm 1000 linhas cada e não
   contêm controller_cpu_time_s, exigido pelos scripts atuais de custo.
   As tabelas agregadas existem, mas não substituem esses dados brutos.

## O que foi confirmado

- Recompensa: distância à origem, norma da velocidade translacional, yaw
  envolvido em [-pi, pi], roll/pitch, pesos 1/0.1/0.5/0.5 e bônus 1 dentro de
  0.2 m; coincide com as equações do artigo.
- Cinco modelos seed 42–46 com arquitetura pi/vf [256,256], rollout 4096,
  minibatch 256, 10 épocas e learning_rate 0.0003. Manifestos e demais
  hiperparâmetros descritos são coerentes com o treinador.
- Currículo de cinco etapas com orçamento nominal de 3 milhões e parâmetros
  da tabela. O arquivo curriculum_config.yaml é outro currículo; o treinador
  atual usa a lista curriculum do training_config/defaults.
- Cada modelo contém 3.010.560 passos efetivos. A diferença para 3 milhões
  decorre do preenchimento dos rollouts em cada etapa; relatar orçamento
  nominal e total efetivo evita uma divergência documental.
- Avaliação usa observação relativa à referência conforme a descrição atual
  do artigo. Isso não demonstra equivalência com o treino: treina-se posição
  inercial, avalia-se erro girado por yaw; a correção de velocidade modifica
  apenas u e subtrai componentes que precisam de uma convenção comum.
- Reset de episódio reinicia tempo e filtros sem reamostrar fases espectrais.
  Há variação do ruído, mas episódios de uma etapa compartilham as fases;
  isso limita a diversidade de realizações vistas no treino.

## Diagnósticos dos cinco treinos

Médias dos últimos 20 registros de cada métrica no TensorBoard; retorno é o
indicador ep_rew_mean registrado pelo Monitor, não uma avaliação determinística.

| Seed | Retorno | Explained variance | Comprimento médio |
|---|---:|---:|---:|
| 42 | 2575.02 | 0.850 | 5000 |
| 43 | 1495.77 | 0.917 | 5000 |
| 44 | -444.20 | 0.895 | 5000 |
| 45 | 158.35 | 0.897 | 5000 |
| 46 | 590.90 | 0.876 | 5000 |

As diferenças entre seeds são relevantes. Explained variance alta indica ajuste
do crítico, não garante bom controle. Retorno negativo da seed 44 não prova
instabilidade, mas impede alegar sucesso equivalente sem avaliar as políticas.

RMSE mediano recalculado nas cinco trajetórias severe: PPO 0.17076 m,
NMPC 0.10203 m, PID 0.67912 m, SMC 0.27889 m. A ordenação descrita no artigo
é confirmada nesse subconjunto. O valor PPO severe de aproximadamente 0.18 m
no texto deve ser revisto: estes CSVs resultam em aproximadamente 0.17 m.

## Próximos passos recomendados

Corrigir Hs, referenciais e o limite efetivo de episódios; decidir entre campo
fixo ou dependente da posição e documentar a escolha. Essas mudanças alteram
o ambiente e exigem novos treinamentos para sustentar resultados do método
corrigido. Depois, avaliar cada seed com realizações de distúrbio pareadas
entre controladores, salvando identificação/hash do modelo e normalizador,
configuração física, seeds e tempos junto aos resultados. Só então regenerar
as figuras e atualizar as conclusões. Nenhuma dessas alterações foi aplicada
nesta auditoria, para preservar os artefatos examinados.
