# Proposta de organização: treino, avaliação e gráficos

Status: implementada; consulte [migration.md](migration.md) para o mapa e as
verificações realizadas. O texto abaixo registra a proposta original. O objetivo é reorganizar o acesso aos fluxos
existentes, preservando dinâmica, controladores, métricas, CSVs e figuras.

## Diagnóstico do checkout

| Responsabilidade | Implementação atual | Problema de organização |
|---|---|---|
| Treino com currículo | `training/train_curriculum.py`, `config/training_config.yaml` | Entrada válida, mas concorre com `train.py`, otimização e comandos espalhados nos READMEs |
| Avaliação dos quatro controladores | `experiments/runner.py`, `controllers/factory.py` | Base reutilizável já existente; seleção está no YAML, mas não há resumo prévio da campanha |
| Avaliação de um agente/setpoint | `scripts/evaluate_stationkeeping.py`, `training/test_with_plot.py` | Fluxos distintos; não assumir equivalência entre scripts |
| Figuras agregadas do artigo | `plots/run_all.py`, cinco `plot_*.py`, `plots/run_data.py` | Imports dependem do diretório do script; cenários, controladores e destinos fixos em `plots/config.py` |
| Figuras de station keeping | `experiments/plots/stationkeeping.py`, `station_cost.py` | Leitura de YAML, criação de pastas e configuração de gráficos durante import |
| Figuras de trajetórias/custos | `experiments/plots/plot_trajectory.py`, `plot.py`, `cost.py`, `cost_tradeoff.py`, `violin.py` | Rotinas independentes, caminhos e famílias de métricas diferentes |
| Curvas de treino | `training/plot_training.py` | Entrada própria; deve continuar distinguindo seeds de treino de repetições de avaliação |
| FTC | Configs FTC e `experiments/comparison_ftc.py` | Fluxo específico; comparação lê arquivos e desenha na importação |
| Projetos de controle | `control_design/` | Inclui outra implementação de dinâmica; não substituir pela dinâmica do ambiente nesta refatoração |
| Artefatos | `models*`, `training_runs`, `results*`, `oceans2026/figs`, tabelas | Não existe um diretório obrigatório por nova execução |

O README afirma que a avaliação exporta plots automaticamente, mas
`ExperimentRunner.run()` despacha os modos de avaliação e exporta CSVs; a
pipeline de figuras do artigo é executada separadamente.

`run_oceans.py` contém um comando de shell apesar da extensão `.py`. Deve ser
documentado/substituído por um comando válido, preservando seus argumentos.

## Interface proposta

Um único ponto de entrada, instalado como `bluerov`, com subcomandos separados:

```bash
# Comandos propostos, ainda não disponíveis.
bluerov train --config configs/training/curriculum.yaml
bluerov evaluate --config configs/evaluation/stationkeeping_single.yaml
bluerov evaluate --config configs/evaluation/trajectory_comparison.yaml
bluerov plot --run runs/evaluation/<run_id> --suite stationkeeping
bluerov plot --run runs/evaluation/<run_id> --suite article
bluerov plot-training --config configs/plots/training.yaml
bluerov inspect --config configs/evaluation/trajectory_comparison.yaml
```

`inspect` mostra, sem carregar políticas nem iniciar simulação: tarefa,
controladores habilitados, parâmetros, política + normalizador, mecânica e
flags de ondas efetivas, cenários, setpoints/trajetórias, seeds, repetições,
passo, horizonte e total de execuções. A avaliação imprime o mesmo plano.

Treinar não inicia avaliação implicitamente. Plotar lê resultados existentes,
sem repetir a simulação. Configs de PPO sempre especificam modelo e normalizador;
a seleção não depende do arquivo mais recente ou de busca ambígua por nome.

A configuração mantém o formato já utilizado pelo runner:

```yaml
controllers:
  - name: ppo
    enabled: true
    params:
      model_path: bluerov_ppo.zip
      vecnormalize_path: bluerov_vec_normalize.pkl
  - name: pid
    enabled: false
```

A tarefa, as condições e o perfil de figuras também ficam explícitos. O perfil
de figuras nunca decide quais controladores serão simulados. Perfis existentes
podem mostrar subconjuntos, por exemplo somente NMPC/PPO em algumas figuras 3D;
essas decisões visuais precisam continuar preservadas e documentadas.

## Organização alvo

```text
bluerov2_gym/
  envs/                         # ambiente existente
  workflows/
    cli.py                      # roteia comandos; sem fórmulas de controle
    training/                   # implementação atual do treino
    evaluation/
      runner.py                 # ExperimentRunner atual
      controllers/              # fábrica e controladores atuais
      references/               # referências e trajetórias atuais
      single_policy.py           # avaliação isolada atual
    plotting/
      manager.py                # PlotManager
      registry.py               # nomes públicos e suites
      article/                  # cinco plots agregados atuais
      stationkeeping/           # stationkeeping e station_cost
      trajectories/             # trajetória, custo e tradeoff atuais
      training/                 # diagnósticos TensorBoard atuais
      ftc/                      # comparação específica existente
    artifacts.py                # caminhos/manifestações de uma execução
configs/
  dynamics/                     # parâmetros físicos existentes
  training/                     # treino/currículo existentes
  evaluation/                   # single PPO, station keeping, trajetória, FTC
  plots/                        # entradas/saídas e seleção de suites
runs/
  training/<run_id>/
  evaluation/<run_id>/
docs/
  quickstart.md
  training.md
  evaluation.md
  plotting.md
```

Esta é a organização final, não uma movimentação em massa no primeiro passo.
Configuração de parâmetros de controladores permanece separada dos parâmetros
da planta. Não unificar controladores ou modelos só porque possuem nomes iguais.

## Classe de plots

A proposta é uma classe de orquestração, não uma classe gigante com todos os
métodos de Matplotlib. Cada figura mantém sua rotina própria:

```python
# Interface proposta.
manager = PlotManager(run_directory)
manager.generate_suite('article')
manager.generate(['tracking_error', 'rmse', 'energy'])
```

O registro associa nomes a rotinas, colunas exigidas, família de dados e arquivos
gerados. Suites registradas: `article`, `stationkeeping`, `trajectories`,
`training` e `ftc`. As duas formas de uso (CLI e Python) chamam a mesma classe.

Responsabilidades do manager: localizar dados da execução, verificar entradas,
selecionar rotinas e registrar saídas. Ele não agrega dados, recalcula métricas,
reescreve estilos nem impõe cores/legendas comuns. Seleção e agregação continuam
nas rotinas atuais até existir uma mudança científica deliberada.

Primeira versão: adaptadores para os módulos existentes, inclusive execução
isolada por processo quando necessário para preservar imports e estado global
do Matplotlib. Depois da referência visual registrada, tornar as rotinas
importáveis e receber caminhos explicitamente. Remover efeitos colaterais de
importação sem alterar o estado efetivo do desenho dentro de cada rotina.

## Resultados por execução

```text
runs/evaluation/<run_id>/
  config.requested.yaml
  config.resolved.yaml
  manifest.yaml                 # commit, dirty state, seeds, versão e modelos
  data/                         # CSVs atuais com os mesmos nomes/colunas
  figures/                      # PNG/PDF com os nomes existentes
  tables/                       # tabelas atuais
  logs/
```

O manifesto deve registrar SHA-256 de política e normalizador e distinguir seed
de treinamento da seed de distúrbio. O plano de avaliação deve registrar mecânica,
ondas, condições iniciais e motivos de encerramento antecipado.

A primeira migração não move `results-*`, modelos nem figuras do artigo. Eles
continuam legíveis via configuração de caminhos legados. Novas execuções usam
pastas exclusivas. A publicação em `oceans2026/figs` passa a ser uma etapa explícita
posterior, evitando sobrescrever figuras ao explorar um agente.

## O que precisa permanecer igual

- Ações, normalização de observação, sinal dos erros, limites e adaptador PPO.
- Inicialização de estado, sorteios, seeds, ordem das avaliações, integração e
  momento em que se mede estado/erro em relação ao `env.step`.
- Cabeçalhos, unidades, nomes e agrupamentos dos CSVs.
- Cálculo de RMSE, médias, bandas, interpolações e tratamento dos dados legados.
- Modelo de energia de cada família. A curva T200 de `plots/plot_energy.py`
  não deve substituir as estimativas registradas pelo runner, nem o inverso.
- Cores, fontes, linhas, eixos, tamanhos, DPI, legendas, painéis, nomes das figuras
  e subconjuntos de controladores de cada plot.
- Temporizadores dos controladores. Tempo de CPU não será numericamente idêntico
  entre execuções; preservar seu escopo e comparar a instrumentação.
- Valores padrão e configurações dos comandos antigos durante a transição.

Inconsistências científicas encontradas, como uma variável chamada RMSE contendo
média de erro em um script, são registradas e não corrigidas durante esta etapa.

## Migração em etapas verificáveis

1. Inventariar comandos válidos, configs, artefatos e famílias de figuras. Capturar
   resultados/figuras de referência antes de qualquer movimentação.
2. Adicionar CLI e `inspect` como fachadas. Reutilizar runner/factory e treino;
   manter scripts antigos como entradas compatíveis.
3. Introduzir `PlotManager`, registro e suites com adaptadores para os plots
   existentes. Fazer o fluxo CSV → figuras funcionar sem nova simulação.
4. Parametrizar somente caminhos e remover efeitos de importação. Migrar cada
   família de plots e verificar equivalência antes da próxima.
5. Adicionar diretório por execução para novas campanhas; manter leitura das
   pastas históricas e exportação explícita das figuras do artigo.
6. Mover módulos/configs para a estrutura alvo; deixar wrappers nos caminhos
   antigos. Reescrever README com um caminho principal e links por tarefa.

## Critérios de aceitação

- Avaliação curta e determinística antes/depois: mesmas ações, estados, erros,
  condições de parada e CSVs, exceto metadados e medições temporais variáveis.
- Plots antes/depois a partir dos mesmos CSVs congelados: tabelas numericamente
  iguais e imagens raster equivalentes no mesmo ambiente de fontes/Matplotlib.
  Comparar PDF rasterizado, não bytes de PDF que podem conter datas diferentes.
- Testar station keeping, trajetórias e CSV legado sem `repetition_id`; não
  interpretar a repetição de avaliação como seed independente de treinamento.
- Comando de plot não carrega modelo PPO nem executa o simulador. Um plot de
  PID não deve exigir CasADi ou carregar os outros controladores.
- Rodar por CLI de qualquer diretório, com caminhos resolvidos explicitamente.
- Nenhum resultado histórico precisa ser regenerado para continuar acessível.

Recomendação: iniciar pelas etapas 1–3. Elas tornam o uso compreensível e
centralizam os plots com o menor risco de modificar os resultados existentes.
