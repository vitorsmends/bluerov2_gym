# Migração dos workflows

A implementação principal está em `bluerov2_gym/workflows`. Após a validação da
migração, os wrappers e diretórios antigos foram removidos. Use `bluerov` ou
`python -m bluerov2_gym.workflows`. O mapa abaixo identifica os arquivos removidos
e os módulos que preservam suas implementações.

| Caminho antigo | Implementação principal |
|---|---|
| `training/train_curriculum.py` | `workflows/training/curriculum.py` |
| `training/train.py` | `workflows/training/legacy_train.py` |
| `training/optmize.py` | `workflows/training/optimize.py` |
| `experiments/runner.py` | `workflows/evaluation/runner.py` |
| `experiments/controllers/` | `workflows/evaluation/controllers/` |
| `experiments/references.py`, `trajectories.py` | `workflows/evaluation/` |
| `scripts/evaluate_stationkeeping.py` | `workflows/evaluation/single_policy.py` |
| `training/test_with_plot.py` | `workflows/evaluation/legacy_stationkeeping.py` |
| `plots/plot_*.py` e suporte | `workflows/plotting/article/` |
| `experiments/plots/station*.py` | `workflows/plotting/stationkeeping/` |
| Demais `experiments/plots/` | `workflows/plotting/trajectories/` |
| `training/plot_training.py` | `workflows/plotting/training/curves.py` |
| `training/plot_optmize.py` | `workflows/plotting/training/optimization.py` |
| `experiments/comparison_ftc.py` | `workflows/plotting/ftc/comparison.py` |

Os caminhos da coluna direita são relativos a `bluerov2_gym/`.
`control_design/` e `scripts/validate_gazebo_dynamics.py` permanecem específicos
de projeto/auditoria; não fazem parte da avaliação comparativa.

## Configurações

`configs/` é a única árvore de configuração ativa, com caminhos relativos ao
próprio YAML. `config/` e `experiments/config/` foram removidos após a migração
dos perfis. Edite a configuração passada explicitamente ao comando atual.

O runner, a fábrica, os controladores, referências e funções numéricas dos
plots foram movidos. As funções de entrada receberam parâmetros de caminhos;
leitura de arquivos/criação de pastas dos plots passou para a execução explícita.
Os controladores agora são importados sob demanda pela fábrica.

## Saídas

Novos comandos usam `artifacts/<tipo>/<run_id>` e recusam reutilizar uma pasta de
avaliação/treino. Os módulos canônicos de comportamentos legados mantêm seus destinos próprios.
O atalho redundante `run_oceans.py` foi removido; use `bluerov train` com
`--seeds 42 43 44 45 46`.

Os CSVs não foram convertidos para um schema universal. Metadados extras ficam
em YAML, e famílias de plots diferentes continuam lendo seus formatos originais.
Os temporizadores preservam seu escopo; valores de CPU/wall time variam entre
execuções e não são usados como critério de igualdade numérica.

## Verificação da migração

Na migração foram realizados:

- 27 testes de dinâmica, ondas e contratos de workflows.
- Reexecução determinística de 100 s do agente `bluerov_ppo.zip`: CSV exatamente
  igual ao resultado anterior e PNG idêntico pixel a pixel.
- Comparação antes/depois de station keeping e trajetória, com duas repetições
  curtas de PID/SMC/PPO: dados, ações e erros idênticos, excluindo temporizadores.
- Comparação curta de NMPC antes/depois: ações, estados e erros idênticos em
  station keeping e trajetória, excluindo temporizadores.
- Os cinco plots do artigo: PNGs idênticos pixel a pixel e tabelas idênticas em
  uma fixture congelada com quatro controladores, dois cenários e duas trajetórias.
- Station keeping, custo de station keeping, path tracking legado, custo,
  tradeoff e violin: todos os PNGs comparados idênticos pixel a pixel e CSVs
  de tabelas idênticos, usando os mesmos dados de entrada.
- Gráficos espaciais/representativos de trajetórias e tabelas: igualdade exata;
  comparação FTC e curvas individuais de treino: canvases renderizados idênticos.
- Funções de cálculo e desenho dessas famílias mantêm a mesma árvore sintática;
  só suas entradas e a preparação de caminhos foram reorganizadas.
- Teste curto da CLI de treino com duas seeds, oito passos por seed, salvando
  política/normalizador/manifestações separadamente.

Essas comparações foram feitas no mesmo ambiente de fontes, NumPy e Matplotlib.
Não implicam que valores temporais ou renderização entre versões diferentes de
bibliotecas sejam idênticos. As figuras históricas não foram sobrescritas.

## Centralização posterior dos artefatos

Os dados históricos também foram movidos para `artifacts/`, com verificação
de checksums e ajuste dos YAMLs. O inventário e os novos destinos estão em
[DATA.md](../DATA.md). Após essa etapa, 30 testes passaram; a avaliação de
100 s do agente preservou exatamente o CSV e o PNG anteriores.
