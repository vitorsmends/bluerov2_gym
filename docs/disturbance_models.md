# Modelos de ondas e ablações

O modelo padrão continua sendo `std`. Omitir `disturbance_model` preserva o
comportamento anterior das ondas. A mecânica é selecionada separadamente por
`mechanics_model` (padrão `corrected`; veja [mechanics.md](mechanics.md)).
`corrected` seleciona cinco mudanças no modelo
de ondas, sem alterar os parâmetros mecânicos do ROV.

No `configs/dynamics/dynamics_config.yaml`:

```yaml
disturbance_model:
  preset: corrected
```

Cada flag explícita prevalece sobre o preset:

| Flag | `std` | `corrected` | Efeito quando `true` |
|---|---|---|---|
| `normalize_spectrum` | false | true | Normaliza o espectro discreto para `sum(S*dw) = (Hs/4)^2` |
| `finite_depth` | false | true | Usa as razões de Airy com denominador `sinh(kd)` |
| `body_frame` | false | true | Transforma velocidade global para BODY antes de calcular velocidade relativa e força empírica |
| `wave_filter` | true | false | Aplica o filtro `alpha_wave` à soma de velocidades e ruído |
| `noise_filter` | true | false | Aplica o filtro `alpha_noise` ao ruído gaussiano |

Por exemplo, para testar apenas a normalização, partindo do comportamento antigo:

```yaml
disturbance_model:
  preset: std
  normalize_spectrum: true
```

Para testar o conjunto corrigido mantendo o filtro do ruído:

```yaml
disturbance_model:
  preset: corrected
  noise_filter: true
```

## Treino por currículo

Em uma cópia de `configs/training/curriculum.yaml`, sobrescreva os parâmetros do
arquivo de dinâmica pelo bloco inline:

```yaml
environment:
  # Preserve as demais entradas environment da configuração original.
  dynamics_config_file: ../dynamics/dynamics_config.yaml
  dynamics_config:
    disturbance_model:
      preset: std
      normalize_spectrum: true
```

Execute `bluerov train --config configs/training/minha_ablacao.yaml`.
Dê nomes/diretórios distintos a `paths.model_name`, `paths.vecnormalize_name`,
`paths.model_dir` e `training.tb_log_name` para separar as execuções. Use o mesmo
seed, currículo e orçamento de treino entre comparações. As flags não são
parâmetros JONSWAP: persistem em `set_jonswap_params` e nos resets do currículo.

O manifesto `*_config.yaml` contém `resolved_dynamics_config`, incluindo o
conteúdo do arquivo externo e todas as flags efetivas. O preset e as flags também
aparecem no console e em `info['disturbance_model']` no reset e no step do Gym.

## Experimentos

Em `configs/evaluation/trajectory_comparison.yaml`, use o bloco `environment.dynamics_config`
para configurar a planta compartilhada pelos controladores:

```yaml
environment:
  dynamics_config:
    disturbance_model:
      preset: corrected
```

Use um diretório de saída por ablação. O runner salva `experiment_config.yaml`
com a configuração da campanha e `resolved_disturbance_model` junto aos CSVs.
Os parâmetros de modelo de predição de um controlador continuam separados dos
parâmetros da planta.

## Interpretação física

- A normalização é exata para a energia do espectro discretizado retido. Usa a
  forma JONSWAP e ignora `alpha_js` nessa opção. Não é uma reprodução literal da
  correção aproximada de energia por gamma do MSS. A altura significativa
  estimada em uma realização temporal finita pode diferir do alvo espectral.
- `Hs=0` elimina as componentes de onda com normalização habilitada; ruído
  ambiental só desaparece também se `noise_std=0`.
- Desabilitar `noise_filter` mantém ruído branco de desvio `noise_std`. Para
  estudar só a retirada do filtro das ondas, mantenha `noise_filter: true`.
- `scale`, `max_current`, `enable_wave_force`, ganhos e limites de força continuam
  ativos em ambos os presets. Para estudar velocidade orbital sem escala use
  `scale: 1.0`; escolha `max_current` suficientemente alto para evitar clipping.
  `Hs` controla a elevação, mas escala, clipping e filtros alteram a velocidade
  efetivamente aplicada ao veículo.
- O campo continua sendo amostrado em `x_eval`, `y_eval`, `z_eval`. As convenções
  verticais existentes são preservadas: a flag BODY só faz a rotação dos eixos.
- `nu_wave_raw` permanece em eixos globais, depois de escala/clipping;
  `nu_current` passa a estar em BODY quando `body_frame` está ativo. Os ganhos
  e o ponto de aplicação da força empírica são interpretados em BODY.
- O preset de ondas não acrescenta excitação por RAOs nem termos de aceleração
  do fluido. Massa rígida, Coriolis e hidrostática são controlados separadamente
  por `mechanics_model`. Veja a auditoria histórica em
  [gazebo_dynamics_validation.md](gazebo_dynamics_validation.md).

As expressões antigas são preservadas para reprodução de resultados. O preset
`corrected` reúne somente as correções de ondas listadas nesta tabela.

## Verificação

```bash
python -m unittest discover -s tests -v
```

Os testes incluem uma trajetória de referência com `mechanics_model: std`
obtida antes das alterações,
energia espectral, condições na superfície e no fundo, rotação com roll/pitch/yaw,
independência dos filtros, persistência no currículo e as 32 combinações das flags.
