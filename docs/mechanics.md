# Mecânica do ROV

A dinâmica usa mundo com Z positivo para cima, velocidades em BODY e um corpo
rígido equivalente de massa `base_mass + n_thrusters * prop_mass`. A geometria
dos rotores/câmera não é composta automaticamente; a inércia configurada descreve
o corpo equivalente no seu centro de massa, expressa em eixos BODY.

```yaml
mechanics_model: corrected  # padrão, inclusive quando omitido

disturbance_model:
  preset: std              # escolha das ondas, independente da mecânica
```

O modo `corrected`:

- transporta `vehicle.inertia` do centro de massa para a origem BODY por
  `I_origin = I_g - m*S(r_g)^2`;
- usa o operador espacial de força com blocos
  `[[S(omega), 0], [S(v), S(omega)]]`, preservando a potência nula de Coriolis;
- calcula peso/empuxo em BODY a partir de forças verticais no mundo e momentos
  `r_g x F_weight + r_b x F_buoyancy`, incluindo centros fora do eixo Z;
- usa, quando não há override, o mesmo amortecimento quadrático que
  `config/dynamics_config.yaml` e os coeficientes ativos da referência Gazebo.

A força hidrostática é armazenada com sinal oposto em `restoring`, pois a equação
subtrai esse vetor. Um veículo alinhado com empuxo maior que o peso acelera para
Z positivo. Os momentos de roll/pitch são restauradores para os centros padrão.

Para repetir o modelo anterior integralmente, selecione ambos os modos:

```yaml
mechanics_model: std
disturbance_model:
  preset: std
```

Overrides explícitos de coeficientes no YAML continuam prevalecendo sobre os
padrões de cada modo. Portanto, para reproduzir um treino antigo, também use os
mesmos parâmetros que ele carregava. O modo é registrado no `info` do Gym e nos
manifestos dos treinos por currículo e campanhas de experimentos. No YAML de
treino, pode ser sobrescrito em `environment.dynamics_config.mechanics_model`.

## Convenção dos comandos

As ações continuam sendo forças em newtons na convenção da matriz de alocação
do ambiente. Para o URDF de referência, o comando equivalente no plugin é:

```
T_gazebo = [-T1, -T2, -T3, -T4, T5, T6]
```

Essa mudança de sinais é uma transformação de interface e não uma alteração
necessária na dinâmica interna. Os braços geométricos e limites permanecem os
mesmos.

## Limites do modelo

O modelo assume submersão completa, volume constante e corpo rígido equivalente;
não modela a travessia da superfície, juntas e movimento dos rotores. O fator
linear de volume `1.01` mantém o valor histórico do URDF (3.03% no volume).
A excitação por aceleração do fluido/RAOs não foi introduzida nesta correção.
As cinco ablações do modelo de ondas continuam disponíveis separadamente.

Os resultados de políticas antigas mudam com as correções mecânicas, mesmo
mantendo `disturbance_model.preset: std`. Reavalie/retreine as políticas para a
mecânica corrigida ou selecione explicitamente `mechanics_model: std` para
comparações históricas.

## Validação

```bash
python -m unittest discover -s tests -v
python scripts/validate_gazebo_dynamics.py
```

Os testes verificam energia cinética no centro de massa e na origem, potência
nula de Coriolis, acoplamento de yaw, forças/momentos hidrostáticos em orientações
e centros arbitrários, sinal da aceleração vertical e compatibilidade dos
coeficientes padrão com o YAML. A trajetória histórica usa a mecânica `std`.
