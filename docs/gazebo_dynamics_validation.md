# Auditoria histórica da dinâmica contra URDF e Gazebo

Os problemas abaixo descrevem a implementação anterior às correções mecânicas.
Coriolis, transporte da inércia, hidrostática e os coeficientes padrão foram
corrigidos em `mechanics_model: corrected`, agora o padrão. O modo `std` permite
reproduzir as equações antigas. Veja [mechanics.md](mechanics.md).
O script compara o código atual, portanto já mostra esses termos corrigidos.

Referências locais: `_references/bluerov2/bluerov2_description/urdf/`:
`bluerov2.xacro`, `thrusters.xacro`, `hydrodynamics.xacro`, `sensors.xacro` e
`demo_world.sdf`. Dinâmica auditada: `bluerov2_gym/envs/core/dynamics.py`, com e
sem `configs/dynamics/dynamics_config.yaml`. A auditoria não modifica a mecânica do modelo.

## Escopo e limitações

Foi realizada comparação de parâmetros, reconstrução geométrica da alocação,
composição de inércias e avaliação numérica de forças. Não foi executada uma
simulação Gazebo: os comandos `gz` e `ign` não estão disponíveis nesta máquina.

Para reproduzir os cálculos usando os Xacros locais:

```bash
python scripts/validate_gazebo_dynamics.py
```

O Dockerfile da referência declara Fortress (`ign-gazebo6`), mas não fixa versão
de pacote, e os Xacros também usam nomes `gz-sim`. Foram consultados os fontes
oficiais disponíveis na branch `ign-gazebo6` em 2026-11-03. Revisões históricas do
plugin podem diferir. Uma equivalência de trajetórias precisa fixar a versão e
executar a descrição SDF efetivamente carregada no simulador.

Fontes oficiais:

- [Hydrodynamics.cc](https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/src/systems/hydrodynamics/Hydrodynamics.cc)
- [HydrodynamicsUtils.hh](https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/src/systems/hydrodynamics/HydrodynamicsUtils.hh)
- [Thruster.cc](https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/src/systems/thruster/Thruster.cc)
- [Buoyancy.cc](https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/src/systems/buoyancy/Buoyancy.cc)

## Resultado

Os coeficientes do YAML correspondem aos coeficientes ativos do Xacro, com
conversão de sinal apropriada. Isso não basta para equivalência dinâmica:
Coriolis, hidrostática e inércia rígida apresentam discrepâncias. Os sinais dos
comandos horizontais também precisam ser mapeados na comparação.

### 1. Coriolis: operador espacial incorreto

Para velocidades ordenadas como `[v, omega]` e momento `[P, H]`, a força de
Coriolis pode ser calculada por:

```
c = [omega x P, v x P + omega x H]
```

O operador de força correspondente tem blocos `[[S(omega), 0], [S(v), S(omega)]]`.
A implementação atual usa `[[S(omega), S(v)], [0, S(omega)]]`, trocando a posição
do bloco `S(v)`. Isso afeta tanto o corpo rígido quanto a massa adicionada.

Com `nu=[1,1,0,0,0,0]` e a massa adicionada do YAML:

- Fossen/Gazebo: `C_A(nu)*nu=[0,0,0,0,0,7.2]`.
- Código: vetor nulo; perde o acoplamento de yaw.
- Corpo rígido: o código também gera uma componente espúria de força Z
  `-3.53647755 N` em `C_RB*nu`, enquanto a expressão correta é nula nesse caso.

Com `nu=[1,0.5,-0.2,0.1,-0.2,0.3]`, a potência `nu.T*C_A(nu)*nu` deveria ser
zero; o código retorna `-0.6985 W`. Para corpo rígido retorna `-0.01768239 W`.
A expressão por produtos vetoriais retorna zero até precisão numérica.

### 2. Peso e empuxo: sinal translacional invertido para Z para cima

O mundo Gazebo e os centros negativos no Xacro são coerentes com Z positivo
para cima. Para corpo alinhado, plenamente submerso e sem ondas:

- massa do ambiente: `15.22 kg`;
- volume da caixa: `0.01568118122 m^3`;
- peso: `149.3082 N`;
- empuxo: `153.83238777 N`;
- força física resultante: `+4.52418777 N` em Z;
- código: `-4.52418777 N` em Z, pois a equação subtrai `g_eta`.

A comparação usa a mesma massa do ambiente para isolar o erro de sinal.
Incluindo a câmera de 15 g do URDF, o empuxo líquido seria aproximadamente
`+4.37704 N` para essa geometria plenamente submersa.

Em orientação genérica, as três componentes translacionais de `-g_eta` ficam
invertidas em relação à rotação das forças físicas peso/empuxo para BODY.

Os momentos restauradores têm sinal estabilizante na convenção Z para cima;
não são invertidos junto com as forças. Seu coeficiente aproximado é
`13.00981680 N m/rad`, contra `12.87841332` usando os centros e peso/empuxo do
próprio ambiente: diferença de aproximadamente 1.02%. A fórmula atual também
ignora deslocamentos X/Y configuráveis dos centros.

Isso retifica a interpretação NED da revisão inicial: centros negativos no
URDF não indicam instabilidade; nesse modelo, o centro de gravidade está abaixo
do centro de empuxo em coordenadas Z para cima.

### 3. Inércia: tensor no centro de massa usado na origem

O URDF fornece inércia no frame `<inertial>`, localizado em `z_cog`. O código
chama esse tensor de `I_g`, mas o insere diretamente no bloco rotacional da
matriz de massa na origem, embora mantenha os acoplamentos com `r_g`.

Falta a transformação `I_origin = I_g - m*S(r_g)^2`. Apenas para o corpo base
URDF de 14.8 kg, isso acrescenta `0.199762623 kg m^2` a Ixx e Iyy.

Compondo corpo, seis rotores e câmera, assumindo articulações congeladas na
configuração inicial e câmera sem rotação, o tensor na origem fica:

```
[[5.461780106, 0.014400000, 0.333973825],
 [0.014400000, 8.149747846, 0.026000000],
 [0.333973825, 0.026000000, 6.924036873]]
```

O bloco atual tem diagonal `[5.2539, 7.9420, 6.9123]`. A composição acima é uma
redução de corpo rígido, não uma simulação das articulações móveis.
Os elementos fora da diagonal do URDF já são entradas do tensor: seus sinais
positivos não devem ser invertidos automaticamente por uma notação de produtos
de inércia diferente em livros.

### 4. Massa e centro de gravidade compostos

A massa `14.8 + 6*0.07 = 15.22 kg` está correta para corpo e rotores. A câmera
ativa em `sensors.xacro` adiciona `0.015 kg`: total explícito `15.235 kg`.
Os links IMU/sonar não declaram massas adicionais ativas.

O código usa o centro do corpo base para toda a massa. Ao compor as massas e
os offsets dos rotores e câmera na configuração inicial, obtém-se:

```
r_G_composto = [0.000080079, 0, -0.114079012] m
r_G_ambiente = [0,           0, -0.116178632] m
```

### 5. Alocação: sinais horizontais opostos aos comandos do plugin

Reconstrução por coluna: direção `R(rpy)*joint_axis` e momento `r x direction`.
O plugin aplica força ao longo do eixo da articulação para comando positivo.
A matriz do ambiente satisfaz aproximadamente:

```
T_env = T_urdf @ diag([-1, -1, -1, -1, 1, 1])
```

O erro residual é menor que `4e-9`, compatível com arredondamento. Portanto,
geometria e braços são consistentes, mas os quatro comandos horizontais têm
sinal oposto. Isso pode ser uma convenção intencional: documentar o mapeamento
antes de comparar políticas ou aplicar comandos aprendidos ao Gazebo.

Os limites `[-40,40] N` coincidem. O plugin recebe força em newtons e, com
`velocity_control=true`, controla a velocidade do rotor sem usar o torque do
PID. Assim, não se deve interpretar a ação do ambiente como PWM ou RPM.

### 6. Massa adicionada e amortecimento

No YAML:

```
M_A diagonal = [5.5, 12.7, 14.57, 0.12, 0.12, 0.12]
D_linear     = [25.15, 7.364, 17.955, 10.888, 20.761, 3.744]
D_quadratic  = [17.77, 125.9, 72.36, 0.1246, 0.1246, 0.1857]
```

Correspondem aos valores ativos negativos no plugin. Construir matrizes
positivas no Python e subtrair amortecimento é consistente com essa convenção.
O problema de Coriolis é independente da escolha correta desses coeficientes.

Sem carregar o YAML, `Dynamics()` usa amortecimento quadrático diferente:
`[33.8,54.26875,73.37135,40,40,40]`. Em roll a `0.5 rad/s`, só a parcela
quadrática passa de `0.03115` para `10 N m`. `gym.make('BlueRov-v0')` não carrega
automaticamente `configs/dynamics/dynamics_config.yaml`; o runner antigo também não o
carregava. O preset de ondas não seleciona um conjunto de parâmetros mecânicos.

### 7. Volume, superfície e integração

A fórmula de correção do volume é a mesma do Xacro. O fator linear `1.01` é
aplicado às três dimensões: o volume é multiplicado por `1.01^3=1.030301`, não
por `1.01`. O comentário do URDF que diz 1% não descreve seu resultado efetivo.

O Gazebo usa densidade estratificada (água abaixo de Z=0, ar acima), com empuxo
dependente da parte submersa. O ambiente mantém empuxo integral em qualquer Z.
A comparação hidrostática acima vale para submersão completa.

O mundo declara passo de `0.01 s`; o ambiente usa `0.1 s`, velocidade atualizada
antes da posição e ângulos de Euler. A implementação legada do plugin estima a
aceleração por diferenças de velocidades; o Python resolve a massa total
diretamente. Mesmo após igualar os termos contínuos, trajetórias discretas não
precisam coincidir sem estudo de convergência do passo.

## Prioridades

1. Corrigir o operador de Coriolis sob uma opção mecânica explícita.
2. Fixar e documentar a convenção Z para cima e corrigir as forças hidrostáticas.
3. Transformar/compor os parâmetros inerciais no mesmo frame.
4. Documentar o mapeamento dos comandos horizontais e usar o YAML explicitamente.
5. Só então comparar respostas temporais contra uma versão fixa do Gazebo.

O preset de ondas é independente da mecânica. As correções mecânicas agora são
ativadas por `mechanics_model: corrected`; use `mechanics_model: std` para
reproduzir o comportamento anterior.
