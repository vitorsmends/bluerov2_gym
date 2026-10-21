# Conferência das equações do artigo e do simulador

> Registro histórico da auditoria anterior às correções. Os blocos implementados
> posteriormente estão descritos em [physics_conventions.md](physics_conventions.md),
> versão `zup-airy-coriolis-v2`. Os valores de resultados anteriores não são v2.

Referência: `oceans2026/cas-dc-template.tex`, PDF recompilado após os ajustes
visuais, e `bluerov2_gym/envs/core/dynamics.py`. Auditoria em 21/10/2026.
Nenhuma equação do simulador, CSV ou figura foi alterada nesta conferência.

## JONSWAP: consistente após a correção

A equação 24 e a implementação usam

S(ω) = Aγ (5/16) Hs² ωp⁴ ω⁻⁵ exp[-(5/4)(ωp/ω)⁴] γ^r,
Aγ = 1 - 0.287 ln γ,
r = exp[-(ω-ωp)²/(2σ²ωp²)], ωp = 2π/Tp.

Corresponde ao caso 7 de waveSpectrum.m do MSS/Fossen. O coeficiente alpha
é adimensional; S em frequência angular tem unidade m² s, com radiano
adimensional. A_i=sqrt(2 S_i Δω) tem unidade de comprimento. A correção
analítica de energia é aproximada, não uma identidade exata para todo gamma.

Com Tp=8 s, gamma=3.3, N=64 e corte em 3ωp, a energia discreta fornece
Hm0=4 sqrt(sum(A_i²)/2):

| Hs informado | Hm0 reconstruído |
|---:|---:|
| 2.50 m | 2.491087 m |
| 4.50 m | 4.483957 m |

As equações de r, sigma, frequência de pico, amplitude espectral, fase,
superposição da elevação e dispersão são consistentes com o código.
O texto deve explicitar Δω=3ωp/N, ω_i=iΔω, a distribuição uniforme das
fases e das direções no intervalo de espalhamento. A amostragem de uma
direção por frequência é uma aproximação espectral direcional.

## Equação 31: velocidade orbital não é a expressão exata em profundidade finita

O artigo e o código usam A_i ω_i / cosh(k_i d) como prefator. Quando A_i
é a amplitude da elevação e ω²=gk tanh(kd), a expressão de Airy é

v_wave = Σ A_i ω_i / sinh(k_i d) ×
[cosθ_i cosh(k_i(z+d)) cosΨ_i,
 sinθ_i cosh(k_i(z+d)) cosΨ_i,
 sinh(k_i(z+d)) sinΨ_i]^T.

Equivalentemente, pode-se manter cosh(kd) no denominador se o numerador
Aω for substituído por Agk/ω. A troca entre essas duas formas exige a
relação de dispersão; não se pode trocar somente o denominador.

Prova independente: em z=0, a condição cinemática linear exige
w=∂η/∂t=Aω sinΨ para z positivo para cima. A expressão atual produz
Aω tanh(kd) sinΨ. Com Tp=8 s e d=30 m, tanh(kp d)=0.961272: no pico,
a amplitude atual é aproximadamente 3.87% menor que a expressão finita
correta. O erro depende da frequência e aumenta para componentes longas.
O sinal positivo de sinΨ é coerente com Ψ=kx-ωt+φ e z positivo para cima.

## Convenção vertical e referenciais: inconsistência a resolver

O artigo declara NED (z positivo para baixo), mas a cinemática das ondas
usa z≤0 abaixo da superfície e os alvos de profundidade são negativos.
É necessário distinguir explicitamente a coordenada vertical de ondas da
coordenada do veículo e definir a transformação entre elas, ou adotar uma
convenção consistente no conjunto. Alterar apenas os nomes não resolve os
sinais das forças e momentos.

Além disso, a velocidade gerada em direções espaciais fixas é subtraída
diretamente da velocidade do corpo. A relação física exige v_c^b=R_n^b v_c^n,
com a transformação vertical correspondente à convenção escolhida.
A equação 12 precisa usar o vetor 6D [u_c,v_c,w_c,0,0,0]^T; o símbolo nu_c
passa a significar um vetor 3D mais adiante, tornando a subtração ambígua.

## Coriolis: erro confirmado na implementação

Para ν=[v;ω], o operador de produto vetorial espacial de forças deve atuar
sobre o momento [P;H] como

ad*ν = [[S(ω), 0], [S(v), S(ω)]].

O código usa [[S(ω), S(v)], [0, S(ω)]]. O bloco S(v) está na posição errada.
Isso mistura momento angular na equação de força e omite v×P no momento.
Pode-se evitar esse erro implementando diretamente a construção m2c do Fossen.

Teste reproduzível com os parâmetros de config/dynamics_config.yaml,
ν=[0.4,-0.2,0.1,0.1,-0.15,0.2] e corrente nula:

- νᵀ(c_RB+c_A) atual = 0.0455114331.
- Com o operador correto: -5.55e-17, isto é, zero à precisão numérica.

Coriolis deve ser energeticamente neutro: νᵀC(ν)ν=0. O teste é em corrente
nula para não confundir transferência de energia pelo fluido com erro do
operador. As equações do artigo apresentam o termo padrão, mas o código não
implementa esse termo corretamente.

## Equação 7: origem do tensor de inércia

O artigo define I_g como tensor no centro de gravidade, mas o insere sem
transformação no bloco inferior direito de M_RB referido à origem do corpo,
com r_g≠0. Nesse caso, o bloco deve ser I_O=I_G-m S(r_g)S(r_g).
Se os valores disponíveis já forem referidos à origem, a nomenclatura deve
ser I_O, e a correção não deve ser aplicada duas vezes.

O código faz a mesma escolha do artigo. A procedência dos valores de inércia
precisa ser confirmada antes de mudar números. Se forem I_G, o maior elemento
da correção paralela com a configuração atual é 0.205432 kg m².

## Equações 16 e 19: corretas na convenção do Fossen; código diferente

A forma geral e a redução x_g=y_g=x_b=y_b=0 correspondem a gvect.m.
Porém o código usa |z_b-z_g| W nos momentos de roll e pitch, enquanto o artigo
usa z_g W-z_b B. Na configuração atual:

- Coeficiente da equação do artigo: -12.8784133 N m.
- Coeficiente implementado: +13.0098168 N m.

Não é apenas uma diferença de arredondamento: o sinal é oposto. Como B não
é exatamente W, nem a igualdade de módulos vale. A decisão física correta
exige resolver conjuntamente as posições dos centros e a convenção vertical;
não se deve trocar isoladamente esse sinal e presumir o sistema validado.

## Equação 38: inconsistente com a equação dinâmica inicial

A equação inicial separa C_RB(ν)ν e C_A(ν_r)ν_r. A equação 38 volta a escrever
C(ν)ν. Com corrente não nula, não é a mesma expressão sem definição adicional.
Para coerência documental, deve repetir a decomposição anterior.

Para ondas/correntes aceleradas, uma derivação hidrodinâmica completa também
precisa esclarecer os termos de aceleração do fluido e a eventual excitação
inercial associada à massa adicionada. O modelo atual é uma aproximação com
velocidade relativa e carga quadrática adicional.

## Cargas de onda: expressão empírica, descrição incompleta

F=K(v⊙|v|) e M=r×F correspondem ao núcleo do código. K tem unidade
N s²/m². O código aplica escala, saturação da velocidade, ruído e dois
filtros antes da carga, além de saturar a norma da força e do momento.
As equações do artigo mostram apenas o núcleo sem esses operadores.

Uma carga quadrática não é uma carga de primeira ordem em amplitude. O texto
que afirma capturar a excitação de primeira ordem deve ser corrigido ou
qualificado. Essa formulação não é a equação completa de Morison, pois não
inclui explicitamente sua parcela inercial. Sua adição ao arrasto relativo
precisa ser descrita como aproximação empírica, sem afirmar equivalência a
uma derivação completa das cargas de ondas.

## Blocos sem divergência algébrica identificada

- Cinemática J(η)ν: as expressões de rotação e taxas de Euler implementadas
  são compatíveis com a forma matricial; permanece a questão da convenção.
- Operador S(a), massa adicionada diagonal positiva e amortecimento linear
  mais quadrático: coerentes com as equações e configuração.
- Alocação τ=B_T f e saturação dos comandos: coerentes.
- Recompensa e bônus por passo no estudo curricular: coerentes.
- Distância euclidiana, RMSE, média de CPU, interpolação de potência, energia
  acumulada e desvio padrão populacional entre trajetórias: coerentes com
  o processamento dos resultados.
- Não há equações detalhadas dos controladores PID/SMC/NMPC nem da perda PPO
  no texto auditado; não se deve interpretar esta conferência como validação
  dessas derivações ausentes.

## Consequência para os resultados atuais

O espectro JONSWAP de elevação foi corrigido e está verificado. Isso não
valida automaticamente os demais blocos físicos. Os CSVs existentes descrevem
o simulador com os problemas acima. Correções físicas exigem nova avaliação
dos controladores e regeneração dos resultados; corrigir só o LaTeX não resolve.
Nenhuma dessas mudanças foi aplicada nesta auditoria.

## Fontes primárias

- Fossen/MSS, JONSWAP: https://github.com/cybergalactic/MSS/blob/master/LIBRARY/environment/waveSpectrum.m
- Fossen/MSS, Coriolis: https://github.com/cybergalactic/MSS/blob/master/LIBRARY/modeling/m2c.m
- Fossen/MSS, corpo rígido: https://github.com/cybergalactic/MSS/blob/master/LIBRARY/modeling/rbody.m
- Fossen/MSS, restauração: https://github.com/cybergalactic/MSS/blob/master/LIBRARY/modeling/gvect.m
- MIT, velocidades de Airy: https://web.mit.edu/13.021/13021_2003/lecturenotes/lecture21.pdf
