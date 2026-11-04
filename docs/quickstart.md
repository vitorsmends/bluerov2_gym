# Guia rápido

Instale com `python -m pip install -e .`. Todos os comandos abaixo podem ser
substituídos por `python -m bluerov2_gym.workflows ...`.

1. Abra `configs/evaluation/stationkeeping_single.yaml`: ele identifica os dois
   arquivos do seu agente atual, setpoint, duração e seed.
2. Confira: `bluerov inspect --config configs/evaluation/stationkeeping_single.yaml`.
3. Execute: `bluerov evaluate --config configs/evaluation/stationkeeping_single.yaml`.
4. Copie o caminho `Run:` impresso ao terminar.
5. Execute: `bluerov plot --run CAMINHO_DO_RUN --suite single_policy`.
6. Abra `CAMINHO_DO_RUN/figures/single_policy/errors.png`.

Para escolher um diretório em vez do timestamp:

```bash
bluerov evaluate --config configs/evaluation/stationkeeping_single.yaml \
  --output runs/evaluation/meu_teste
bluerov plot --run runs/evaluation/meu_teste --suite single_policy
```

O diretório de avaliação precisa ser novo. Repetir apenas o comando `plot` é
permitido: ele regenera figuras sem atualizar pesos ou estatísticas.

Caminhos dentro dos novos YAMLs são relativos ao arquivo YAML, não ao terminal.
Passe o caminho absoluto de um YAML se estiver fora do repositório.
