# Compatibilidade: avaliações e controladores

A implementação está em `bluerov2_gym/workflows/evaluation/`.
Os módulos e comandos desta pasta permanecem como entradas compatíveis.

O fluxo recomendado é:

```bash
bluerov inspect --config configs/evaluation/article.yaml
bluerov evaluate --config configs/evaluation/article.yaml
bluerov plot --run runs/evaluation/SEU_RUN --suite article
```

A seleção de controladores é feita no YAML (`controllers[].enabled`). Confira
os caminhos do modelo e do VecNormalize antes de iniciar a campanha.

Consulte [o guia de avaliação](../docs/evaluation.md),
[o guia de plots](../docs/plotting.md) e [dados/LFS](../DATA.md).
