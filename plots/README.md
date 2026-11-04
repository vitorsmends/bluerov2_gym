# Compatibilidade: figuras do artigo

A implementação está em `bluerov2_gym/workflows/plotting/article/`.
Os scripts desta pasta continuam funcionando com os caminhos históricos.

Para uso novo, prefira:

```bash
bluerov plot --config configs/plots/article_legacy.yaml
bluerov plot --run runs/evaluation/SEU_RUN --suite article
```

Consulte [o guia de plots](../docs/plotting.md) e
[o mapa de migração](../docs/migration.md). As rotinas de desenho, agregação
e energia por curva T200 foram preservadas.
