# Dados e artefatos

Todo dado gerado tem como destino padrão `artifacts/`. A origem desse caminho
é única: `bluerov2_gym/workflows/paths.py`. Os comandos funcionam fora da raiz do
repositório quando o YAML é passado pelo caminho absoluto.

```text
artifacts/
  models/
    current/                 # agente/VecNormalize que estavam na raiz
    curriculum/              # modelos e checkpoints que estavam em models/
    extracted/               # pesos PyTorch e metadados extraídos
  training/
    <run_id>/                # novos treinos; models/, logs/, config e manifesto
    imported/                # antigos training_runs, estrutura interna preservada
  evaluation/
    <run_id>/                # novas avaliações; data/, figures/, tables/, logs/
    imported/                # resultados históricos; antigos results*/
  logs/
    tensorboard/             # antigos bluerov_tensorboard/
    tensorboard_ftc/
  figures/                   # saídas de plots fora de uma execução específica
  tables/                    # tabelas fora de uma execução específica
  optimization/              # banco Optuna e seus logs
  control_design/            # relatórios JSON de PID/LQR/NMPC
  archive/                   # backups como experiments.zip
  migration/                 # mapas de origem/destino e checksums da reorganização
```

Diretórios são criados quando necessários. Subpastas são criadas somente
quando algum dado precisa ser escrito. `--output`, configurações explícitas e
`export-figures --destination` permitem destinos personalizados; sem override,
as saídas permanecem em `artifacts/`.

## Modelos atuais

O agente de station keeping solicitado está em:

- `artifacts/models/current/bluerov_ppo.zip`
- `artifacts/models/current/bluerov_vec_normalize.pkl`

Os YAMLs ativos já usam esses destinos. Use sempre o normalizador correspondente
ao modelo. Os checkpoints de currículo preservam nomes e seeds em
`artifacts/models/curriculum/`; políticas de outras campanhas permanecem em
`artifacts/training/imported/`, junto dos seus arquivos originais.

Os arquivos de manifesto históricos continuam intactos, inclusive caminhos
registrados na época em que foram produzidos. Para localizar um arquivo movido,
consulte os mapas em `artifacts/migration/`. Não se deve reescrever a proveniência
de um experimento antigo como se ele tivesse sido executado na organização nova.

## Integridade e limpeza

A migração verificou SHA-256 antes e depois de cada arquivo movido. Nenhum modelo,
log de treinamento ou resultado numérico foi descartado. Foram removidos apenas
metadados `.DS_Store`, cópias temporárias de build/cache e pastas vazias fora dos
ambientes e referências protegidos.

O utilitário é reutilizável para conferir se restaram dados nos caminhos antigos:

```bash
python scripts/organize_artifacts.py          # apenas lista movimentos pendentes
python scripts/organize_artifacts.py --apply  # move com checksums e remove vazios
```

Não sobrescreve destinos existentes. Não atravessa `.git`, ambientes virtuais
ou `_references` para excluir pastas vazias.

## Git LFS

Os artefatos já versionados continuam versionados nos novos caminhos. Novos
artefatos locais são ignorados por padrão, para não incluir cada avaliação ou
checkpoint acidentalmente. Para preservar uma execução específica, use
`git add -f` somente nos arquivos desejados.

CSV e TensorBoard usam Git LFS, assim como `.zip`, `.pkl` e `.pth` sob `artifacts/`.
Após clonar, execute `git lfs pull` para obter os objetos disponíveis. A presença
de um ponteiro LFS não comprova que o conteúdo está baixado.

Seed de treinamento e seed de distúrbio continuam sendo conceitos distintos.
Uma tabela agregada ou um nome de pasta não comprova múltiplos treinamentos;
confira os manifestos e os arquivos originais antes de interpretar variabilidade.
