# muon_research migration record

## Scope

This project was initialized on 2026-11-17 as a selective, sparse Git checkout.
It is intentionally a new working area: no pre-existing experiment directory was
renamed, copied, or deleted.

## Upstream provenance

- Upstream repository: `https://github.com/SiennaDai/benchmark-experiments.git`
- Pinned archive commit: `ea684ef1627c4a679da82bfbb4a5eb0d56e19eb5`
- Commit subject: `Add Muon spectral and residual experiment reports`
- Checkout mode: detached, sparse checkout, partial clone (`blob:none`)
- License retained in `platform/LICENSE`: Apache-2.0.

`platform/.git` retains the upstream remote and the pinned commit.  The sparse
working tree is the authoritative record of which upstream source files were
materialized; `git sparse-checkout list` prints the selection.

## Materialized content

`platform/` contains the reusable platform: `src/`, `recipes/`, `tests/`,
`benchmarks/`, dependency files, the data-preparation/evaluation/launch scripts,
and the operational, protocol, validation, upstream-provenance and optimizer
extension documentation listed in its sparse-checkout specification.

## Deliberate exclusions

- No `data/` content: it is not Git-tracked at the pinned commit.  The platform
  retains `scripts/prepare_data.py` and the manifest/hash verification code.
- No `runs/`, checkpoints, tensor snapshots, caches, virtual environments,
  Python bytecode, or old local `experiments/` files.
- No `reports/`, `assets/`, or `scripts/analyze_*.py`: these are prior-study
  outputs or report-specific analysis programs, not required to run the base
  training/evaluation platform.
- No automatic download or materialization of SlimPajama.  Any future dataset
  acquisition must be performed explicitly and its manifest retained beside it.

## New-project directory contract

- `platform/`: the pinned, reusable platform checkout only.
- `data/`: future data manifests and intentionally acquired datasets; ignored by
  the project root.  Do not place caches or many-small-file corpora here.
- `analysis/`: future analysis code; `analysis/cache/` is local and ignored.
- `results/`: future generated metrics, figures and reports; ignored.
- `research/`: future project notes and literature records.

## Reproduction boundary

The selected platform can run synthetic-data tests offline.  The legacy
SlimPajama recipes require an external public source (`DKYoon/SlimPajama-6B`)
and a generated, hash-verified manifest.  They do not imply that any old
dataset, source cache, run, report, or public intermediate result has been
migrated.
