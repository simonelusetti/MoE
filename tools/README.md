# Tooling Overview

This directory hosts helper scripts for both local experimentation (no Slurm) and HPC runs that rely on pre-cached Hugging Face assets.

## Local runners (no Slurm required)

- `pseudo_grid.py` / `pseudo_grid_composite.py`: sequential “grid” launchers that iterate over the standard dataset overrides. Use them to sanity-check the expert or composite trainers on workstations:
  ```bash
  python tools/pseudo_grid.py --only wikiann conll2003 --extra train.epochs=3
  python tools/pseudo_grid_composite.py --dry-run
  ```
- These scripts accept arbitrary Hydra overrides via `--extra`, so they mirror the Slurm launcher arguments without needing `sbatch`.

## Dataset utilities

- See `tools/datasets/` for scripts that materialise model-ready caches (`build_dataset.py`) or download raw corpora for exploration (`download_dataset.py`). Each script documents the output format and expected usage in `tools/datasets/README.md`.

## HPC / Slurm workflow

- `tools/precache/`: manages offline Hugging Face caches (datasets, tokenizer weights). Run `python tools/precache/download.py --cache-dir <path>` to mirror required artefacts before submitting Slurm jobs, or `python -m tools.precache --rerun-grid` to refresh caches and restart a Dora grid.
- `tools/slurm/`: contains batch scripts grouped by task (`datasets/`, `training/`, `analysis/`). They assume caches were prepared with the `precache` helpers and call into the same dataset builders used locally.

Keeping these utilities together ensures the project runs the same way on laptops (pseudo-grid) and clusters (Slurm + precache) with minimal duplication. Add new scripts under the appropriate subdirectory and update the relevant README so both workflows stay discoverable.
