# SLURM Utilities

This folder groups all batch scripts by task type:

- `datasets/`: dataset materialisation, cache population, and related utilities.
- `training/`: Expert MoE training jobs (product-manifold scripts now live under `product_manifold/tools/slurm/`).
- `analysis/`: exploratory or evaluation jobs (partition analysis).
- `common.sh`: shared environment bootstrap (module load, virtualenv activation, HF caches).

## Conventions

- Logs are written under `logs/{datasets|training|analysis}/` in the repository root.
- Scripts source `common.sh`, so update that file if the virtualenv or cache paths change.
- Additional Hydra overrides can be passed to training scripts after `--` when submitting via `sbatch`.
- Dataset scripts call `tools/build_dataset.py`; ensure datasets/caches exist before launching training jobs.
