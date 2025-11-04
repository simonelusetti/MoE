# SLURM Utilities

This folder groups all batch scripts by task type:

- `datasets/`: dataset materialisation, cache population, and related utilities.
- `training/`: training jobs (e.g., product-manifold runs).
- `analysis/`: exploratory or evaluation jobs (partition analysis).
- `common.sh`: shared environment bootstrap (module load, virtualenv activation, HF caches).

## Conventions

- Logs are written under `logs/{datasets|training|analysis}/` in the repository root.
- Scripts source `common.sh`, so update that file if the virtualenv or cache paths change.
- Additional Hydra overrides can be passed to training scripts after `--` on submission, e.g.:
  ```
  sbatch tools/slurm/training/train_product.sbatch -- data.train.subset=0.1
  ```
- Dataset scripts call `tools/build_dataset.py`; ensure datasets/caches exist before launching training jobs.
