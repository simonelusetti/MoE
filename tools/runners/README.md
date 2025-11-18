# Local Runners

Sequential pseudo-grid scripts for launching expert or composite experiments without Slurm. They iterate over the standard dataset overrides and run each configuration with `python -m <train_module>`.

## `pseudo_grid.py`
- Routes to `src.train`.
- Usage examples:
  ```bash
  python tools/runners/pseudo_grid.py --only wikiann conll2003 --extra train.epochs=3
  python tools/runners/pseudo_grid.py --dry-run
  ```

## `pseudo_grid_composite.py`
- Routes to `src.train_composite`.
- Usage example:
  ```bash
  python tools/runners/pseudo_grid_composite.py --python python3.11 --extra composite.selector_threshold=0.4
  ```

Shared flags:
- `--only LABEL ...` to pick a subset of datasets.
- `--extra key=value ...` to append Hydra overrides.
- `--dry-run` to print commands without running them.

Both scripts use the shared helpers in `_pseudo_grid_base.py` so any new runner just needs to provide a default training module and optional label prefix.
