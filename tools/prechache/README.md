# Dataset precache utilities

Utilities under this folder populate the Hugging Face cache with the datasets and models required by the MoE experiments. Pre-caching lets you run training jobs with offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`) without hitting the network.

## Choosing a cache location

Pick a filesystem with enough quota (e.g. `/leonardo_work/<project>/<user>` on Leonardo) and create a directory that will hold all HF artefacts:

```bash
export HF_CACHE_ROOT=/leonardo_work/IscrC_LUSE/$USER/hf-cache
mkdir -p "$HF_CACHE_ROOT"
```

The precache script manages the `hub/`, `datasets/`, and `transformers/` subfolders inside this root and exports the environment variables `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and `HF_HUB_CACHE` before downloading.

## Download & verify

`download.py` fetches CoNLL-2003, WikiANN splits, and all necessary transformer artefacts (config, tokenizer, weights) for each model you list:

```bash
module load python/3.11.7
source ~/MoE/.venv/bin/activate

python tools/prechache/download.py   --cache-dir "$HF_CACHE_ROOT"   --wikiann-langs en   --models sentence-transformers/all-MiniLM-L6-v2
```

Script options (pass `--no-verify` to skip the offline check):

- `--cache-dir`: target cache root (defaults to existing `HF_HOME` or `~/hf-cache`).
- `--revision`: CoNLL-2003 dataset revision (default: `refs/convert/parquet`).
- `--wikiann-langs`: space-separated WikiANN language codes to materialise (default: `en`).
- `--models`: transformer model ids to cache (default: `sentence-transformers/all-MiniLM-L6-v2`).

## Integrating with Slurm jobs

Ensure your training config exports the same cache paths inside the job setup. For example in `src/conf/default.yaml`:

```yaml
slurm:
  setup:
    - "module load python/3.11.7"
    - "source $HOME/MoE/.venv/bin/activate"
    - "export SLURM_CPU_BIND=none"
    - "export HF_HOME=/leonardo_work/IscrC_LUSE/$USER/hf-cache"
    - "export HF_DATASETS_CACHE=$HF_HOME/datasets"
    - "export TRANSFORMERS_CACHE=$HF_HOME/transformers"
    - "export HF_HUB_CACHE=$HF_HOME/hub"
    - "export HF_HUB_OFFLINE=1"
    - "export TRANSFORMERS_OFFLINE=1"
    - "export HF_DATASETS_OFFLINE=1"
```

If you prefer jobs to download new artefacts on demand, drop the last three offline flags.

## Batch helper (optional)

`precache_and_rerun.sbatch` is a sample script for running the download on a compute node and, if desired, re-launching a Dora grid afterwards. Adjust the `#SBATCH` directives and cache path to match your project before submitting:

```bash
sbatch tools/prechache/precache_and_rerun.sbatch
```

## Pre-build processed datasets

Use `tools/build_dataset.py` to materialise the processed MoE datasets under `./data` ahead of time. This mirrors the naming logic in `src/data.py` so the training code can operate purely offline:

```bash
module load python/3.11.7
source ~/MoE/.venv/bin/activate

python tools/build_dataset.py --dataset wikiann --splits train validation --subset 0.0
```

Pass `--rebuild` to force regeneration if a cached dataset already exists.
