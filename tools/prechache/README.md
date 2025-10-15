# Dataset precache utilities

Utilities under this folder populate the Hugging Face cache with the datasets required by the MoE experiments. Pre-caching lets you run training jobs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` without hitting the network.

## Choosing a cache location

Pick a filesystem with enough quota (e.g. `/leonardo_work/<project>/<user>` on Leonardo) and create a directory that will hold all HF artefacts:

```bash
export HF_CACHE_ROOT=/leonardo_work/IscrC_LUSE/$USER/hf-cache
mkdir -p "$HF_CACHE_ROOT"
```

The precache script will manage the `hub/`, `datasets/`, and `transformers/` subfolders inside this root and export the appropriate environment variables (`HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE`) before downloading.

## Download script

`download_datasets.py` fetches both CoNLL-2003 and WikiANN splits:

```bash
module load python/3.11.7
source ~/RatCon/.venv/bin/activate

python tools/prechache/download_datasets.py   --cache-dir "$HF_CACHE_ROOT"   --wikiann-langs en
```

Script options:

- `--cache-dir`: target cache root (defaults to existing `HF_HOME` or `~/hf-cache`).
- `--revision`: CoNLL-2003 dataset revision (default: `refs/convert/parquet`).
- `--wikiann-langs`: space-separated WikiANN language codes to materialise (default: `en`).

## Integrating with Slurm jobs

Ensure your training config exports the same cache paths inside the job setup. For example in `src/conf/default.yaml`:

```yaml
slurm:
  setup:
    - "module load python/3.11.7"
    - "source $HOME/RatCon/.venv/bin/activate"
    - "export HF_HOME=/leonardo_work/IscrC_LUSE/$USER/hf-cache"
    - "export HF_DATASETS_CACHE=$HF_HOME/datasets"
    - "export TRANSFORMERS_CACHE=$HF_HOME/transformers"
    - "export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub"
    - "export HF_HUB_OFFLINE=1"
    - "export TRANSFORMERS_OFFLINE=1"
```

If you prefer the jobs to download new artefacts on demand, drop the last two lines (the offline flags).

## Batch helper (optional)

`precache_and_rerun.sbatch` is a sample script for running the download on a compute node and, if desired, re-launching a Dora grid afterwards. Adjust the `#SBATCH` directives and cache path to match your project before submitting:

```bash
sbatch tools/prechache/precache_and_rerun.sbatch
```
