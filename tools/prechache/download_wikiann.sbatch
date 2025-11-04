#!/bin/bash
# Populate the shared HF cache with WikiANN (en) and SBERT artefacts on Leonado.

#SBATCH --job-name=precache_wikiann
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/outputs/precache_wikiann_%j.log

set -euo pipefail

module load python/3.11.7

VENV_ROOT=${VENV_ROOT:-/leonardo/home/userexternal/slusetti/.venv311}
if [[ ! -d "$VENV_ROOT" ]]; then
  python3 -m venv "$VENV_ROOT"
fi

source "$VENV_ROOT/bin/activate"

export HF_HOME=/leonardo_work/IscrC_LUSE/slusetti/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

cd /leonardo/home/userexternal/slusetti/MoE

python tools/prechache/download.py \
  --cache-dir "$HF_HOME" \
  --wikiann-langs en \
  --models sentence-transformers/all-MiniLM-L6-v2
