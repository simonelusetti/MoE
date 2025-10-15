#!/bin/bash
#SBATCH -J ratcon_precache
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH -o logs/precache_and_rerun-%j.out
#SBATCH -e logs/precache_and_rerun-%j.err

set -euo pipefail

mkdir -p logs

module load python/3.11.7
source "$HOME/RatCon/.venv/bin/activate"

export HF_HOME=${HF_HOME:-$HOME/hf-cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH=$PWD

python -m tools.prechache --rerun-grid
