#!/bin/bash
# Populate the shared HF cache with WikiANN (en) and SBERT artefacts on Leonado.

#SBATCH --job-name=precache_wikiann
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/datasets/precache_wikiann_%j.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"

mkdir -p logs/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

python tools/prechache/download.py \
  --cache-dir "$HF_HOME" \
  --wikiann-langs en \
  --models sentence-transformers/all-MiniLM-L6-v2
