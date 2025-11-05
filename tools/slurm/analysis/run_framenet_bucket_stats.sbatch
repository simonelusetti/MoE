#!/bin/bash
# Generate FrameNet bucket statistics on a compute node.

#SBATCH --job-name=framenet_stats
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/analysis/framenet_bucket_stats_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/analysis"

python exploration/framenet_bucket_stats.py \
  --split train \
  --subset 1.0 \
  --framenet-cache "$HF_DATASETS_CACHE/liyucheng__FrameNet_v17"
