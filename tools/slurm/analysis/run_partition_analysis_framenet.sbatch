#!/bin/bash
# Run exploration/partition_analysis.py on the FrameNet dataset.

#SBATCH --job-name=partition_framenet
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/analysis/partition_framenet_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/analysis"

OUTPUT_FILE=$REPO_ROOT/logs/analysis/partition_framenet_${SLURM_JOB_ID}.txt
FRAMENET_CACHE=$HF_DATASETS_CACHE/liyucheng__FrameNet_v17

python exploration/partition_analysis.py \
  --dataset framenet \
  --split train \
  --subset 1 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --partitions-per-length 512 \
  --framenet-cache "$FRAMENET_CACHE" \
  --output "$OUTPUT_FILE"
