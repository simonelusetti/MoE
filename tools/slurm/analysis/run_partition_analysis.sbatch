#!/bin/bash
# Run exploration/partition_analysis.py on a compute node.

#SBATCH --job-name=partition_analysis
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/analysis/partition_analysis_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/analysis"

OUTPUT_FILE=$REPO_ROOT/logs/analysis/partition_analysis_${SLURM_JOB_ID}.txt

python exploration/partition_analysis.py \
  --subset 1 \
  --split train \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --partitions-per-length 4096 \
  --output "$OUTPUT_FILE"
