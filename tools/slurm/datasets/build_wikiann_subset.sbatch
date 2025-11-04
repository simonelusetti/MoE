#!/bin/bash
# Pre-build WikiANN train subset caches (including fractional subset 0.1).

#SBATCH --job-name=build_wikiann_subset
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/datasets/build_wikiann_subset_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/datasets"

python tools/build_dataset.py \
  --dataset wikiann \
  --splits train \
  --subset 0.1 \
  --rebuild
