#!/bin/bash
#
# Pre-build FrameNet fulltext datasets on a compute node.

#SBATCH --job-name=build_framenet
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/datasets/build_framenet_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/datasets"

python tools/build_dataset.py \
  --dataset framenet \
  --config fulltext \
  --splits train validation test \
  --subset 1.0
