#!/bin/bash
# Train the product-manifold model via Dora on a compute node.

#SBATCH --job-name=train_product
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/training/train_product_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/training"

# Forward any additional Hydra overrides passed to sbatch.
dora --package src --main_module train_product run "$@"
