#!/bin/bash
#SBATCH -J ratcon_precache
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/datasets/precache_and_rerun_%j.log

set -euo pipefail

REPO_ROOT=/leonardo/home/userexternal/slusetti/MoE
source "$REPO_ROOT/tools/slurm/common.sh"

mkdir -p "$REPO_ROOT/logs/datasets"

python -m tools.precache --rerun-grid
