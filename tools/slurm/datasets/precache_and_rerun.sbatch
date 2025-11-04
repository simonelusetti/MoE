#!/bin/bash
#SBATCH -J ratcon_precache
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/datasets/precache_and_rerun_%j.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"

mkdir -p logs/datasets

python -m tools.prechache --rerun-grid
