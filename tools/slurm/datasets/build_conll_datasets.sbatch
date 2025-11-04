#!/bin/bash
# Pre-build CoNLL-2003 train/validation datasets on a compute node.

#SBATCH --job-name=build_conll
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/logs/datasets/build_conll_%j.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../common.sh"

mkdir -p logs/datasets

python tools/build_dataset.py   --dataset conll2003   --splits train validation   --subset 1.0   --rebuild
