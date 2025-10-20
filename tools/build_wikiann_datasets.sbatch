#!/bin/bash
#
# Pre-build WikiANN train/validation datasets on a compute node.

#SBATCH --job-name=build_wikiann
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/outputs/build_wikiann_%j.log

set -euo pipefail

module load python/3.11.7
source /leonardo/home/userexternal/slusetti/MoE/.venv/bin/activate

export HF_HOME=/leonardo_work/IscrC_LUSE/slusetti/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /leonardo/home/userexternal/slusetti/MoE

mkdir -p /leonardo/home/userexternal/slusetti/MoE/outputs

python tools/build_dataset.py   --dataset wikiann   --splits train validation   --subset 1.0
