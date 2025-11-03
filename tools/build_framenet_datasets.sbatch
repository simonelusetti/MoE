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
#SBATCH --output=/leonardo/home/userexternal/slusetti/MoE/outputs/build_framenet_%j.log

set -euo pipefail

module load python/3.11.7
source /leonardo/home/userexternal/slusetti/.venv/bin/activate

export HF_HOME=/leonardo_work/IscrC_LUSE/slusetti/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
# Allow downloads if the cache misses when running on the compute node.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /leonardo/home/userexternal/slusetti/MoE

mkdir -p outputs

python tools/build_dataset.py \
  --dataset framenet \
  --config fulltext \
  --splits train validation test \
  --subset 1.0
