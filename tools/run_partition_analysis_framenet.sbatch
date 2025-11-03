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
#SBATCH --output=/leonardo_work/IscrC_LUSE/slusetti/MoE-outputs/partition_framenet_%j.log

set -euo pipefail

module load python/3.11.7
source /leonardo/home/userexternal/slusetti/.venv/bin/activate

export HF_HOME=/leonardo_work/IscrC_LUSE/slusetti/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo/home/userexternal/slusetti/MoE

OUTPUT_FILE=/leonardo_work/IscrC_LUSE/slusetti/MoE-outputs/partition_framenet_${SLURM_JOB_ID}.txt
FRAMENET_CACHE=$HF_DATASETS_CACHE/liyucheng__FrameNet_v17

python exploration/partition_analysis.py \
  --dataset framenet \
  --split train \
  --subset 1 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --partitions-per-length 512 \
  --framenet-cache "$FRAMENET_CACHE" \
  --output "$OUTPUT_FILE"
