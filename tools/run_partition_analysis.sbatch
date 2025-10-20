#!/bin/bash
# Run exploration/partition_analysis.py on a compute node.

#SBATCH --job-name=partition_analysis
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/IscrC_LUSE/slusetti/MoE-outputs/partition_analysis_%j.log

set -euo pipefail

module load python/3.11.7
source /leonardo/home/userexternal/slusetti/MoE/.venv/bin/activate

export HF_HOME=/leonardo_work/IscrC_LUSE/slusetti/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo/home/userexternal/slusetti/MoE

OUTPUT_FILE=/leonardo_work/IscrC_LUSE/slusetti/MoE-outputs/partition_analysis_${SLURM_JOB_ID}.txt

python exploration/partition_analysis.py --subset 0.1 --split train --model sentence-transformers/all-MiniLM-L6-v2 --output "$OUTPUT_FILE"
