#!/bin/bash
# Common environment setup for SLURM jobs.

set -euo pipefail

module load python/3.11.7

source "$HOME/.venv/bin/activate"

export HF_HOME=${HF_HOME:-/leonardo_work/IscrC_LUSE/slusetti/hf-cache}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

export SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}

cd /leonardo/home/userexternal/slusetti/MoE
