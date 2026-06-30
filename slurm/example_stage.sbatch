#!/bin/bash
#SBATCH --job-name=slm-stage
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=runs/poc/slurm_logs/%x-%j.out

set -euo pipefail
STAGE="${1:?usage: sbatch example_stage.sbatch <stage> <config>}"
CONFIG="${2:?usage: sbatch example_stage.sbatch <stage> <config>}"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src

case "$STAGE" in
  pretrain-multi)
    torchrun --standalone --nproc_per_node=4 -m slm.pretrain --config "$CONFIG" ;;
  *)
    python3 -m "slm.${STAGE}" --config "$CONFIG" ;;
esac
