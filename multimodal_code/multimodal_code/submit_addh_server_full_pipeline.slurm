#!/usr/bin/env bash
#SBATCH --job-name=addh_server_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# GPU request is cluster-dependent. If your Slurm supports one of these, add it
# at submit time, for example:
#   sbatch --gpus=1 submit_addh_server_full_pipeline.slurm
#   sbatch --gres=gpu:1 submit_addh_server_full_pipeline.slurm
#   sbatch --gres=gpu:a100:1 submit_addh_server_full_pipeline.slurm
# This script still sets CUDA_VISIBLE_DEVICES from GPU_ID below.

set -Eeuo pipefail

# Submit from the repository root:
#   sbatch submit_addh_server_full_pipeline.slurm
#
# Required before submit, or edit below:
#   export PY_MM=/path/to/conda/env/bin/python
#   export PY_FAIRCHEM=/path/to/fairchem/env/bin/python
#   export FAIRCHEM_MODEL_DIR=/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd
#
# Optional SCNet LLM call:
#   export RUN_SCNET_LLM=1
#   export SCNET_API_KEY=your_key

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export ROOT="${ROOT:-$(pwd)}"
export PY_MM="${PY_MM:-python}"
export PY_FAIRCHEM="${PY_FAIRCHEM:-$PY_MM}"
export FAIRCHEM_MODEL_DIR="${FAIRCHEM_MODEL_DIR:-/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd}"
export RUN_ID="${RUN_ID:-slurm_${SLURM_JOB_ID:-manual}}"
export RUN_ROOT="${RUN_ROOT:-$ROOT/runs_addh_server/$RUN_ID}"

export DEVICE="${DEVICE:-cuda}"
export GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

# Conservative defaults for a first server run.
export RUN_DATA_PREP="${RUN_DATA_PREP:-auto}"
export RUN_EMBEDDINGS="${RUN_EMBEDDINGS:-auto}"
export RUN_LLM_ROUTE="${RUN_LLM_ROUTE:-1}"
export RUN_SCNET_LLM="${RUN_SCNET_LLM:-0}"
export RUN_GRAPH_GRID="${RUN_GRAPH_GRID:-1}"
export RUN_MULTIVIEW_GRID="${RUN_MULTIVIEW_GRID:-1}"
export MULTIVIEW_SCOPE="${MULTIVIEW_SCOPE:-targeted}"
export RUN_STRICT_BLIND="${RUN_STRICT_BLIND:-1}"
export RUN_SERVER_FINAL_BLEND="${RUN_SERVER_FINAL_BLEND:-1}"
export RUN_AUDIT="${RUN_AUDIT:-0}"

export SEEDS="${SEEDS:-42,52,62,72,82,92,102,112,122,132}"
export TARGET_ABS_MAX="${TARGET_ABS_MAX:-10}"
export ADDH2_BASE_MILLER_MAP="${ADDH2_BASE_MILLER_MAP:-2542=100,2858=111,643=111}"
export ADDHOUT_MILLER_MAP="${ADDHOUT_MILLER_MAP:-CeO2=111,ZnO=100}"

mkdir -p "$RUN_ROOT"

echo "[SLURM] job_id=${SLURM_JOB_ID:-manual}"
echo "[SLURM] host=$(hostname)"
echo "[SLURM] root=$ROOT"
echo "[SLURM] run_root=$RUN_ROOT"
echo "[SLURM] py=$PY_MM"
echo "[SLURM] py_fairchem=$PY_FAIRCHEM"
echo "[SLURM] fairchem_model_dir=${FAIRCHEM_MODEL_DIR:-}"

bash run_addh_server_full_pipeline.sh
