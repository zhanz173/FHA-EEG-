#!/bin/bash
#SBATCH --job-name=eeg-shard-preprocess
#SBATCH --account=def-vvakorin
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
trap 'echo "Failed at line $LINENO" >&2' ERR

: "${SLURM_TMPDIR:?SLURM_TMPDIR is not set}"

REPO_ROOT="$SCRATCH/FHA-EEG-"
PREPROCESS_DIR="$REPO_ROOT/preprocess"

INPUT_ROOT_ON_CLUSTER="$HOME/projects/rpp-doesburg/databases/eeg_fha/preprocessed/001_a01_01"
FINAL_OUTPUT_ROOT_ON_CLUSTER="$HOME/projects/rpp-doesburg/zhenyu/FHA/Resting_Features_4s"

USE_VENV=1
VENV_PATH="$HOME/venvs/eeg"

ROWS_PER_SHARD=400000
WRITE_BUFFER_SIZE=1000
COMPRESSION="gzip"
CHECKPOINT_PATH="$PREPROCESS_DIR/LaBraM/checkpoints/labram-base.pth"
STAGE_INPUT_TO_SCRATCH=0

module purge
module load StdEnv/2023 python/3.11 arrow

if [[ "$USE_VENV" -eq 1 ]]; then
  [[ -f "$VENV_PATH/bin/activate" ]] || { echo "Missing venv activate script: $VENV_PATH/bin/activate" >&2; exit 1; }
  source "$VENV_PATH/bin/activate"
fi

echo "Checking configured paths..."
[[ -d "$PREPROCESS_DIR" ]] || { echo "Missing preprocess dir: $PREPROCESS_DIR" >&2; exit 1; }
[[ -d "$INPUT_ROOT_ON_CLUSTER" ]] || { echo "Missing input dir: $INPUT_ROOT_ON_CLUSTER" >&2; exit 1; }
[[ -f "$CHECKPOINT_PATH" ]] || { echo "Missing checkpoint: $CHECKPOINT_PATH" >&2; exit 1; }
mkdir -p "$FINAL_OUTPUT_ROOT_ON_CLUSTER"

cd "$PREPROCESS_DIR"

RUN_INPUT_ROOT="$INPUT_ROOT_ON_CLUSTER"
if [[ "$STAGE_INPUT_TO_SCRATCH" -eq 1 ]]; then
  RUN_INPUT_ROOT="$SLURM_TMPDIR/eeg_input_${SLURM_JOB_ID}"
  mkdir -p "$RUN_INPUT_ROOT"
  echo "Staging input to local scratch: $RUN_INPUT_ROOT"
  time rsync -a --info=progress2 "$INPUT_ROOT_ON_CLUSTER/" "$RUN_INPUT_ROOT/"
fi

RUN_OUTPUT_ROOT="$SCRATCH/Resting_Features_4s"
mkdir -p "$RUN_OUTPUT_ROOT"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "Input root (source): $INPUT_ROOT_ON_CLUSTER"
echo "Input root (run): $RUN_INPUT_ROOT"
echo "Scratch output: $RUN_OUTPUT_ROOT"
echo "Final output: $FINAL_OUTPUT_ROOT_ON_CLUSTER"
echo "Checkpoint: $CHECKPOINT_PATH"
python --version
which python

time python extract_features_variable_length.py \
  --data_dir "$RUN_INPUT_ROOT" \
  --output_dir "$RUN_OUTPUT_ROOT" \
  --rows_per_shard "$ROWS_PER_SHARD" \
  --duration 4 \
  --overlap 0 \
  --write_buffer_size "$WRITE_BUFFER_SIZE" \
  --compression "$COMPRESSION" \
  --compression_opts 4 \
  --checkpoint_path "$CHECKPOINT_PATH"

STAMP=$(date +"%Y%m%d_%H%M%S")
DEST_DIR="$FINAL_OUTPUT_ROOT_ON_CLUSTER/run_${STAMP}_job${SLURM_JOB_ID}"
mkdir -p "$DEST_DIR"

rsync -a --info=progress2 "$RUN_OUTPUT_ROOT/" "$DEST_DIR/"

echo "Completed. Results at: $DEST_DIR"