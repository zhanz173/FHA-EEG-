#!/bin/bash
#SBATCH --job-name=eeg-env-smoke
#SBATCH --account=def-vvakorin
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Submit this script from a login node with:
#   sbatch preprocess/run_test_env_cpu.slurm
#
# This is a CPU-only smoke test. It checks that the job can read input FIF files,
# import the Python environment, load the LaBraM checkpoint, extract a few rows,
# and copy the result back to shared/cloud storage.

# -------- User config (edit these) --------
REPO_ROOT="$SCRATCH/FHA-EEG-"
PREPROCESS_DIR="$REPO_ROOT/preprocess"

# Input FIF root on shared storage.
INPUT_ROOT_ON_CLUSTER="$HOME/scratch/Burnaby"

# Destination for the small test output.
FINAL_OUTPUT_ROOT_ON_CLUSTER="$SCRATCH/EEG_Features/env_smoke_tests"

# Python environment activation.
USE_VENV=1
VENV_PATH="$HOME/venvs/eeg"

CHECKPOINT_PATH="$PREPROCESS_DIR/LaBraM/checkpoints/labram-base.pth"
STAGE_INPUT_TO_SCRATCH=0
# -----------------------------------------

module purge
module load StdEnv/2023 python/3.11 arrow

if [[ "$USE_VENV" -eq 1 ]]; then
  source "$VENV_PATH/bin/activate"
fi

pip install -r requirements.txt --no-index 

cd "$PREPROCESS_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Checking configured paths..."
echo "INPUT_ROOT_ON_CLUSTER=$INPUT_ROOT_ON_CLUSTER"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "FINAL_OUTPUT_ROOT_ON_CLUSTER=$FINAL_OUTPUT_ROOT_ON_CLUSTER"

[[ -d "$INPUT_ROOT_ON_CLUSTER" ]] || { echo "Missing input dir: $INPUT_ROOT_ON_CLUSTER" >&2; exit 1; }
[[ -f "$CHECKPOINT_PATH" ]] || { echo "Missing checkpoint: $CHECKPOINT_PATH" >&2; exit 1; }
mkdir -p "$FINAL_OUTPUT_ROOT_ON_CLUSTER"


echo "First input files:"
FIRST_FIF=$(find "$INPUT_ROOT_ON_CLUSTER" -type f -name "*.fif" -print -quit)
test -n "$FIRST_FIF"
find "$INPUT_ROOT_ON_CLUSTER" -type f -name "*.fif" | awk 'NR <= 5 { print }'

echo "Checking Python imports..."
python - <<'PY'
import h5py
import mne
import numpy
import pyarrow
import scipy
import timm
import torch
from einops import rearrange

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("mne:", mne.__version__)
PY

RUN_INPUT_ROOT="$INPUT_ROOT_ON_CLUSTER"
if [[ "$STAGE_INPUT_TO_SCRATCH" -eq 1 ]]; then
  RUN_INPUT_ROOT="$SLURM_TMPDIR/eeg_input_${SLURM_JOB_ID}"
  mkdir -p "$RUN_INPUT_ROOT"
  echo "Staging input to local scratch: $RUN_INPUT_ROOT"
  time rsync -a --info=progress2 "$INPUT_ROOT_ON_CLUSTER/" "$RUN_INPUT_ROOT/"
fi

RUN_OUTPUT_ROOT="$SLURM_TMPDIR/eeg_env_smoke_out_${SLURM_JOB_ID}"
mkdir -p "$RUN_OUTPUT_ROOT"

echo "Running CPU feature extraction smoke test..."
time python extract_features_variable_length.py \
  --data_dir "$RUN_INPUT_ROOT" \
  --output_dir "$RUN_OUTPUT_ROOT" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --device cpu \
  --duration 4 \
  --overlap 0 \
  --write_buffer_size 4 \
  --rows_per_shard 4 \
  --compression lzf \
  --debug

echo "Output files on node-local scratch:"
find "$RUN_OUTPUT_ROOT" -maxdepth 2 -type f -print

STAMP=$(date +"%Y%m%d_%H%M%S")
DEST_DIR="$FINAL_OUTPUT_ROOT_ON_CLUSTER/run_${STAMP}_job${SLURM_JOB_ID}"
mkdir -p "$DEST_DIR"

echo "Copying smoke-test output to: $DEST_DIR"
rsync -a --info=progress2 "$RUN_OUTPUT_ROOT/" "$DEST_DIR/"

echo "Copied output files:"
find "$DEST_DIR" -maxdepth 2 -type f -print

echo "Smoke test completed successfully."