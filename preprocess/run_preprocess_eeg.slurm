#!/bin/bash
#SBATCH --job-name=eeg-shard-preprocess
#SBATCH --account=def-vvakorin
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# -------- User config (edit these) --------
# Absolute path to your repo on the cluster.
REPO_ROOT="$SCRATCH/FHA-EEG-"
PREPROCESS_DIR="$REPO_ROOT/preprocess"

# Input FIF root on shared storage (project/scratch/nearline)
#INPUT_ROOT_ON_CLUSTER="$HOME/projects/rpp-doesburg/databases/eeg_fha/preprocessed/001_a01_01"

##debug path
INPUT_ROOT_ON_CLUSTER="$HOME/scratch/Burnaby"

# Final destination for outputs on shared storage
#FINAL_OUTPUT_ROOT_ON_CLUSTER="$HOME/projects/rpp-doesburg/zhenyu/FHA/Resting"

##debug path
FINAL_OUTPUT_ROOT_ON_CLUSTER="$HOME/scratch/EEG_Features" 

# Python environment activation (pick one method)
# 1) module-only:
USE_VENV=1
# 2) module + virtualenv:
VENV_PATH="$HOME/venvs/eeg"

# Optional preprocessing settings
ROWS_PER_SHARD=400000  # Each row is a segment. Adjust based on output file size.
WRITE_BUFFER_SIZE=1000 # 1000 segment rows are buffered before writing to disk.
COMPRESSION="gzip"  # gzip|lzf
CHECKPOINT_PATH="$PREPROCESS_DIR/LaBraM/checkpoints/labram-base.pth"
STAGE_INPUT_TO_SCRATCH=1  # 1=yes (recommended), 0=no
# -----------------------------------------

module purge
module load StdEnv/2023 python/3.11 arrow

if [[ "$USE_VENV" -eq 1 ]]; then
  source "$VENV_PATH/bin/activate"
fi

cd "$PREPROCESS_DIR"

# Use node-local scratch for heavy reads/writes, then copy back to shared storage.
RUN_INPUT_ROOT="$INPUT_ROOT_ON_CLUSTER"
if [[ "$STAGE_INPUT_TO_SCRATCH" -eq 1 ]]; then
  RUN_INPUT_ROOT="$SLURM_TMPDIR/eeg_input_${SLURM_JOB_ID}"
  mkdir -p "$RUN_INPUT_ROOT"
  echo "Staging input to local scratch: $RUN_INPUT_ROOT"
  time rsync -a --info=progress2 "$INPUT_ROOT_ON_CLUSTER/" "$RUN_INPUT_ROOT/"
fi

RUN_OUTPUT_ROOT="$SLURM_TMPDIR/eeg_preprocess_out_${SLURM_JOB_ID}"
mkdir -p "$RUN_OUTPUT_ROOT"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Input root (source): $INPUT_ROOT_ON_CLUSTER"
echo "Input root (run): $RUN_INPUT_ROOT"
echo "Scratch output: $RUN_OUTPUT_ROOT"
echo "Final output: $FINAL_OUTPUT_ROOT_ON_CLUSTER"
echo "Checkpoint: $CHECKPOINT_PATH"

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

# Copy results back to shared storage atomically-ish (new timestamped folder).
STAMP=$(date +"%Y%m%d_%H%M%S")
DEST_DIR="$FINAL_OUTPUT_ROOT_ON_CLUSTER/run_${STAMP}_job${SLURM_JOB_ID}"
mkdir -p "$DEST_DIR"

#----------Debug------------------
echo "PWD: $(pwd)"
echo "REPO_ROOT: $REPO_ROOT"
echo "PREPROCESS_DIR: $PREPROCESS_DIR"
echo "INPUT_ROOT_ON_CLUSTER: $INPUT_ROOT_ON_CLUSTER"
ls "$INPUT_ROOT_ON_CLUSTER" | head
#-----------------------------------


rsync -a --info=progress2 "$RUN_OUTPUT_ROOT/" "$DEST_DIR/"

echo "Completed. Results at: $DEST_DIR"
