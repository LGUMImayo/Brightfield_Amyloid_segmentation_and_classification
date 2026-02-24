#!/bin/bash
# ============================================================================
# Master script to submit the full Amyloid pipeline as 3 dependent jobs:
#   1. Pipeline 1 (CPU, high memory) -> huge-n128-512g
#   2. Prediction (GPU) -> gpu-n12-85g-1x-a100-40g
#   3. Pipeline 2 + Cleanup (CPU) -> gpu-n12-85g-1x-a100-40g (or CPU node)
# ============================================================================

STAIN="Amyloid"
BASE_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged/RO1_${STAIN}/Cases"
BATCH_DIR="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/batch_scripts"

# Submit Pipeline 1 job (high-memory CPU)
JOB1=$(sbatch --parsable ${BATCH_DIR}/stage1_pipeline1_amyloid.sh)
echo "Submitted Stage 1 (Pipeline 1): Job $JOB1"

# Submit Prediction job (GPU) - depends on Pipeline 1
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} ${BATCH_DIR}/stage2_prediction_amyloid.sh)
echo "Submitted Stage 2 (Prediction): Job $JOB2 (depends on $JOB1)"

# Submit Pipeline 2 + Cleanup job - depends on Prediction
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} ${BATCH_DIR}/stage3_pipeline2_cleanup_amyloid.sh)
echo "Submitted Stage 3 (Pipeline2+Cleanup): Job $JOB3 (depends on $JOB2)"

echo ""
echo "Full pipeline submitted as job chain: $JOB1 -> $JOB2 -> $JOB3"
echo "Monitor with: squeue -u \$USER"
