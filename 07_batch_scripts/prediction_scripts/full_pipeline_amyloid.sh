#!/bin/bash
#SBATCH --job-name=amyloid_full_pipe
#SBATCH --output=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/full_pipeline_amyloid_%A_%a.log
#SBATCH --error=/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/full_pipeline_amyloid_%A_%a.err
#SBATCH --partition=gpu-n12-85g-1x-a100-40g
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=99:00:00

# ============================================================================
# Full Pipeline for AMYLOID: Pipeline1 + Prediction + Pipeline2 + Cleanup
# ============================================================================
# Model: EfficientNet-B4 UNet (epoch 191, val_loss=0.10)
# Keeps only: heat_map_0.1_res10.nii (~1GB per slide)
# Deletes: output/, mask/, seg_tiles/, TAU_seg_tiles/, other hm files (~143GB)
# ============================================================================

WSI_DIR_ROOT="/fslustre/qhs/ext_chen_yuheng_mayo_edu/RO1_GCP/Pipeline_merged"
STAIN_TYPE="RO1_Amyloid"
DIR_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/full_pipeline_amyloid_dir_list.txt"
FAILED_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/full_pipeline_amyloid_failed.txt"
SUCCESS_LIST_FILE="/fslustre/qhs/ext_chen_yuheng_mayo_edu/script/out/full_pipeline_amyloid_success.txt"
PIPELINE_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline_config.txt"
PREDICTION_CONFIG="/fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/data/setup_files/setup-predict_Amyloid_efficientnet_v2.json"

# ============================================================================
# JOB PREPARATION
# ============================================================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "============================================================"
    echo "Full Pipeline for AMYLOID - Job Preparation"
    echo "============================================================"
    
    # Find all slide directories
    find "$WSI_DIR_ROOT/$STAIN_TYPE" -type d -name "*_files" 2>/dev/null | \
        sed 's|_files$||' | sort -u > "$DIR_LIST_FILE"
    
    NUM_DIRS=$(wc -l < "$DIR_LIST_FILE")
    echo "Found $NUM_DIRS Amyloid slides to process."
    
    # Initialize tracking files
    echo "# Failed - Full Pipeline Amyloid" > "$FAILED_LIST_FILE"
    echo "# Generated: $(date)" >> "$FAILED_LIST_FILE"
    echo "" >> "$FAILED_LIST_FILE"
    
    echo "# Success - Full Pipeline Amyloid" > "$SUCCESS_LIST_FILE"
    echo "# Generated: $(date)" >> "$SUCCESS_LIST_FILE"
    echo "" >> "$SUCCESS_LIST_FILE"

    if [ "$NUM_DIRS" -gt 0 ]; then
        echo "Submitting $NUM_DIRS tasks..."
        sbatch --array=1-$NUM_DIRS "$0"
    fi
    exit 0
fi

# ============================================================================
# ARRAY TASK EXECUTION
# ============================================================================
echo "============================================================"
echo "Full Pipeline AMYLOID - Task $SLURM_ARRAY_TASK_ID | $(date)"
echo "============================================================"

SLIDE_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR_LIST_FILE")
SLIDE_NAME=$(basename "$SLIDE_DIR")
FILES_DIR="${SLIDE_DIR}_files"

echo "Slide: $SLIDE_NAME"
echo "Path: $SLIDE_DIR"

PIPELINE1_SUCCESS=0
PREDICTION_SUCCESS=0
PIPELINE2_SUCCESS=0

# --- STAGE 1: Pipeline 1 ---
echo ""
echo "=== STAGE 1: Pipeline 1 (Tiling & Masks) ==="
source /home/ext_chen_yuheng_mayo_edu/miniconda3/etc/profile.d/conda.sh
conda activate gdaltest
export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau

if [ -d "$FILES_DIR/heatmap/seg_tiles" ] && [ "$(ls -A $FILES_DIR/heatmap/seg_tiles 2>/dev/null)" ]; then
    echo "Pipeline 1 already done. Skipping..."
    PIPELINE1_SUCCESS=1
else
    bash /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/scripts/run_pipeline_full.sh \
        "$SLIDE_DIR" "$PIPELINE_CONFIG"
    [ $? -eq 0 ] && [ -d "$FILES_DIR/heatmap/seg_tiles" ] && PIPELINE1_SUCCESS=1
fi
echo "Pipeline 1: $([ $PIPELINE1_SUCCESS -eq 1 ] && echo 'SUCCESS' || echo 'FAILED')"

# --- STAGE 2: Prediction ---
if [ $PIPELINE1_SUCCESS -eq 1 ]; then
    echo ""
    echo "=== STAGE 2: EfficientNet Prediction ==="
    conda activate rhizonet
    export WANDB_MODE=disabled
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/usr/local/biotools/cuda/12.1/lib64
    export PATH=/usr/local/biotools/cuda/12.1/bin:${PATH}
    
    rm -rf "$FILES_DIR/heatmap/TAU_seg_tiles" 2>/dev/null
    
    python /fslustre/qhs/ext_chen_yuheng_mayo_edu/rhizonet/rhizonet/predict_iron_effnet.py \
        --config_file "$PREDICTION_CONFIG" --wsi_dir "$FILES_DIR"
    
    [ $? -eq 0 ] && [ -d "$FILES_DIR/heatmap/TAU_seg_tiles" ] && PREDICTION_SUCCESS=1
    echo "Prediction: $([ $PREDICTION_SUCCESS -eq 1 ] && echo 'SUCCESS' || echo 'FAILED')"
fi

# --- STAGE 3: Pipeline 2 ---
if [ $PREDICTION_SUCCESS -eq 1 ]; then
    echo ""
    echo "=== STAGE 3: Pipeline 2 (Heatmap) ==="
    conda activate gdaltest
    export PYTHONPATH=$PYTHONPATH:/fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau
    
    find "$FILES_DIR/heatmap" -maxdepth 1 -type d -name "hm_map_*" -exec rm -rf {} \; 2>/dev/null
    
    python /fslustre/qhs/ext_chen_yuheng_mayo_edu/high-res-3D-tau/pipeline/run_pipeline_part2.py \
        "$FILES_DIR" "$PIPELINE_CONFIG"
    
    [ $? -eq 0 ] && [ -f "$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii" ] && PIPELINE2_SUCCESS=1
    echo "Pipeline 2: $([ $PIPELINE2_SUCCESS -eq 1 ] && echo 'SUCCESS' || echo 'FAILED')"
fi

# --- STAGE 4: Cleanup ---
if [ $PIPELINE2_SUCCESS -eq 1 ]; then
    echo ""
    echo "=== STAGE 4: Cleanup ==="
    SPACE_BEFORE=$(du -s "$FILES_DIR" 2>/dev/null | cut -f1)
    
    FINAL_HEATMAP="$FILES_DIR/heatmap/hm_map_0.1/heat_map_0.1_res10.nii"
    TEMP_HEATMAP="/tmp/heatmap_amyloid_${SLURM_ARRAY_TASK_ID}.nii"
    
    cp "$FINAL_HEATMAP" "$TEMP_HEATMAP"
    
    rm -rf "$FILES_DIR/output"
    rm -rf "$FILES_DIR/mask"
    rm -rf "$FILES_DIR/heatmap/seg_tiles"
    rm -rf "$FILES_DIR/heatmap/TAU_seg_tiles"
    rm -rf "$FILES_DIR/heatmap/hm_map_0.1"
    
    mkdir -p "$FILES_DIR/heatmap/hm_map_0.1"
    mv "$TEMP_HEATMAP" "$FINAL_HEATMAP"
    
    SPACE_AFTER=$(du -s "$FILES_DIR" 2>/dev/null | cut -f1)
    echo "Space saved: $(numfmt --to=iec $(((SPACE_BEFORE - SPACE_AFTER) * 1024)))"
fi

# --- Final Status ---
echo ""
echo "============================================================"
if [ $PIPELINE2_SUCCESS -eq 1 ]; then
    echo "$SLIDE_DIR" >> "$SUCCESS_LIST_FILE"
    echo "✓ SUCCESS - $SLIDE_NAME"
else
    echo "$SLIDE_DIR | P1=$PIPELINE1_SUCCESS P2=$PREDICTION_SUCCESS P3=$PIPELINE2_SUCCESS" >> "$FAILED_LIST_FILE"
    echo "✗ FAILED - $SLIDE_NAME"
fi
echo "End: $(date)"
