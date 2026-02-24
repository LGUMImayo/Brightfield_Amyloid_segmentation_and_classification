# Brightfield Amyloid Segmentation & Classification Pipeline

## Pipeline Overview

This pipeline processes brightfield-stained Amyloid whole-slide images (WSI) through multiple stages:

1. **Tiling & Masking** — Tile the WSI and create tissue masks
2. **Amyloid Segmentation** — EfficientNet-B4 U-Net (rhizonet) to segment amyloid plaques
3. **Heatmap Generation** — Convert segmentation tiles into quantitative heatmaps (NIfTI)
4. **Gray/White Matter Segmentation** — Matt's model to isolate gray matter regions
5. **Plaque Classification** — Matt's Faster R-CNN (4-class: CGP, CCP, CAA1, CAA2)
6. **Statistics Aggregation** — Combine per-slide stats into analysis-ready CSVs

---

## Directory Structure

```
UCSF_AI_DAY_AMYLOID/
├── README.md
├── matt_code.yml                          # Conda env for Matt's models
├── matt_code_final.yml                    # Conda env (final version)
│
├── 01_tiling_and_masking/                 # Stage 1: WSI → Tiles + Masks
│   ├── run_pipeline_full.sh               #   Entry point (calls run_pipeline.py)
│   ├── run_pipeline_full2.sh              #   Alternative entry point
│   ├── pipeline_config.txt                #   Config (resolution, mask params, heatmap params)
│   └── pipeline/
│       ├── run_pipeline.py                #   Main pipeline runner
│       ├── PipelineRunner.py              #   Pipeline orchestrator
│       ├── ImageTiler.py                  #   Tiles full-res images
│       ├── MaskTiler.py                   #   Tiles tissue masks
│       └── TileMasker.py                  #   Applies masks to tiles → seg_tiles/
│
├── 02_amyloid_segmentation/               # Stage 2: Amyloid Segmentation (rhizonet)
│   ├── rhizonet_core/
│   │   ├── predict_iron_effnet.py         #   Main prediction (EfficientNet, used for all stains)
│   │   ├── predict_gpu.py                 #   GPU prediction (original rhizonet U-Net)
│   │   ├── predict_gpu_focal_amyloid.py   #   Focal-loss variant + bubble artifact removal
│   │   ├── predict_amyloid_thin_slides.py #   Thin-slide variant (uses GM mask for ROI)
│   │   ├── predict_manual_amyloid.py      #   Manual/single-slide prediction
│   │   ├── unet2D.py                      #   U-Net model architecture
│   │   ├── unet2D_focal_amyloid.py        #   Focal-loss U-Net variant
│   │   ├── utils.py                       #   Dataset/dataloader utilities
│   │   ├── metrics.py                     #   Evaluation metrics
│   │   ├── postprocessing.py              #   Post-processing (morphological ops)
│   │   └── __init__.py
│   └── configs/
│       ├── setup-predict_Amyloid_efficientnet_v2.json    # ← PRIMARY: EfficientNet-B4, epoch 191
│       ├── setup-predict_Amyloid.json                    # Original rhizonet U-Net
│       ├── setup-predict_Amyloid_priority.json           # Priority batch
│       ├── setup-predict_Amyloid_priority_last.json      # Priority last checkpoint
│       ├── setup-predict_Amyloid_focal_amyloid_pipeline.json   # Focal model (Pipeline)
│       └── setup-predict_Amyloid_focal_amyloid_pipeline2.json  # Focal model (Pipeline2)
│
├── 03_heatmap_generation/                 # Stage 3: Seg tiles → Heatmap NIfTI
│   ├── run_pipeline_part2.py              #   Main heatmap script
│   ├── HeatmapCreator.py                 #   Creates quantitative heatmaps
│   ├── ColormapCreator.py                #   Creates color overlay maps
│   └── export_heatmap_metadata.py        #   Exports heatmap metadata
│
├── 04_gray_white_segmentation/            # Matt's Gray/White Matter Model
│   └── src/
│       ├── inference_test.py              #   Main inference script
│       ├── inference.py                   #   Base inference
│       ├── inference_utils.py             #   Inference utilities
│       ├── train_gray_white_5_fold.py     #   5-fold training
│       ├── calc_auc_from_saved.py         #   AUC calculation
│       ├── generate_roc_pr_curve.py       #   ROC/PR curves
│       ├── visualize_model.py             #   Model visualization
│       └── visualize_qualitative_and_quantitative.py
│
├── 05_plaque_classification/              # Matt's Plaque Classification (Faster R-CNN)
│   ├── src/
│   │   ├── run_on_test_set_validation_Amyloid.py   # ← MAIN: Whole-slide plaque inference
│   │   ├── run_on_test_set_validation_BBM.py       #   BBM variant
│   │   ├── tangle_ai_bbox_utils.py                 #   Model builder + dataset class
│   │   ├── train_plaque_ai_ccp.py                  #   Training (saves plaque_ai_final.pt)
│   │   ├── generate_plaque_eval.py                 #   Evaluation metrics
│   │   ├── generate_plaque_report_v2.py            #   Report generation
│   │   ├── visualize_plaque_qualitative.py         #   Qualitative visualizations
│   │   └── download_weights.py                     #   Weight downloader
│   └── tile_based_inference/
│       ├── run_inference_on_tiles_9117.py           #   Tile-based plaque detection
│       └── run_9117_amyloid_inference_test.sh       #   SLURM launcher for tile inference
│
├── 06_stats_aggregation/                  # Combining Results
│   ├── combine_amyloid_final.py           #   Concatenate *_stats.csv from segmentation
│   ├── combine_classification_stats.py    #   Summarize plaque detection per-slide
│   ├── combine_gray_white_stats.py        #   Combine gray/white matter stats
│   ├── merge_amyloid_stats.py             #   Merge with neuropath master CSV
│   └── AAIC_abstract_statistic/           #   AAIC abstract analysis scripts + CSVs
│
├── 07_batch_scripts/                      # SLURM Job Submission Scripts
│   ├── slurm_launchers/
│   │   ├── run_pipeline_part1_RO1_all.sh            # Stage 1 for all RO1 slides
│   │   ├── run_pipeline_part2_RO1_Amyloid_all.sh    # Stage 3 for all Amyloid
│   │   ├── run_pipeline_part2_RO1_Amyloid_9117.sh   # Stage 3 for case 9117
│   │   ├── run_inference_gray_white_matter.sh        # Gray/white matter inference
│   │   └── run_inference_Amyloid_plaque.sh           # Plaque classification
│   ├── single_slide_scripts/
│   │   ├── single_slide_stage1_amyloid.sh   # Stage 1 per slide (tiling)
│   │   ├── single_slide_stage2_amyloid_cpu.sh # Stage 2 per slide (segmentation, CPU)
│   │   ├── single_slide_stage3_amyloid.sh   # Stage 3 per slide (heatmap, GPU)
│   │   └── single_slide_stage3_amyloid_cpu.sh # Stage 3 per slide (heatmap, CPU)
│   ├── parallel_launchers/
│   │   ├── launch_amyloid_stage1_parallel.sh  # Submit all Stage 1 in parallel
│   │   ├── launch_amyloid_stage2_parallel_cpu.sh
│   │   ├── launch_amyloid_stage3_parallel.sh
│   │   ├── launch_amyloid_stage3_parallel_cpu.sh
│   │   └── launch_all_amyloid_reprocess.sh
│   ├── mayo_brain_bank_runs/                     # ← ACTUAL SCRIPTS RUN ON 183 MAYO BBM SLIDES
│   │   ├── submit_all_thin_slides_inference_in_slurm.sh # Amyloid seg (production)
│   │   ├── run_amyloid_thin_inference.sh        #   Single-slide test
│   │   ├── run_inference_gray_white_matter.sh   #   Gray/white inference
│   │   ├── run_inference_Amyloid_plaque.sh      #   Plaque classification
│   │   ├── run_combine_csv.sh                   #   Combine CSVs
│   │   ├── run_combine_new_stats.sh             #   Combine stats
│   │   ├── submit_all_slides.py                 #   Python job submitter
│   │   └── prepare_ground_truth_tiles.py        #   GT tile extraction
│   └── prediction_scripts/
│       ├── full_pipeline_amyloid.sh           # ← ALL-IN-ONE: Stage 1→2→3 + cleanup (RO1_GCP)
│       ├── submit_full_pipeline_amyloid.sh    # Submit full pipeline
│       ├── stage1_pipeline1_amyloid.sh
│       ├── stage2_prediction_amyloid.sh
│       ├── stage3_pipeline2_cleanup_amyloid.sh
│       ├── prediction_amyloid_pipeline_all.sh
│       ├── prediction_amyloid_pipeline_all_focal_amyloid.sh
│       ├── prediction_amyloid_pipeline2_all_focal_amyloid.sh
│       ├── prediction_amyloid_effnet_v2_all.sh
│       ├── pipeline2_amyloid_effnet_v2_all.sh
│       └── prediction_amyloid_rhizonet_9117_last.sh
│
└── 08_training_scripts/                   # Training Code
    ├── train.py                           #   Base rhizonet training
    ├── train_focal_amyloid.py             #   Focal-loss amyloid training
    ├── training_amyloid_rhizonet_data_with_edges.sh
    ├── run_generate_weights_and_train_Amyloid_efficientnet.sh
    ├── run_generate_attention_maps_Amyloid.sh
    └── run_prepare_Amyloid_cleanup_and_prep.sh
```

---

## Conda Environments

| Environment | Used For |
|-------------|----------|
| `gdaltest`  | Stage 1 (tiling/masking) and Stage 3 (heatmap) |
| `rhizonet`  | Stage 2 (amyloid segmentation with EfficientNet) |
| `matt_code` / `matt_code_final` | Stage 4 (gray/white) and Stage 5 (plaque classification) |

---

## Key Model Checkpoints (not copied — large files)

| Model | Path |
|-------|------|
| Amyloid EfficientNet-B4 U-Net | `RO1_CNN/RO1_Amyloid_testing/log_efficientnet_v2/checkpoint-epoch=191-val_loss=0.10.ckpt` |
| Amyloid U-Net (original) | `RO1_CNN/RO1_Amyloid_testing/log_new_data_with_edges/last.ckpt` |
| Plaque Faster R-CNN | `RO1_GCP/Matt_codes/s311590_plaque_ai/s311590_plaque_ai/plaque_ai/models/plaque_ai_final.pt` |
| Gray/White Segmentation | `RO1_GCP/Matt_codes/s311590_gray_white/gray_white_segmentation/models/` |

---

## Quick Start — Run Full Pipeline on a Single Slide

```bash
# All-in-one (Stage 1 → 2 → 3 + cleanup):
sbatch 07_batch_scripts/prediction_scripts/full_pipeline_amyloid.sh

# Or stage by stage:
# Stage 1: Tiling
sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> \
    07_batch_scripts/single_slide_scripts/single_slide_stage1_amyloid.sh

# Stage 2: Amyloid Segmentation (EfficientNet)
sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> \
    07_batch_scripts/single_slide_scripts/single_slide_stage2_amyloid_cpu.sh

# Stage 3: Heatmap
sbatch --export=SLIDE_NAME=<name>,SLIDE_DIR=<path> \
    07_batch_scripts/single_slide_scripts/single_slide_stage3_amyloid.sh

# Stage 4: Gray/White + Plaque Classification
sbatch 07_batch_scripts/slurm_launchers/run_inference_gray_white_matter.sh
sbatch 07_batch_scripts/slurm_launchers/run_inference_Amyloid_plaque.sh
```

---

## SLURM Log Cross-Reference — Mayo Brain Bank Amyloid Runs

The pipeline was run on **183 Mayo Brain Bank brightfield Amyloid slides** (NOT RO1_GCP data).
Data was at `Amyloid_Slides/tiff/` (now moved to GCP storage).

### What was actually run (verified from SLURM logs):

| Step | SLURM Job IDs | Script | Python Code | Model | Env |
|------|--------------|--------|-------------|-------|-----|
| **Gray/White Segmentation** | 252800 (launcher) → 252801 (183 tasks) | `run_inference_gray_white_matter.sh` | `inference_test.py` | Matt's gray/white model | `matt_code` |
| **Amyloid Segmentation** | 569797 (launcher) → 569811 (183 tasks) | `submit_all_thin_slides_inference_in_slurm.sh` | `predict_amyloid_thin_slides.py` | `last.ckpt` (original U-Net, NOT EfficientNet) | `rhizonet` |
| **Plaque Classification** | 507291 (launcher) → 507292 (183 tasks) | `run_inference_Amyloid_plaque.sh` | `run_on_test_set_validation_Amyloid.py` | `plaque_ai_final.pt` (Faster R-CNN) | `matt_code` |

### Key parameters used for amyloid segmentation (from logs):
```
--resolution 0.2827
--model_input_size 128
--gm_threshold 0.9
--blur_strength 0
--min_grain_size 100
--model_path .../RO1_CNN/RO1_Amyloid_testing/log_new_data_with_edges/last.ckpt
--mask_root_dir .../Amyloid_Slides/prediction_priority_threshold_wmw_1.5_gmt_0.2_wmt_0.2_clean_21_best_parameter
--output_dir .../Amyloid_Slides/amyloid_predictions_final_v1
```

### Important notes — differences from RO1_GCP pipeline:
1. **Different pipeline flow**: Mayo BBM used `predict_amyloid_thin_slides.py` (direct TIFF → gray matter mask → segmentation). RO1_GCP used `predict_iron_effnet.py` on pre-tiled pipeline data.
2. **Different model**: Mayo BBM used the **original U-Net** (`last.ckpt`), not the EfficientNet-B4 (`epoch=191`).
3. **Gray matter masking**: Mayo BBM used GM probability masks from Matt's `inference_test.py` output to restrict ROI before amyloid segmentation.
4. **Matt_codes path**: Scripts referenced `/fslustre/.../Matt_codes/` (root level) — now moved to `RO1_GCP/Matt_codes/`. Content is the same.
5. **Data path**: `Amyloid_Slides/` no longer on local filesystem (moved to GCP).

### Output locations (from logs):
- Amyloid seg masks + stats: `Amyloid_Slides/amyloid_predictions_final_v1/`
- Gray/white masks: `Amyloid_Slides/prediction_priority_threshold_.../*.npy`
- Plaque detections: Output of `run_on_test_set_validation_Amyloid.py` per TIFF

### Mayo Brain Bank batch scripts (in `07_batch_scripts/mayo_brain_bank_runs/`):
| File | Purpose |
|------|---------|
| `submit_all_thin_slides_inference_in_slurm.sh` | **Production run** — submits 183 amyloid seg jobs |
| `run_amyloid_thin_inference.sh` | Test run on single slide |
| `run_inference_gray_white_matter.sh` | Gray/white matter inference (183 slides) |
| `run_inference_Amyloid_plaque.sh` | Plaque classification (183 slides) |
| `run_combine_csv.sh` | Combine result CSVs |
| `run_combine_new_stats.sh` | Combine gray/white + amyloid stats |
| `submit_all_slides.py` | Alternative Python-based job submitter |
| `prepare_ground_truth_tiles.py` | Ground truth validation tile extraction |

---

## Original Source Locations

| Component | Original Path |
|-----------|---------------|
| Pipeline core | `high-res-3D-tau/pipeline/` and `high-res-3D-tau/scripts/` |
| Rhizonet prediction | `rhizonet/rhizonet/` |
| Prediction configs | `rhizonet/data/setup_files/` |
| Batch scripts | `rhizonet/batch_scripts/` |
| SLURM scripts | `high-res-3D-tau/slurm_scripts/` |
| Matt's gray/white | `RO1_GCP/Matt_codes/s311590_gray_white/` (moved from root `Matt_codes/`) |
| Matt's plaque AI | `RO1_GCP/Matt_codes/s311590_plaque_ai/` (moved from root `Matt_codes/`) |
| Stats aggregation | `Amyloid_code/` |
| Tile-based plaque | `script/Plaque_classification_with_amyloid_segmentation_in_rhizonet/` |
| Mayo BBM batch scripts | `RO1_GCP/Amyloid_Slides/batch_script_for_thin_slides_Amyloid_segmentation/` |
