#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# One-shot optimized AddH -> AddH-out training + strict-blind prediction pipeline
# Goal:
#   1) Efficiently train / reuse graph-embedding ensembles.
#   2) Train / reuse selected multi-view architecture grid.
#   3) Build family-diverse strict-blind ensembles compatible with graph ensemble + multi-view.
#   4) Build a final strict-blind strategy ensemble without using addH-out labels.
#
# Strict blind rule:
#   addH-out h_ads_excel / target / target_computed are NOT used for training,
#   model selection, weights, calibration, or final prediction. They are only used
#   if RUN_AUDIT=1, and then only in separate post-hoc audit folders.
#
# Run from: /data/home/terminator/RL/multi-view
# ============================================================

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"

# Thread control for classical regressors
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true

# Reuse/resume controls
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Stages
RUN_GRAPH_GRID="${RUN_GRAPH_GRID:-1}"
RUN_MULTIVIEW_GRID="${RUN_MULTIVIEW_GRID:-1}"
RUN_STRICT_BLIND="${RUN_STRICT_BLIND:-1}"
RUN_STRATEGY_ENSEMBLE="${RUN_STRATEGY_ENSEMBLE:-1}"
RUN_AUDIT="${RUN_AUDIT:-0}"   # post-hoc only; not strict-blind selection

# Experiment scale
SEEDS="${SEEDS:-42,52,62,72,82,92,102,112,122,132}"

# multi-view grid can be expensive.
# all       = run all 53 model-variant experiments in run_addh_model_experiment_grid_v2_full.sh
# targeted  = run high-value graph_only + concat/residual/gated subset
# off       = skip multi-view grid
MULTIVIEW_SCOPE="${MULTIVIEW_SCOPE:-targeted}"
MULTIVIEW_GRID_ROOT="${MULTIVIEW_GRID_ROOT:-outputs_addh_modelgrid_v2_full}"
GRAPH_GRID_ROOT="${GRAPH_GRID_ROOT:-outputs_addh_graph_ensemble_refine_v2}"

# Strict-blind settings
TOP_K="${TOP_K:-16}"
MIN_COVERAGE="${MIN_COVERAGE:-0.90}"
WEIGHT_MODE="${WEIGHT_MODE:-soft_inverse_rmse}"

# Required scripts
REQ=(
  "04_make_multiview_data_cv_multimodal.py"
  "13_train_graph_embedding_ensemble_v2.py"
  "14_summarize_graph_ensemble_addhout_v2.py"
  "run_graph_embedding_refine_addh_full_once.sh"
  "16_make_strict_blind_addhout_ensemble_v2.py"
  "run_strict_blind_addhout_prediction_diverse.sh"
  "17_build_strict_blind_strategy_ensemble.py"
)
if [[ "$RUN_MULTIVIEW_GRID" == "1" && "$MULTIVIEW_SCOPE" != "off" ]]; then
  REQ+=("run_addh_model_experiment_grid_v2_full.sh" "05_run_multiview_cv_ensemble_multimodal_staged_aligned_blend3_fixclip_v2.py" "regress_run_multimodal_staged_aligned_v2.py" "regress_predict_multimodal_aligned_v2.py")
fi
for f in "${REQ[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    exit 1
  fi
done

mkdir -p logs_addh_strict_blind_optimized

echo "============================================================"
echo "[CONFIG] ROOT=$ROOT"
echo "[CONFIG] PY_MM=$PY_MM"
echo "[CONFIG] GPU_ID=$GPU_ID DEVICE=$DEVICE"
echo "[CONFIG] SEEDS=$SEEDS"
echo "[CONFIG] SKIP_COMPLETED=$SKIP_COMPLETED FORCE_RERUN=$FORCE_RERUN"
echo "[CONFIG] RUN_GRAPH_GRID=$RUN_GRAPH_GRID"
echo "[CONFIG] RUN_MULTIVIEW_GRID=$RUN_MULTIVIEW_GRID MULTIVIEW_SCOPE=$MULTIVIEW_SCOPE"
echo "[CONFIG] RUN_STRICT_BLIND=$RUN_STRICT_BLIND RUN_STRATEGY_ENSEMBLE=$RUN_STRATEGY_ENSEMBLE RUN_AUDIT=$RUN_AUDIT"
echo "============================================================"

# ----------------------------
# Stage 1: graph embedding ensemble grid
# ----------------------------
if [[ "$RUN_GRAPH_GRID" == "1" ]]; then
  echo "[STAGE 1] Graph embedding ensemble refine grid"
  # This script has its own SKIP_COMPLETED logic.
  GRID_ROOT="$GRAPH_GRID_ROOT" \
  SEEDS="$SEEDS" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  FORCE_RERUN="$FORCE_RERUN" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  "$PY_MM" -m py_compile 13_train_graph_embedding_ensemble_v2.py

  GRID_ROOT="$GRAPH_GRID_ROOT" \
  SEEDS="$SEEDS" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  FORCE_RERUN="$FORCE_RERUN" \
  bash run_graph_embedding_refine_addh_full_once.sh \
    > logs_addh_strict_blind_optimized/stage1_graph_grid.log 2>&1
  echo "[DONE] Stage 1 graph grid"
else
  echo "[SKIP] Stage 1 graph grid"
fi

# ----------------------------
# Stage 2: multi-view architecture grid
# ----------------------------
if [[ "$RUN_MULTIVIEW_GRID" == "1" && "$MULTIVIEW_SCOPE" != "off" ]]; then
  echo "[STAGE 2] Multi-view model architecture grid: $MULTIVIEW_SCOPE"

  if [[ "$MULTIVIEW_SCOPE" == "all" ]]; then
    MV_EXPS="all"
  else
    # High-value, efficient subset. Focus on graph_only because graph is dominant,
    # plus a few concat/gated/residual controls for model-combination diversity.
    MV_EXPS="m13_t10_ssl_graphonly_u2_wp05,m16_t9_ssl_graphonly_u2_wp05,m17_t8_ssl_graphonly_u2_wp05,m18_t7_ssl_graphonly_u2_wp05,m19_t10_ssl_graphonly_u2_wp00,m20_t10_ssl_graphonly_u2_wp02,m21_t10_ssl_graphonly_u2_wp08,m22_t10_ssl_graphonly_u2_wp10,m23_t10_ssl_graphonly_u2_drop00,m24_t10_ssl_graphonly_u2_drop05,m25_t10_ssl_graphonly_u2_drop15,m26_t10_ssl_graphonly_u2_drop20,m27_t10_ssl_graphonly_u2_drop30,m28_t10_ssl_graphonly_u2_noise002,m29_t10_ssl_graphonly_u2_noise005,m30_t10_ssl_graphonly_u2_noise010,m31_t10_ssl_graphonly_depth1,m32_t10_ssl_graphonly_depth3,m34_t10_ssl_graphonly_proj128,m35_t10_ssl_graphonly_proj512,m42_t10_nossl_graphonly_u2_wp05,m43_t10_ssl_graphonly_valweighted,m01_t10_ssl_concat_u2_wp05,m02_t10_ssl_residual_u2_wp05,m44_t9_ssl_gated_u2_wp05,m45_t8_ssl_gated_u2_wp05,m46_t10_ssl_concat_proj128,m47_t10_ssl_concat_proj512,m50_t10_ssl_concat_valweighted,m51_t10_ssl_residual_drop20"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  GPU_ID="$GPU_ID" \
  DEVICE="$DEVICE" \
  RUN_FINAL_ALL=0 \
  USE_REPO_LOCK=1 \
  GRID_ROOT="$MULTIVIEW_GRID_ROOT" \
  EXPS_TO_RUN="$MV_EXPS" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  FORCE_RERUN="$FORCE_RERUN" \
  bash run_addh_model_experiment_grid_v2_full.sh \
    > logs_addh_strict_blind_optimized/stage2_multiview_grid.log 2>&1
  echo "[DONE] Stage 2 multi-view grid"
else
  echo "[SKIP] Stage 2 multi-view grid"
fi

# ----------------------------
# Stage 3: family-diverse strict blind variants
# ----------------------------
if [[ "$RUN_STRICT_BLIND" == "1" ]]; then
  echo "[STAGE 3] Build strict-blind family-diverse ensembles"

  # A. balanced/conservative default: still keeps source OOF strong models, but caps families.
  OUT_DIR=outputs_addh_strict_blind_diverse_balanced \
  TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=conservative WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
  AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
  FEATURE_MODE_CAP_MAP="full=3,bare_delta=2,addh_delta=2,delta=1,addh=3,addh_bare=3,bare=1,graph_only=2,concat_interact=2,gated_sum=1,residual_graph=1,text_only=0" \
  MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
  REQUIRE_FEATURE_MODES="addh_bare=1,addh=1,full=1,graph_only=1" \
  bash run_strict_blind_addhout_prediction_diverse.sh \
    > logs_addh_strict_blind_optimized/stage3a_diverse_balanced.log 2>&1

  # B. more conservative: stronger cap on full/delta, more addh/addh_bare/graph_only.
  OUT_DIR=outputs_addh_strict_blind_diverse_conservative \
  TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=conservative WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
  AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
  FEATURE_MODE_CAP_MAP="full=2,bare_delta=1,addh_delta=1,delta=1,addh=4,addh_bare=4,bare=1,graph_only=3,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0" \
  MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
  REQUIRE_FEATURE_MODES="addh_bare=2,addh=2,full=1,graph_only=1" \
  bash run_strict_blind_addhout_prediction_diverse.sh \
    > logs_addh_strict_blind_optimized/stage3b_diverse_conservative.log 2>&1

  # C. source-stability-first: emphasizes low uncertainty on addH-out unlabeled predictions.
  OUT_DIR=outputs_addh_strict_blind_diverse_stability \
  TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=stability_first WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
  AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
  FEATURE_MODE_CAP_MAP="full=2,bare_delta=2,addh_delta=2,delta=1,addh=3,addh_bare=3,bare=1,graph_only=3,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0" \
  MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
  REQUIRE_FEATURE_MODES="addh_bare=1,addh=1,graph_only=1" \
  bash run_strict_blind_addhout_prediction_diverse.sh \
    > logs_addh_strict_blind_optimized/stage3c_diverse_stability.log 2>&1

  # D. source-OOF-first but still diverse: a strict source-validation reference.
  OUT_DIR=outputs_addh_strict_blind_diverse_oof \
  TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=oof_first WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
  AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
  FEATURE_MODE_CAP_MAP="full=3,bare_delta=2,addh_delta=2,delta=1,addh=2,addh_bare=2,bare=1,graph_only=2,concat_interact=2,gated_sum=1,residual_graph=1,text_only=0" \
  MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
  REQUIRE_FEATURE_MODES="full=1,graph_only=1,addh_bare=1" \
  bash run_strict_blind_addhout_prediction_diverse.sh \
    > logs_addh_strict_blind_optimized/stage3d_diverse_oof.log 2>&1

  echo "[DONE] Stage 3 strict blind variants"
else
  echo "[SKIP] Stage 3 strict blind variants"
fi

# ----------------------------
# Stage 4: combine strict-blind strategies into one final blind prediction
# ----------------------------
if [[ "$RUN_STRATEGY_ENSEMBLE" == "1" ]]; then
  echo "[STAGE 4] Build final strict-blind strategy ensemble"
  "$PY_MM" 17_build_strict_blind_strategy_ensemble.py \
    --strategy-dirs \
      outputs_addh_strict_blind_diverse_balanced \
      outputs_addh_strict_blind_diverse_conservative \
      outputs_addh_strict_blind_diverse_stability \
      outputs_addh_strict_blind_diverse_oof \
    --addhout-master-csv outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv \
    --train-master-csv outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv \
    --out-dir outputs_addh_strict_blind_final \
    --weight-mode soft_inverse_oof \
    --min-coverage 0.80 \
    --strict-output-no-labels \
    > logs_addh_strict_blind_optimized/stage4_strategy_ensemble.log 2>&1
  echo "[DONE] Stage 4 final strategy ensemble"
fi

# ----------------------------
# Stage 5: optional post-hoc audit; labels are only used here.
# ----------------------------
if [[ "$RUN_AUDIT" == "1" ]]; then
  echo "[STAGE 5] Post-hoc audits only; these must not be used for model selection."
  for mode in balanced conservative stability oof; do
    case "$mode" in
      balanced) src="outputs_addh_strict_blind_diverse_balanced" ;;
      conservative) src="outputs_addh_strict_blind_diverse_conservative" ;;
      stability) src="outputs_addh_strict_blind_diverse_stability" ;;
      oof) src="outputs_addh_strict_blind_diverse_oof" ;;
    esac
    # Rerun same selection with AUDIT_WITH_LABELS=1 into separate audit directories.
    OUT_DIR="${src}_audit" TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" FAMILY_DIVERSE=1 AUDIT_WITH_LABELS=1 STRICT_OUTPUT_NO_LABELS=0 \
    bash run_strict_blind_addhout_prediction_diverse.sh \
      > "logs_addh_strict_blind_optimized/stage5_audit_${mode}.log" 2>&1 || true
  done
  "$PY_MM" 17_build_strict_blind_strategy_ensemble.py \
    --strategy-dirs \
      outputs_addh_strict_blind_diverse_balanced \
      outputs_addh_strict_blind_diverse_conservative \
      outputs_addh_strict_blind_diverse_stability \
      outputs_addh_strict_blind_diverse_oof \
    --addhout-master-csv outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv \
    --train-master-csv outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv \
    --out-dir outputs_addh_strict_blind_final_audit \
    --weight-mode soft_inverse_oof \
    --audit-with-labels \
    --target-col h_ads_excel \
    > logs_addh_strict_blind_optimized/stage5_final_strategy_audit.log 2>&1 || true
fi

echo "============================================================"
echo "[ALL DONE] optimized strict-blind training and prediction finished."
echo "[MAIN RESULT] outputs_addh_strict_blind_final/strict_blind_strategy_ensemble_predictions.csv"
echo "[MAIN RESULT] outputs_addh_strict_blind_final/strict_blind_strategy_ensemble_predictions.xlsx"
echo "[WEIGHTS]     outputs_addh_strict_blind_final/strict_blind_strategy_weights.csv"
echo "[LOGS]        logs_addh_strict_blind_optimized/"
echo "============================================================"
