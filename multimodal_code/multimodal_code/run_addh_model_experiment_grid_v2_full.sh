#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# AddH multimodal model-variant experiment grid runner v2 FULL
# Run from: /data/home/terminator/RL/multi-view
# Reuses existing master tables and dual graph embeddings.
# Does NOT rerun FAIR-Chem embedding extraction.
#
# Main differences from the compact v2 runner:
#   1) Expanded model/data/training grid, especially graph_only.
#   2) Resume/skip completed experiments by default.
#   3) Optional final-all train resumes per seed.
#   4) Adds projection/head-size variants.
# ============================================================

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
SRC_OUT="${SRC_OUT:-$ROOT/outputs_addh_full_mm_envsplit}"
GRID_ROOT="${GRID_ROOT:-$ROOT/outputs_addh_modelgrid_v2_full}"
REPO_ROOT="${REPO_ROOT:-$ROOT}"

ADDH_MASTER="${ADDH_MASTER:-$SRC_OUT/addH_master_target_weighted_mild.csv}"
ADDH_DUAL_EMB="${ADDH_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_MASTER="${ADDHOUT_MASTER:-$SRC_OUT/addH_out_master_normalized.csv}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"

SEEDS="${SEEDS:-42,52,62}"
EPOCHS_CLIP="${EPOCHS_CLIP:-4}"
EPOCHS_REGRESS="${EPOCHS_REGRESS:-36}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR_CLIP="${LR_CLIP:-2e-6}"
LR_REGRESS="${LR_REGRESS:-1e-6}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-26}"
LR_STAGE1_NEW="${LR_STAGE1_NEW:-4e-5}"
LR_STAGE1_TEXT_PROJ="${LR_STAGE1_TEXT_PROJ:-4e-6}"
LR_STAGE2_NEW="${LR_STAGE2_NEW:-1.5e-5}"
LR_STAGE2_TEXT_PROJ="${LR_STAGE2_TEXT_PROJ:-2e-6}"
LR_STAGE2_TEXT_TOP="${LR_STAGE2_TEXT_TOP:-7e-7}"
RUN_FINAL_ALL="${RUN_FINAL_ALL:-0}"
EXPS_TO_RUN="${EXPS_TO_RUN:-all}"
USE_REPO_LOCK="${USE_REPO_LOCK:-1}"   # 1 = serialize calls that modify repo-root yml files; safer when multiple grids run.
SKIP_COMPLETED="${SKIP_COMPLETED:-1}" # 1 = skip CV/final results that already exist.
FORCE_RERUN="${FORCE_RERUN:-0}"       # 1 = delete and rerun selected experiments even if completed.
PRINT_ONLY="${PRINT_ONLY:-0}"         # 1 = print selected experiments and exit.

mkdir -p "$GRID_ROOT"

cat <<INFO
[INFO] ROOT      = $ROOT
[INFO] GRID_ROOT = $GRID_ROOT
[INFO] PY_MM     = $PY_MM
[INFO] GPU_ID    = $GPU_ID
[INFO] SEEDS     = $SEEDS
[INFO] RUN_FINAL_ALL = $RUN_FINAL_ALL
[INFO] USE_REPO_LOCK = $USE_REPO_LOCK
[INFO] SKIP_COMPLETED = $SKIP_COMPLETED
[INFO] FORCE_RERUN = $FORCE_RERUN
INFO
echo

# Columns:
# name | target_abs_max | include_addhout_ssl(1/0) | model_variant | ensemble |
# unfreeze_top_n | sampler_power | dropout | graph_noise | reg_depth | fusion_hidden_mult |
# projection_dim | graph_proj_depth | graph_hidden_mult | regressor_hidden_mult |
# max_val_mae | max_test_rmse | max_abs_pred
EXPERIMENTS=(
# ---------- Original/key baselines ----------
"m00_t10_ssl_gated_u2_wp05|10|1|gated_sum|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m01_t10_ssl_concat_u2_wp05|10|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m02_t10_ssl_residual_u2_wp05|10|1|residual_graph|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m03_t10_ssl_concat_u1_wp05|10|1|concat_interact|median|1|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m04_t10_ssl_concat_u3_wp05|10|1|concat_interact|median|3|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m05_t10_nossl_concat_u2_wp05|10|0|concat_interact|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m06_t10_ssl_concat_u2_wp00|10|1|concat_interact|median|2|0.0|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m07_t10_ssl_concat_u2_wp08|10|1|concat_interact|median|2|0.8|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m08_t10_ssl_concat_u2_drop20|10|1|concat_interact|median|2|0.5|0.20|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m09_t10_ssl_concat_u2_noise005|10|1|concat_interact|median|2|0.5|0.10|0.005|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m10_t10_ssl_residual_u2_noise005|10|1|residual_graph|median|2|0.5|0.10|0.005|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m11_t9_ssl_concat_u2_wp05|9|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.8|9"
"m12_t8_ssl_concat_u2_wp05|8|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.5|8"
"m13_t10_ssl_graphonly_u2_wp05|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m14_t10_ssl_textonly_u2_wp05|10|1|text_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m15_t10_ssl_concat_deep_u2_wp05|10|1|concat_interact|median|2|0.5|0.15|0.000|3|3.0|256|2|1.0|1.0|3.0|4.0|10"

# ---------- Graph-only expanded grid: target range ----------
"m16_t9_ssl_graphonly_u2_wp05|9|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.8|9"
"m17_t8_ssl_graphonly_u2_wp05|8|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.5|8"
"m18_t7_ssl_graphonly_u2_wp05|7|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.2|7"

# ---------- Graph-only expanded grid: sampler power ----------
"m19_t10_ssl_graphonly_u2_wp00|10|1|graph_only|median|2|0.0|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m20_t10_ssl_graphonly_u2_wp02|10|1|graph_only|median|2|0.2|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m21_t10_ssl_graphonly_u2_wp08|10|1|graph_only|median|2|0.8|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m22_t10_ssl_graphonly_u2_wp10|10|1|graph_only|median|2|1.0|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"

# ---------- Graph-only expanded grid: dropout ----------
"m23_t10_ssl_graphonly_u2_drop00|10|1|graph_only|median|2|0.5|0.00|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m24_t10_ssl_graphonly_u2_drop05|10|1|graph_only|median|2|0.5|0.05|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m25_t10_ssl_graphonly_u2_drop15|10|1|graph_only|median|2|0.5|0.15|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m26_t10_ssl_graphonly_u2_drop20|10|1|graph_only|median|2|0.5|0.20|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m27_t10_ssl_graphonly_u2_drop30|10|1|graph_only|median|2|0.5|0.30|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"

# ---------- Graph-only expanded grid: graph noise ----------
"m28_t10_ssl_graphonly_u2_noise002|10|1|graph_only|median|2|0.5|0.10|0.002|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m29_t10_ssl_graphonly_u2_noise005|10|1|graph_only|median|2|0.5|0.10|0.005|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m30_t10_ssl_graphonly_u2_noise010|10|1|graph_only|median|2|0.5|0.10|0.010|2|2.0|256|2|1.0|1.0|3.0|4.0|10"

# ---------- Graph-only expanded grid: head/projection size ----------
"m31_t10_ssl_graphonly_depth1|10|1|graph_only|median|2|0.5|0.10|0.000|1|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m32_t10_ssl_graphonly_depth3|10|1|graph_only|median|2|0.5|0.10|0.000|3|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m33_t10_ssl_graphonly_depth4|10|1|graph_only|median|2|0.5|0.10|0.000|4|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m34_t10_ssl_graphonly_proj128|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|128|2|1.0|1.0|3.0|4.0|10"
"m35_t10_ssl_graphonly_proj512|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|512|2|1.0|1.0|3.0|4.0|10"
"m36_t10_ssl_graphonly_gproj1|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|1|1.0|1.0|3.0|4.0|10"
"m37_t10_ssl_graphonly_gproj3|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|3|1.0|1.0|3.0|4.0|10"
"m38_t10_ssl_graphonly_gh05|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|0.5|1.0|3.0|4.0|10"
"m39_t10_ssl_graphonly_gh20|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|2.0|1.0|3.0|4.0|10"
"m40_t10_ssl_graphonly_rh05|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|0.5|3.0|4.0|10"
"m41_t10_ssl_graphonly_rh20|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|2.0|3.0|4.0|10"

# ---------- Graph-only controls: SSL / ensemble ----------
"m42_t10_nossl_graphonly_u2_wp05|10|0|graph_only|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m43_t10_ssl_graphonly_valweighted|10|1|graph_only|val_mae_weighted|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"

# ---------- Best multimodal-family expanded checks ----------
"m44_t9_ssl_gated_u2_wp05|9|1|gated_sum|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.8|9"
"m45_t8_ssl_gated_u2_wp05|8|1|gated_sum|median|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|3.5|8"
"m46_t10_ssl_concat_proj128|10|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|128|2|1.0|1.0|3.0|4.0|10"
"m47_t10_ssl_concat_proj512|10|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|512|2|1.0|1.0|3.0|4.0|10"
"m48_t10_ssl_concat_depth1|10|1|concat_interact|median|2|0.5|0.10|0.000|1|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m49_t10_ssl_concat_depth3|10|1|concat_interact|median|2|0.5|0.10|0.000|3|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m50_t10_ssl_concat_valweighted|10|1|concat_interact|val_mae_weighted|2|0.5|0.10|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m51_t10_ssl_residual_drop20|10|1|residual_graph|median|2|0.5|0.20|0.000|2|2.0|256|2|1.0|1.0|3.0|4.0|10"
"m52_t10_ssl_residual_depth3|10|1|residual_graph|median|2|0.5|0.10|0.000|3|2.0|256|2|1.0|1.0|3.0|4.0|10"
)

should_run() {
  local name="$1"
  if [[ "$EXPS_TO_RUN" == "all" ]]; then return 0; fi
  IFS=',' read -ra arr <<< "$EXPS_TO_RUN"
  for x in "${arr[@]}"; do [[ "$x" == "$name" ]] && return 0; done
  return 1
}

make_by_id() {
  local in_csv="$1"; local out_csv="$2"; local clip_abs="$3"
  "$PY_MM" 12_make_addhout_by_id.py --input-csv "$in_csv" --output-csv "$out_csv" --clip-abs "$clip_abs"
}

cv_done() {
  local work_dir="$1"
  [[ -s "$work_dir/test_pred_oof_ensemble_robust_metrics.json" && -s "$work_dir/addH_out_pred_ensemble_robust_final_by_id.csv" ]]
}

final_done() {
  local exp_dir="$1"
  IFS=',' read -ra seed_arr <<< "$SEEDS"
  for S in "${seed_arr[@]}"; do
    [[ -s "$exp_dir/final_work_seed${S}/addH_out_pred_ensemble_final_by_id.csv" ]] || return 1
  done
  return 0
}

run_one_exp() {
  local spec="$1"
  IFS='|' read -r NAME TABS SSL MODELVAR ENSEMBLE UNFREEZE WP DROPOUT GNOISE REGDEPTH FUSIONMULT PROJDIM GPROJDEPTH GHIDDENMULT RHIDDENMULT MAXVAL MAXTEST MAXPRED <<< "$spec"
  if ! should_run "$NAME"; then echo "[SKIP] $NAME"; return 0; fi

  local EXP_DIR="$GRID_ROOT/$NAME"
  local CV_DIR="$EXP_DIR/cv_data"
  local WORK_DIR="$EXP_DIR/cv_work"
  local LOG_DIR="$EXP_DIR/logs"
  mkdir -p "$EXP_DIR" "$LOG_DIR"

  echo "============================================================"
  echo "[EXP] $NAME"
  echo " target_abs_max=$TABS ssl=$SSL model=$MODELVAR ensemble=$ENSEMBLE unfreeze=$UNFREEZE sampler_power=$WP dropout=$DROPOUT graph_noise=$GNOISE reg_depth=$REGDEPTH fusion_mult=$FUSIONMULT proj_dim=$PROJDIM gproj_depth=$GPROJDEPTH gh_mult=$GHIDDENMULT rh_mult=$RHIDDENMULT"
  echo "============================================================"

  local CV_ALREADY_DONE=0
  local FINAL_ALREADY_DONE=0
  if [[ "$FORCE_RERUN" != "1" && "$SKIP_COMPLETED" == "1" ]] && cv_done "$WORK_DIR"; then CV_ALREADY_DONE=1; fi
  if [[ "$FORCE_RERUN" != "1" && "$SKIP_COMPLETED" == "1" ]] && final_done "$EXP_DIR"; then FINAL_ALREADY_DONE=1; fi

  if [[ "$CV_ALREADY_DONE" == "1" && ( "$RUN_FINAL_ALL" != "1" || "$FINAL_ALREADY_DONE" == "1" ) ]]; then
    echo "[SKIP-DONE] $NAME already has completed CV/final outputs. Set FORCE_RERUN=1 to rerun."
    return 0
  fi

  cat > "$EXP_DIR/experiment_config.json" <<EOF2
{
  "name": "$NAME",
  "target_abs_max": $TABS,
  "ssl": $SSL,
  "model_variant": "$MODELVAR",
  "ensemble": "$ENSEMBLE",
  "unfreeze_top_n": $UNFREEZE,
  "sampler_power": $WP,
  "dropout": $DROPOUT,
  "graph_noise_std": $GNOISE,
  "regressor_depth": $REGDEPTH,
  "fusion_hidden_mult": $FUSIONMULT,
  "projection_dim": $PROJDIM,
  "graph_proj_depth": $GPROJDEPTH,
  "graph_hidden_mult": $GHIDDENMULT,
  "regressor_hidden_mult": $RHIDDENMULT,
  "max_val_mae": $MAXVAL,
  "max_test_rmse": $MAXTEST,
  "max_abs_pred": $MAXPRED,
  "seeds": "$SEEDS"
}
EOF2

  local SSL_ARG=()
  [[ "$SSL" == "1" ]] && SSL_ARG=(--include-addhout-in-clip)

  if [[ "$CV_ALREADY_DONE" == "1" ]]; then
    echo "[STEP A-C] CV already done; reuse $WORK_DIR"
  else
    rm -rf "$CV_DIR" "$WORK_DIR"

    echo "[STEP A] Build stratified group CV data..."
    "$PY_MM" 04_make_multiview_data_cv_multimodal.py \
      --addh-master-csv "$ADDH_MASTER" \
      --eq-emb-pkl "$ADDH_DUAL_EMB" \
      --addhout-master-csv "$ADDHOUT_MASTER" \
      --addhout-eq-emb-pkl "$ADDHOUT_DUAL_EMB" \
      --out-dir "$CV_DIR" \
      --group-col family_base_miller \
      --cv-mode stratified_gkfold \
      --n-splits 4 \
      --val-mode stratified_group_holdout \
      --val-frac 0.25 \
      --seed 42 \
      --include-regress-eq-emb \
      "${SSL_ARG[@]}" \
      --exclude-target-outliers \
      --target-abs-max "$TABS" \
      --stratify-bins 5 \
      > "$LOG_DIR/build_cv.log" 2>&1

    echo "[STEP B] Train CV ensemble with model variant: $MODELVAR"
    if [[ "$USE_REPO_LOCK" == "1" ]]; then
      exec 9>"$GRID_ROOT/.repo_yml_train.lock"
      flock 9
      echo "[INFO] repo-root yml lock acquired for $NAME CV training"
    fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" WANDB_MODE=disabled WANDB_SILENT=true TOKENIZERS_PARALLELISM=false \
    "$PY_MM" 05_run_multiview_cv_ensemble_multimodal_staged_aligned_blend3_fixclip_v2.py \
      --repo-root "$REPO_ROOT" \
      --cv-root "$CV_DIR" \
      --work-dir "$WORK_DIR" \
      --clean-run-dirs \
      --python "$PY_MM" \
      --device "$DEVICE" \
      --folds all \
      --seeds "$SEEDS" \
      --epochs-clip "$EPOCHS_CLIP" \
      --epochs-regress "$EPOCHS_REGRESS" \
      --batch-size "$BATCH_SIZE" \
      --lr-clip "$LR_CLIP" \
      --lr-regress "$LR_REGRESS" \
      --projection-dim "$PROJDIM" \
      --dropout-rate "$DROPOUT" \
      --regress-model-variant "$MODELVAR" \
      --fusion-hidden-mult "$FUSIONMULT" \
      --regressor-hidden-mult "$RHIDDENMULT" \
      --regressor-depth "$REGDEPTH" \
      --graph-proj-depth "$GPROJDEPTH" \
      --graph-hidden-mult "$GHIDDENMULT" \
      --graph-noise-std "$GNOISE" \
      --regress-loss-fn SmoothL1Loss \
      --ensemble-method "$ENSEMBLE" \
      --max-val-mae-for-ensemble "$MAXVAL" \
      --standardize-target \
      --regress-script regress_run_multimodal_staged_aligned_v2.py \
      --predict-script regress_predict_multimodal_aligned_v2.py \
      --train-strategy two_stage \
      --stage1-epochs "$STAGE1_EPOCHS" \
      --stage2-epochs "$STAGE2_EPOCHS" \
      --freeze-text-encoder-stage1 \
      --no-freeze-text-projection-stage1 \
      --unfreeze-top-n-layers-stage2 "$UNFREEZE" \
      --lr-stage1-new "$LR_STAGE1_NEW" \
      --lr-stage1-text-projection "$LR_STAGE1_TEXT_PROJ" \
      --lr-stage2-new "$LR_STAGE2_NEW" \
      --lr-stage2-text-projection "$LR_STAGE2_TEXT_PROJ" \
      --lr-stage2-text-top "$LR_STAGE2_TEXT_TOP" \
      --weight-decay 0.01 \
      --sample-weight-col w_domain \
      --use-weighted-sampler \
      --weighted-sampler-power "$WP" \
      --use-val-calibration \
      --calibration-mode bias_only \
      --sanitize-target-abs-max "$TABS" \
      --sanitize-target-outliers \
      --run-clip --run-regress --run-predict-val --run-predict-test --run-predict-out \
      > "$LOG_DIR/train_cv.gpu${GPU_ID}.log" 2>&1
    if [[ "$USE_REPO_LOCK" == "1" ]]; then
      flock -u 9 || true
      echo "[INFO] repo-root yml lock released for $NAME CV training"
    fi

    echo "[STEP C] Robust aggregation..."
    "$PY_MM" 09_reaggregate_existing_predictions_robust.py \
      --work-dir "$WORK_DIR" \
      --method median \
      --max-val-mae "$MAXVAL" \
      --max-test-rmse "$MAXTEST" \
      --max-abs-pred "$MAXPRED" \
      --topk 20 \
      > "$LOG_DIR/robust_aggregate.log" 2>&1 || true

    if [[ -f "$WORK_DIR/addH_out_pred_ensemble_robust.csv" ]]; then
      make_by_id "$WORK_DIR/addH_out_pred_ensemble_robust.csv" "$WORK_DIR/addH_out_pred_ensemble_robust_final_by_id.csv" "$MAXPRED" > "$LOG_DIR/make_by_id.log" 2>&1 || true
    fi
  fi

  if [[ "$RUN_FINAL_ALL" == "1" ]]; then
    if [[ "$FINAL_ALREADY_DONE" == "1" ]]; then
      echo "[STEP D] Final-all already done; reuse final_work_seed*"
    else
      echo "[STEP D] Build and train final-all models..."
      local FINAL_CV="$EXP_DIR/final_all_data"
      if [[ "$FORCE_RERUN" == "1" || ! -d "$FINAL_CV" ]]; then
        rm -rf "$FINAL_CV"
        "$PY_MM" 10_make_multiview_data_final_all_train.py \
          --addh-master-csv "$ADDH_MASTER" \
          --eq-emb-pkl "$ADDH_DUAL_EMB" \
          --addhout-master-csv "$ADDHOUT_MASTER" \
          --addhout-eq-emb-pkl "$ADDHOUT_DUAL_EMB" \
          --out-dir "$FINAL_CV" \
          --group-col family_base_miller \
          --val-frac 0.15 \
          --seed 2026 \
          --include-regress-eq-emb \
          "${SSL_ARG[@]}" \
          --exclude-target-outliers \
          --target-abs-max "$TABS" \
          --stratify-bins 5 \
          > "$LOG_DIR/build_final_all.log" 2>&1
      fi

      IFS=',' read -ra seed_arr <<< "$SEEDS"
      for S in "${seed_arr[@]}"; do
        local FINAL_WORK="$EXP_DIR/final_work_seed${S}"
        if [[ "$FORCE_RERUN" != "1" && "$SKIP_COMPLETED" == "1" && -s "$FINAL_WORK/addH_out_pred_ensemble_final_by_id.csv" ]]; then
          echo "[SKIP-DONE] $NAME final seed $S already done."
          continue
        fi
        rm -rf "$FINAL_WORK"
        if [[ "$USE_REPO_LOCK" == "1" ]]; then
          exec 8>"$GRID_ROOT/.repo_yml_train.lock"
          flock 8
          echo "[INFO] repo-root yml lock acquired for $NAME final seed $S"
        fi
        CUDA_VISIBLE_DEVICES="$GPU_ID" WANDB_MODE=disabled WANDB_SILENT=true TOKENIZERS_PARALLELISM=false \
        "$PY_MM" 05_run_multiview_cv_ensemble_multimodal_staged_aligned_blend3_fixclip_v2.py \
          --repo-root "$REPO_ROOT" \
          --cv-root "$FINAL_CV" \
          --work-dir "$FINAL_WORK" \
          --clean-run-dirs \
          --python "$PY_MM" \
          --device "$DEVICE" \
          --folds 0 \
          --seeds "$S" \
          --epochs-clip "$EPOCHS_CLIP" \
          --epochs-regress "$EPOCHS_REGRESS" \
          --batch-size "$BATCH_SIZE" \
          --lr-clip "$LR_CLIP" \
          --lr-regress "$LR_REGRESS" \
          --projection-dim "$PROJDIM" \
          --dropout-rate "$DROPOUT" \
          --regress-model-variant "$MODELVAR" \
          --fusion-hidden-mult "$FUSIONMULT" \
          --regressor-hidden-mult "$RHIDDENMULT" \
          --regressor-depth "$REGDEPTH" \
          --graph-proj-depth "$GPROJDEPTH" \
          --graph-hidden-mult "$GHIDDENMULT" \
          --graph-noise-std "$GNOISE" \
          --regress-loss-fn SmoothL1Loss \
          --ensemble-method median \
          --standardize-target \
          --regress-script regress_run_multimodal_staged_aligned_v2.py \
          --predict-script regress_predict_multimodal_aligned_v2.py \
          --train-strategy two_stage \
          --stage1-epochs "$STAGE1_EPOCHS" \
          --stage2-epochs "$STAGE2_EPOCHS" \
          --freeze-text-encoder-stage1 \
          --no-freeze-text-projection-stage1 \
          --unfreeze-top-n-layers-stage2 "$UNFREEZE" \
          --lr-stage1-new "$LR_STAGE1_NEW" \
          --lr-stage1-text-projection "$LR_STAGE1_TEXT_PROJ" \
          --lr-stage2-new "$LR_STAGE2_NEW" \
          --lr-stage2-text-projection "$LR_STAGE2_TEXT_PROJ" \
          --lr-stage2-text-top "$LR_STAGE2_TEXT_TOP" \
          --weight-decay 0.01 \
          --sample-weight-col w_domain \
          --use-weighted-sampler \
          --weighted-sampler-power "$WP" \
          --use-val-calibration \
          --calibration-mode bias_only \
          --run-clip --run-regress --run-predict-val --run-predict-out \
          > "$LOG_DIR/final_all_seed${S}.gpu${GPU_ID}.log" 2>&1 || true
        if [[ "$USE_REPO_LOCK" == "1" ]]; then
          flock -u 8 || true
          echo "[INFO] repo-root yml lock released for $NAME final seed $S"
        fi
        if [[ -f "$FINAL_WORK/addH_out_pred_ensemble.csv" ]]; then
          make_by_id "$FINAL_WORK/addH_out_pred_ensemble.csv" "$FINAL_WORK/addH_out_pred_ensemble_final_by_id.csv" "$MAXPRED" > "$LOG_DIR/final_make_by_id_seed${S}.log" 2>&1 || true
        fi
      done
    fi
  fi

  echo "[DONE] $NAME"
  echo
}

if [[ "$PRINT_ONLY" == "1" ]]; then
  echo "[PRINT_ONLY] selected experiments:"
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME _rest <<< "$spec"
    should_run "$NAME" && echo "  $spec"
  done
  exit 0
fi

for spec in "${EXPERIMENTS[@]}"; do
  run_one_exp "$spec"
done

echo "[FINAL] Comparing all experiments..."
"$PY_MM" 11_compare_addh_experiments_v2.py \
  --grid-root "$GRID_ROOT" \
  --output-csv "$GRID_ROOT/experiment_comparison_summary.csv" \
  --output-xlsx "$GRID_ROOT/experiment_comparison_summary.xlsx" \
  > "$GRID_ROOT/compare.log" 2>&1 || true

echo "[OK] all model-variant experiments finished."
echo "[RESULT] $GRID_ROOT/experiment_comparison_summary.csv"
echo "[RESULT] $GRID_ROOT/experiment_comparison_summary.xlsx"
