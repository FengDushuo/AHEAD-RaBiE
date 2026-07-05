#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# AddH multimodal model-variant experiment grid runner v2
# Run from: /data/home/terminator/RL/multi-view
# Reuses existing master tables and dual graph embeddings.
# Does NOT rerun FAIR-Chem embedding extraction.
# ============================================================

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
SRC_OUT="${SRC_OUT:-$ROOT/outputs_addh_full_mm_envsplit}"
GRID_ROOT="${GRID_ROOT:-$ROOT/outputs_addh_modelgrid_v2}"
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

mkdir -p "$GRID_ROOT"

echo "[INFO] ROOT      = $ROOT"
echo "[INFO] GRID_ROOT = $GRID_ROOT"
echo "[INFO] PY_MM     = $PY_MM"
echo "[INFO] GPU_ID    = $GPU_ID"
echo "[INFO] SEEDS     = $SEEDS"
echo "[INFO] RUN_FINAL_ALL = $RUN_FINAL_ALL"
echo "[INFO] USE_REPO_LOCK = $USE_REPO_LOCK"
echo

# Columns:
# name | target_abs_max | include_addhout_ssl(1/0) | model_variant | ensemble | unfreeze_top_n | sampler_power | dropout | graph_noise | reg_depth | fusion_hidden_mult | max_val_mae | max_test_rmse | max_abs_pred
EXPERIMENTS=(
"m00_t10_ssl_gated_u2_wp05|10|1|gated_sum|median|2|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m01_t10_ssl_concat_u2_wp05|10|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m02_t10_ssl_residual_u2_wp05|10|1|residual_graph|median|2|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m03_t10_ssl_concat_u1_wp05|10|1|concat_interact|median|1|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m04_t10_ssl_concat_u3_wp05|10|1|concat_interact|median|3|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m05_t10_nossl_concat_u2_wp05|10|0|concat_interact|median|2|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m06_t10_ssl_concat_u2_wp00|10|1|concat_interact|median|2|0.0|0.10|0.000|2|2.0|3.0|4.0|10"
"m07_t10_ssl_concat_u2_wp08|10|1|concat_interact|median|2|0.8|0.10|0.000|2|2.0|3.0|4.0|10"
"m08_t10_ssl_concat_u2_drop20|10|1|concat_interact|median|2|0.5|0.20|0.000|2|2.0|3.0|4.0|10"
"m09_t10_ssl_concat_u2_noise005|10|1|concat_interact|median|2|0.5|0.10|0.005|2|2.0|3.0|4.0|10"
"m10_t10_ssl_residual_u2_noise005|10|1|residual_graph|median|2|0.5|0.10|0.005|2|2.0|3.0|4.0|10"
"m11_t9_ssl_concat_u2_wp05|9|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|3.0|3.8|9"
"m12_t8_ssl_concat_u2_wp05|8|1|concat_interact|median|2|0.5|0.10|0.000|2|2.0|3.0|3.5|8"
"m13_t10_ssl_graphonly_u2_wp05|10|1|graph_only|median|2|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m14_t10_ssl_textonly_u2_wp05|10|1|text_only|median|2|0.5|0.10|0.000|2|2.0|3.0|4.0|10"
"m15_t10_ssl_concat_deep_u2_wp05|10|1|concat_interact|median|2|0.5|0.15|0.000|3|3.0|3.0|4.0|10"
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

run_one_exp() {
  local spec="$1"
  IFS='|' read -r NAME TABS SSL MODELVAR ENSEMBLE UNFREEZE WP DROPOUT GNOISE REGDEPTH FUSIONMULT MAXVAL MAXTEST MAXPRED <<< "$spec"
  if ! should_run "$NAME"; then echo "[SKIP] $NAME"; return 0; fi

  local EXP_DIR="$GRID_ROOT/$NAME"
  local CV_DIR="$EXP_DIR/cv_data"
  local WORK_DIR="$EXP_DIR/cv_work"
  local LOG_DIR="$EXP_DIR/logs"
  mkdir -p "$EXP_DIR" "$LOG_DIR"

  echo "============================================================"
  echo "[EXP] $NAME"
  echo " target_abs_max=$TABS ssl=$SSL model=$MODELVAR ensemble=$ENSEMBLE unfreeze=$UNFREEZE sampler_power=$WP dropout=$DROPOUT graph_noise=$GNOISE reg_depth=$REGDEPTH fusion_mult=$FUSIONMULT"
  echo "============================================================"

  cat > "$EXP_DIR/experiment_config.json" <<EOF
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
  "seeds": "$SEEDS"
}
EOF

  rm -rf "$CV_DIR" "$WORK_DIR"
  local SSL_ARG=()
  [[ "$SSL" == "1" ]] && SSL_ARG=(--include-addhout-in-clip)

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
    --projection-dim 256 \
    --dropout-rate "$DROPOUT" \
    --regress-model-variant "$MODELVAR" \
    --fusion-hidden-mult "$FUSIONMULT" \
    --regressor-hidden-mult 1.0 \
    --regressor-depth "$REGDEPTH" \
    --graph-proj-depth 2 \
    --graph-hidden-mult 1.0 \
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

  if [[ "$RUN_FINAL_ALL" == "1" ]]; then
    echo "[STEP D] Build and train final-all models..."
    local FINAL_CV="$EXP_DIR/final_all_data"
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

    IFS=',' read -ra seed_arr <<< "$SEEDS"
    for S in "${seed_arr[@]}"; do
      local FINAL_WORK="$EXP_DIR/final_work_seed${S}"
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
        --projection-dim 256 \
        --dropout-rate "$DROPOUT" \
        --regress-model-variant "$MODELVAR" \
        --fusion-hidden-mult "$FUSIONMULT" \
        --regressor-depth "$REGDEPTH" \
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
    done
  fi

  echo "[DONE] $NAME"
  echo
}

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
