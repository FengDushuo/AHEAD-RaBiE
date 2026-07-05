#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# One-shot graph-embedding refine grid for addH-out prediction
# Focus:
#   1) addh-only absolute-value models, because addh has low bias on addH-out
#   2) full/addh_delta/bare_delta ranking-reference models, because full has better ranking
#   3) automatic posterior summary + final candidate ranking table
# Run from: /data/home/terminator/RL/multi-view
# ============================================================

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
SRC_OUT="${SRC_OUT:-$ROOT/outputs_addh_full_mm_envsplit}"
BASE_GRID_ROOT="${BASE_GRID_ROOT:-$ROOT/outputs_addh_graph_ensemble}"
GRID_ROOT="${GRID_ROOT:-$ROOT/outputs_addh_graph_ensemble_refine_v2}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT/13_train_graph_embedding_ensemble_v2.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$ROOT/14_summarize_graph_ensemble_addhout_v2.py}"

ADDH_MASTER="${ADDH_MASTER:-$SRC_OUT/addH_master_target_weighted_mild.csv}"
ADDH_DUAL_EMB="${ADDH_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_MASTER="${ADDHOUT_MASTER:-$SRC_OUT/addH_out_master_normalized.csv}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"

SEEDS="${SEEDS:-42,52,62,72,82,92,102,112,122,132}"
EXPS_TO_RUN="${EXPS_TO_RUN:-all}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
N_SPLITS="${N_SPLITS:-4}"
VAL_FRAC="${VAL_FRAC:-0.25}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

mkdir -p "$GRID_ROOT"

# Basic checks
for f in "$TRAIN_SCRIPT" "$SUMMARY_SCRIPT" "04_make_multiview_data_cv_multimodal.py"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required script/file: $f" >&2
    exit 1
  fi
done
for f in "$ADDH_MASTER" "$ADDH_DUAL_EMB" "$ADDHOUT_MASTER" "$ADDHOUT_DUAL_EMB"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required data file: $f" >&2
    exit 1
  fi
done

# Model sets
COMMON_MODELS="ridge,huber,elasticnet,extratrees,rf,hgb,gbrt,mlp,catboost"
FAST_MODELS="ridge,huber,elasticnet,extratrees,hgb,gbrt,mlp_small,catboost"
TREE_MODELS="extratrees,extratrees_deep,rf,hgb,gbrt,catboost,xgboost,lightgbm"
LINEAR_MODELS="ridge,ridge_strong,huber,elasticnet,mlp_small"
ROBUST_MODELS="ridge,huber,extratrees,hgb,gbrt,catboost"

# Columns:
# name | target_abs_max | feature_mode | pca_dim | models | max_val_mae | max_test_rmse | max_abs_pred | stack(1/0) | calibration | aggregate
EXPERIMENTS=(
# ---------- addh-only absolute-value main line ----------
"ge20_t10_addh_common_bias_median|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge21_t9_addh_common_bias_median|9|addh|0|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge22_t8_addh_common_bias_median|8|addh|0|$COMMON_MODELS|3.0|3.5|8|1|bias|median"
"ge23_t7_addh_common_bias_median|7|addh|0|$COMMON_MODELS|2.8|3.2|7|1|bias|median"
"ge24_t95_addh_common_bias_median|9.5|addh|0|$COMMON_MODELS|3.0|3.9|9.5|1|bias|median"
"ge25_t85_addh_common_bias_median|8.5|addh|0|$COMMON_MODELS|3.0|3.6|8.5|1|bias|median"

# ---------- addh-only PCA / dimension reduction ----------
"ge26_t10_addh_pca32_bias_median|10|addh|32|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge27_t10_addh_pca64_bias_median|10|addh|64|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge28_t10_addh_pca128_bias_median|10|addh|128|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge29_t9_addh_pca64_bias_median|9|addh|64|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge30_t8_addh_pca64_bias_median|8|addh|64|$COMMON_MODELS|3.0|3.5|8|1|bias|median"

# ---------- addh-only model family tests ----------
"ge31_t10_addh_fast_bias_median|10|addh|0|$FAST_MODELS|3.0|4.0|10|1|bias|median"
"ge32_t10_addh_tree_bias_median|10|addh|0|$TREE_MODELS|3.0|4.0|10|1|bias|median"
"ge33_t10_addh_linear_bias_median|10|addh|0|$LINEAR_MODELS|3.0|4.0|10|1|bias|median"
"ge34_t9_addh_tree_bias_median|9|addh|0|$TREE_MODELS|3.0|3.8|9|1|bias|median"
"ge35_t9_addh_linear_bias_median|9|addh|0|$LINEAR_MODELS|3.0|3.8|9|1|bias|median"

# ---------- addh-only calibration / aggregation tests ----------
"ge36_t10_addh_common_none_median|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1|none|median"
"ge37_t10_addh_common_linear_median|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1|linear|median"
"ge38_t10_addh_common_bias_mean|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1|bias|mean"
"ge39_t10_addh_common_bias_trimmed|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1|bias|trimmed_mean"
"ge40_t10_addh_common_bias_valweighted|10|addh|0|$COMMON_MODELS|3.0|4.0|10|1|bias|val_mae_weighted"
"ge41_t10_addh_common_bias_nostack|10|addh|0|$COMMON_MODELS|3.0|4.0|10|0|bias|median"

# ---------- extra absolute-value feature tests ----------
"ge42_t10_addh_bare_common_bias_median|10|addh_bare|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge43_t10_addh_delta_common_bias_median|10|addh_delta|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge44_t9_addh_delta_common_bias_median|9|addh_delta|0|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge45_t8_addh_delta_common_bias_median|8|addh_delta|0|$COMMON_MODELS|3.0|3.5|8|1|bias|median"

# ---------- full embedding ranking-reference line ----------
"ge50_t10_full_common_bias_median|10|full|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge51_t9_full_common_bias_median|9|full|0|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge52_t8_full_common_bias_median|8|full|0|$COMMON_MODELS|3.0|3.5|8|1|bias|median"
"ge53_t7_full_common_bias_median|7|full|0|$COMMON_MODELS|2.8|3.2|7|1|bias|median"
"ge54_t10_full_pca64_bias_median|10|full|64|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge55_t9_full_pca64_bias_median|9|full|64|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge56_t10_full_linear_bias_median|10|full|0|$LINEAR_MODELS|3.0|4.0|10|1|bias|median"
"ge57_t10_full_fast_bias_median|10|full|0|$FAST_MODELS|3.0|4.0|10|1|bias|median"
"ge58_t10_full_common_linear_median|10|full|0|$COMMON_MODELS|3.0|4.0|10|1|linear|median"
"ge59_t10_full_common_bias_valweighted|10|full|0|$COMMON_MODELS|3.0|4.0|10|1|bias|val_mae_weighted"
"ge60_t10_full_common_bias_nostack|10|full|0|$COMMON_MODELS|3.0|4.0|10|0|bias|median"

# ---------- ranking/reference alternatives ----------
"ge61_t10_bare_delta_common_bias_median|10|bare_delta|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge62_t9_bare_delta_common_bias_median|9|bare_delta|0|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge63_t10_delta_common_bias_median|10|delta|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge64_t9_delta_common_bias_median|9|delta|0|$COMMON_MODELS|3.0|3.8|9|1|bias|median"
"ge65_t10_bare_common_bias_median|10|bare|0|$COMMON_MODELS|3.0|4.0|10|1|bias|median"
"ge66_t10_addh_common_robust_bias_median|10|addh|0|$ROBUST_MODELS|3.0|4.0|10|1|bias|median"
"ge67_t9_addh_common_robust_bias_median|9|addh|0|$ROBUST_MODELS|3.0|3.8|9|1|bias|median"
)

should_run() {
  local name="$1"
  if [[ "$EXPS_TO_RUN" == "all" ]]; then return 0; fi
  IFS=',' read -ra arr <<< "$EXPS_TO_RUN"
  for x in "${arr[@]}"; do [[ "$x" == "$name" ]] && return 0; done
  return 1
}

run_one() {
  local spec="$1"
  IFS='|' read -r NAME TABS FMODE PCA MODELS MAXVAL MAXTEST MAXPRED STACK CALIB AGG <<< "$spec"
  if ! should_run "$NAME"; then echo "[SKIP] $NAME"; return 0; fi

  local EXP_DIR="$GRID_ROOT/$NAME"
  local CV_DIR="$EXP_DIR/cv_data"
  local WORK_DIR="$EXP_DIR/work"
  local LOG_DIR="$EXP_DIR/logs"
  mkdir -p "$EXP_DIR" "$LOG_DIR"

  if [[ "$SKIP_COMPLETED" == "1" && "$FORCE_RERUN" != "1" && -f "$WORK_DIR/test_oof_graph_ensemble_metrics.json" && -f "$WORK_DIR/addH_out_graph_ensemble_by_id.csv" ]]; then
    echo "[DONE/SKIP] $NAME already completed"
    return 0
  fi

  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -rf "$CV_DIR" "$WORK_DIR"
  fi
  mkdir -p "$CV_DIR" "$WORK_DIR"

  echo "============================================================"
  echo "[EXP] $NAME"
  echo " target_abs_max=$TABS feature_mode=$FMODE pca_dim=$PCA stack=$STACK calibration=$CALIB aggregate=$AGG"
  echo " models=$MODELS"
  echo "============================================================"

  cat > "$EXP_DIR/experiment_config.json" <<EOF
{
  "name": "$NAME",
  "target_abs_max": $TABS,
  "feature_mode": "$FMODE",
  "pca_dim": $PCA,
  "models": "$MODELS",
  "seeds": "$SEEDS",
  "enable_stacking": $STACK,
  "calibration": "$CALIB",
  "aggregate_method": "$AGG"
}
EOF

  if [[ ! -f "$CV_DIR/fold_0/regress_train.pkl" ]]; then
    echo "[STEP A] Build CV pkl..."
    "$PY_MM" 04_make_multiview_data_cv_multimodal.py \
      --addh-master-csv "$ADDH_MASTER" \
      --eq-emb-pkl "$ADDH_DUAL_EMB" \
      --addhout-master-csv "$ADDHOUT_MASTER" \
      --addhout-eq-emb-pkl "$ADDHOUT_DUAL_EMB" \
      --out-dir "$CV_DIR" \
      --group-col family_base_miller \
      --cv-mode stratified_gkfold \
      --n-splits "$N_SPLITS" \
      --val-mode stratified_group_holdout \
      --val-frac "$VAL_FRAC" \
      --seed 42 \
      --include-regress-eq-emb \
      --include-addhout-in-clip \
      --exclude-target-outliers \
      --target-abs-max "$TABS" \
      --stratify-bins 5 \
      > "$LOG_DIR/build_cv.log" 2>&1
  else
    echo "[STEP A] Reuse CV data: $CV_DIR"
  fi

  echo "[STEP B] Train graph embedding ensemble..."
  STACK_ARG=()
  [[ "$STACK" == "1" ]] && STACK_ARG=(--enable-stacking)
  "$PY_MM" "$TRAIN_SCRIPT" \
    --cv-root "$CV_DIR" \
    --work-dir "$WORK_DIR" \
    --models "$MODELS" \
    --seeds "$SEEDS" \
    --feature-mode "$FMODE" \
    --pca-dim "$PCA" \
    --target-abs-max "$TABS" \
    --drop-outlier-flags \
    --calibration "$CALIB" \
    --aggregate-method "$AGG" \
    --max-val-mae "$MAXVAL" \
    --max-test-rmse "$MAXTEST" \
    --max-abs-pred "$MAXPRED" \
    --clip-pred-abs "$MAXPRED" \
    "${STACK_ARG[@]}" \
    --addhout-master-csv "$ADDHOUT_MASTER" \
    --addhout-eq-emb-pkl "$ADDHOUT_DUAL_EMB" \
    > "$LOG_DIR/train_graph_ensemble.log" 2>&1

  echo "[DONE] $NAME"
}

print_plan() {
  echo "[PLAN] GRID_ROOT=$GRID_ROOT"
  echo "[PLAN] total experiments=${#EXPERIMENTS[@]}"
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME TABS FMODE PCA MODELS MAXVAL MAXTEST MAXPRED STACK CALIB AGG <<< "$spec"
    if should_run "$NAME"; then
      echo "  $NAME | tabs=$TABS | mode=$FMODE | pca=$PCA | calib=$CALIB | agg=$AGG | stack=$STACK"
    fi
  done
}

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
  print_plan
  exit 0
fi

print_plan
for spec in "${EXPERIMENTS[@]}"; do
  run_one "$spec"
done

echo "[STEP C] Summarize posterior addH-out performance and build final candidate table..."
"$PY_MM" "$SUMMARY_SCRIPT" \
  --grid-roots "$BASE_GRID_ROOT,$GRID_ROOT" \
  --addhout-master-csv "$ADDHOUT_MASTER" \
  --out-dir "$GRID_ROOT" \
  --true-col h_ads_excel \
  --absolute-exp auto \
  --ranking-exps auto \
  --top-rank-n 3 \
  > "$GRID_ROOT/summarize_and_final_candidates.log" 2>&1 || true

echo "[OK] refine graph ensemble grid finished."
echo "[RESULT] $GRID_ROOT/graph_ensemble_addhout_posterior_summary.csv"
echo "[RESULT] $GRID_ROOT/graph_ensemble_addhout_posterior_summary.xlsx"
echo "[RESULT] $GRID_ROOT/final_addhout_candidate_ranking.csv"
echo "[RESULT] $GRID_ROOT/final_addhout_candidate_ranking.xlsx"
