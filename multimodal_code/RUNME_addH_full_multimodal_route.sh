#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
ADDH_ROOT="${ADDH_ROOT:-addH}"
ADDH2_ROOT="${ADDH2_ROOT:-addH-2}"
ADDHOUT_ROOT="${ADDHOUT_ROOT:-addH-out}"
MODEL_DIR="${MODEL_DIR:-/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd}"
REPO_ROOT="${REPO_ROOT:-/data/home/terminator/RL/multi-view}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
EH_REF="${EH_REF:--0.0565}"

OUT="${OUT:-outputs_addh_full_mm}"
mkdir -p "$OUT"

# 1) Build normalized training masters.
$PYTHON 01_build_addh_master_with_outlier_drop.py \
  --input-dir "$ADDH_ROOT" \
  --output-csv "$OUT/addH_master_flagged.csv" \
  --outlier-method iqr --outlier-action flag --iqr-multiplier 3.0 \
  --outlier-report-csv "$OUT/addH_outlier_report.csv" \
  --outlier-summary-json "$OUT/addH_outlier_summary.json"

$PYTHON build_addh_master_from_ml2_layout.py \
  --input-root "$ADDH2_ROOT" \
  --output-csv "$OUT/addH2_master_flagged.csv" \
  --base-miller-map "2542=100,2858=111,643=111" \
  --outlier-method iqr --outlier-action flag --iqr-multiplier 3.0 \
  --outlier-report-csv "$OUT/addH2_outlier_report.csv" \
  --outlier-summary-json "$OUT/addH2_outlier_summary.json"

$PYTHON merge_addh_master_tables_robust.py \
  --old-csv "$OUT/addH_master_flagged.csv" \
  --new-csv "$OUT/addH2_master_flagged.csv" \
  --output-csv "$OUT/addH_master_merged_robust.csv" \
  --merged-raw-csv "$OUT/addH_master_merged_raw.csv" \
  --dedup-policy quality_then_new \
  --outlier-method none --outlier-action flag

# 2) Build addH-out master and generate bare_from_addH structures for target-domain dual graph embeddings.
$PYTHON 02_build_addhout_master_normalized.py \
  --input-dir "$ADDHOUT_ROOT" \
  --output-csv "$OUT/addH_out_master_normalized.csv" \
  --miller-map "CeO2=111,ZnO=100" \
  --eh-ref "$EH_REF" \
  --target-source excel_preferred \
  --write-bare-from-addh \
  --bare-output-dir "$OUT/addH_out_bare_from_addH"

# 3) Extract addH and bare Equiformer embeddings, then concatenate [addH, bare, addH-bare].
$PYTHON 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_master_merged_robust.csv" \
  --structure-col contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_addh_eq_emb.pkl" \
  --meta-csv "$OUT/addH_addh_eq_emb.meta.csv" --device "$DEVICE" --hook-module energy_block

$PYTHON 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_master_merged_robust.csv" \
  --structure-col bare_contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_bare_eq_emb.pkl" \
  --meta-csv "$OUT/addH_bare_eq_emb.meta.csv" --device "$DEVICE" --hook-module energy_block

$PYTHON 06_build_dual_graph_embeddings.py \
  --master-csv "$OUT/addH_master_merged_robust.csv" \
  --addh-emb-pkl "$OUT/addH_addh_eq_emb.pkl" \
  --bare-emb-pkl "$OUT/addH_bare_eq_emb.pkl" \
  --save-pkl "$OUT/addH_dual_eq_emb.pkl" \
  --meta-csv "$OUT/addH_dual_eq_emb.meta.csv" \
  --missing-bare skip

$PYTHON 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_out_master_normalized.csv" \
  --structure-col contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_out_addh_eq_emb.pkl" \
  --meta-csv "$OUT/addH_out_addh_eq_emb.meta.csv" --device "$DEVICE" --hook-module energy_block

$PYTHON 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_out_master_normalized.csv" \
  --structure-col bare_contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_out_bare_eq_emb.pkl" \
  --meta-csv "$OUT/addH_out_bare_eq_emb.meta.csv" --device "$DEVICE" --hook-module energy_block

$PYTHON 06_build_dual_graph_embeddings.py \
  --master-csv "$OUT/addH_out_master_normalized.csv" \
  --addh-emb-pkl "$OUT/addH_out_addh_eq_emb.pkl" \
  --bare-emb-pkl "$OUT/addH_out_bare_eq_emb.pkl" \
  --save-pkl "$OUT/addH_out_dual_eq_emb.pkl" \
  --meta-csv "$OUT/addH_out_dual_eq_emb.meta.csv" \
  --missing-bare skip

# 4) Mild target-domain weighting. This is optional but recommended for CeO2/ZnO addH-out.
$PYTHON make_target_domain_weighted_train_table_mild.py \
  --train-csv "$OUT/addH_master_merged_robust.csv" \
  --target-csv "$OUT/addH_out_master_normalized.csv" \
  --output-csv "$OUT/addH_master_target_weighted_mild.csv" \
  --debug-csv "$OUT/target_domain_weight_debug_mild.csv" \
  --profile-json "$OUT/target_domain_profile_mild.json" \
  --train-eq-emb-pkl "$OUT/addH_dual_eq_emb.pkl" \
  --target-eq-emb-pkl "$OUT/addH_out_dual_eq_emb.pkl" \
  --use-emb --emb-min 0.97 --emb-max 1.06

# 5) Build strict group OOF splits. addH-out enters CLIP only as unlabeled graph-text pairs.
$PYTHON 04_make_multiview_data_cv_multimodal.py \
  --addh-master-csv "$OUT/addH_master_target_weighted_mild.csv" \
  --eq-emb-pkl "$OUT/addH_dual_eq_emb.pkl" \
  --addhout-master-csv "$OUT/addH_out_master_normalized.csv" \
  --addhout-eq-emb-pkl "$OUT/addH_out_dual_eq_emb.pkl" \
  --out-dir "$OUT/multiview_local_data_cv_mm_full" \
  --group-col family_base_miller \
  --cv-mode gkfold --n-splits 4 \
  --val-mode group_holdout --val-frac 0.25 \
  --seed 42 \
  --include-regress-eq-emb \
  --include-addhout-in-clip

# 6) Multi-view training/prediction. Add --run-threeway-blend with cat/nn roots after single-view models are available.
nohup env WANDB_MODE=disabled WANDB_SILENT=true $PYTHON 05_run_multiview_cv_ensemble_multimodal_staged_aligned_blend3_fixclip.py \
  --repo-root "$REPO_ROOT" \
  --cv-root "$OUT/multiview_local_data_cv_mm_full" \
  --work-dir "$OUT/multiview_cv_mm_full_fixclip" \
  --python "$PYTHON" --device "$DEVICE" \
  --folds all --seeds 42,52,62 \
  --epochs-clip 4 --epochs-regress 36 --batch-size 8 \
  --lr-clip 2e-6 --lr-regress 1e-6 \
  --regress-loss-fn SmoothL1Loss \
  --ensemble-method val_mae_weighted \
  --standardize-target \
  --regress-script regress_run_multimodal_staged_aligned.py \
  --predict-script regress_predict_multimodal_aligned.py \
  --train-strategy two_stage \
  --stage1-epochs 10 --stage2-epochs 26 \
  --freeze-text-encoder-stage1 --no-freeze-text-projection-stage1 \
  --unfreeze-top-n-layers-stage2 2 \
  --lr-stage1-new 4e-5 --lr-stage1-text-projection 4e-6 \
  --lr-stage2-new 1.5e-5 --lr-stage2-text-projection 2e-6 --lr-stage2-text-top 7e-7 \
  --weight-decay 0.01 \
  --sample-weight-col w_domain --use-weighted-sampler --weighted-sampler-power 0.5 \
  --use-val-calibration --calibration-mode bias_only \
  --run-clip --run-regress --run-predict-val --run-predict-test --run-predict-out \
  > "$OUT/cv_mm_full_fixclip.log" 2>&1 &
