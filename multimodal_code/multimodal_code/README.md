# addH → addH-out multimodal pipeline for `multi-view`

This package builds a stricter two-stage workflow:

1. `addH` is treated as the **development / training dataset**.
2. `addH-out` is treated as the **final inference dataset**.
3. Graph embeddings (`eq_emb`) are extracted with a pretrained **FAIR-Chem** model.
4. `clip_run.py` is used for multimodal SSL/domain adaptation.
5. `regress_run.py` is used for downstream regression fine-tuning.
6. Only after held-out `addH` test performance is acceptable do you run `addH-out` prediction.

## Target definition

The target is

`target = Etotal - Eslab - EH`

with `EH = -0.0565` by default.

For `addH`, this means:

`target = energy_addH - energy_bare - (-0.0565) = delta_e_raw + 0.0565`

## Leakage control

The default split groups by `family_base`, i.e. the prefix before the first `-` in IDs such as:

- `353-011-Au`
- `361-101-Pt`

This is a **strict split**: all dopants and Miller variants from the same base family go to the same split.

## Script order

```bash
python build_addh_master_from_ml2_layout.py \
  --input-root addH-2 \
  --output-csv addH_master_ml2.csv \
  --base-miller-map "2542=100,2858=111,643=111"

python merge_addh_master_tables_robust.py \
  --old-csv addH_master_drop8.csv \
  --new-csv addH_master_ml2.csv \
  --output-csv addH_master_merged_robust.csv \
  --merged-raw-csv addH_master_merged_raw.csv \
  --dedup-policy quality_then_new \
  --outlier-method iqr \
  --outlier-action drop \
  --iqr-multiplier 3.0
```

### 1) Build `addH` master table

conda activate multiview

```bash
python 01_build_addh_master_with_outlier_drop.py \
  --input-dir addH \
  --output-csv addH_master_drop8.csv \
  --outlier-method iqr \
  --outlier-action drop \
  --iqr-multiplier 3.0 \
  --outlier-report-csv addH_master_drop8_outlier_report.csv \
  --outlier-summary-json addH_master_drop8_outlier_summary.json
```

### 2) Build `addH-out` master table

```bash
python 02_build_addhout_master_miller_map.py \
  --input-dir addH-out \
  --excel-path addH-out/氢吸附能.xlsx \
  --output-csv addH_out_master.csv \
  --miller-map "CeO2=111,ZnO=100"
```

### 3) Extract FAIR-Chem graph embeddings for `addH`

conda activate fairchem

```bash
# python 03_extract_eq_emb_fairchem.py \
#   --master-csv addH_master_drop8.csv \
#   --structure-col contcar_path \
#   --id-col id \
#   --text-col text \
#   --target-col target \
#   --model-dir /data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd \
#   --save-pkl addH_eq_emb.pkl \
#   --save-dataset-pkl addH_eq_emb_dataset.pkl \
#   --device cpu \
#   --hook-module blocks.7 

# python 03_extract_eq_emb_fairchem.py \
#   --master-csv addH_out_master.csv \
#   --structure-col contcar_path \
#   --id-col id \
#   --text-col text \
#   --target-col h_ads_excel \
#   --model-dir /data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd \
#   --save-pkl addH_out_eq_emb.pkl \
#   --save-dataset-pkl addH_out_eq_emb_dataset.pkl \
#   --device cpu \
#   --hook-module blocks.7

python 03_extract_eq_emb_fairchem.py \
  --master-csv addH_master_merged_robust.csv \
  --structure-col contcar_path \
  --id-col id \
  --text-col text \
  --target-col target \
  --model-dir /data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd \
  --save-pkl addH_eq_emb_merged.pkl \
  --save-dataset-pkl addH_eq_emb_merged_dataset.pkl \
  --device cpu \
  --hook-module blocks.7
```

Notes:
- This script uses forward hooks on the FAIR-Chem model to capture graph-level descriptors.
- FAIR-Chem does **not** currently provide a stable, official public API specifically for eqV2/eSCN descriptor extraction, so this is a practical hook-based solution.
- If no embeddings are captured, rerun with `--hook-module-substring backbone` or another appropriate substring.

### 4) Build `multi-view` training / validation / test pickles

conda activate multiview

```bash
python 04_make_multiview_data_cv_multimodal.py \
  --addh-master-csv addH_master_merged_robust.csv \
  --eq-emb-pkl addH_eq_emb_merged.pkl \
  --addhout-master-csv addH_out_master.csv \
  --addhout-eq-emb-pkl addH_out_eq_emb.pkl \
  --out-dir multiview_local_data_cv_mm_merged \
  --group-col family_base \
  --cv-mode gkfold \
  --n-splits 4 \
  --val-mode group_holdout \
  --val-frac 0.25 \
  --seed 42 \
  --include-regress-eq-emb
```

Outputs include:

- `clip_train.pkl`
- `clip_val.pkl`
- `regress_train.pkl`
- `regress_val.pkl`
- `regress_test.pkl`
- `addH_out_pred_input.pkl`

### 5) Run multimodal pretraining, regression fine-tuning, held-out test prediction, then `addH-out`

```bash

nohup env WANDB_MODE=disabled WANDB_SILENT=true python 05_run_multiview_cv_ensemble_multimodal_staged.py \
  --repo-root /data/home/terminator/RL/multi-view \
  --cv-root /data/home/terminator/RL/multi-view/multiview_local_data_cv_mm_merged \
  --work-dir /data/home/terminator/RL/multi-view/multiview_cv_mm_merged_staged_v1 \
  --python python \
  --device cuda \
  --folds all \
  --seeds 42,52 \
  --epochs-clip 6 \
  --epochs-regress 40 \
  --batch-size 8 \
  --lr-clip 2e-6 \
  --lr-regress 1e-6 \
  --regress-loss-fn L1Loss \
  --ensemble-method mean \
  --standardize-target \
  --regress-script regress_run_multimodal_staged.py \
  --train-strategy two_stage \
  --stage1-epochs 12 \
  --stage2-epochs 28 \
  --freeze-text-encoder-stage1 \
  --no-freeze-text-projection-stage1 \
  --unfreeze-top-n-layers-stage2 4 \
  --lr-stage1-new 5e-5 \
  --lr-stage1-text-projection 5e-6 \
  --lr-stage2-new 2e-5 \
  --lr-stage2-text-projection 3e-6 \
  --lr-stage2-text-top 1e-6 \
  --weight-decay 0.01 \
  --wandb-mode disabled \
  --run-clip \
  --run-regress \
  --run-predict-test \
  --run-predict-out \
  > cv_mm_merged_staged_v1.log 2>&1 &

```

This script will:
- patch `clip_train.yml`, `regress_train.yml`, and `model/clip.yml` temporarily
- run `clip_run.py`
- run `regress_run.py`
- run `regress_predict.py` on held-out `addH` test
- compute MAE / RMSE / R² on the held-out test
- only then run prediction on `addH-out`
- save merged `addH-out` predictions and top-ranked candidates

## Important practical notes

- `clip_run.py` and `regress_run.py` in `multi-view` read hard-coded YAML file names from the repo root, so the runner temporarily overwrites these YAMLs and restores them afterward.
- `model/clip.yml` should usually keep `Path.pretrain_ckpt: 'roberta-base'`.
- Your environment must already be able to load `roberta-base` locally/offline if the machine cannot reach Hugging Face.
- FAIR-Chem UMA models usually require `fairchem-core` and model access / cache setup.


nohup bash -lc '
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export GPU_ID=3
export CUDA_VISIBLE_DEVICES=3
export EXPS_TO_RUN=m18_t7_ssl_graphonly_u2_wp05
export SKIP_COMPLETED=1
export FORCE_RERUN=0
export RUN_FINAL_ALL=0
export USE_REPO_LOCK=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
bash run_addh_model_experiment_grid_v2_full.sh
' > run_m18_t7_graphonly_newnode.nohup.log 2>&1 &
