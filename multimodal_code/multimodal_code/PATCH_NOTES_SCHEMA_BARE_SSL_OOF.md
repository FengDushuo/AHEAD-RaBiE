# Patch notes: schema + bare graph + addH-out SSL + strict OOF

This patched pipeline implements four accuracy-oriented changes:

1. **Schema correction for addH/addH-2**
   - `build_addh_master_from_ml2_layout.py` now writes `family_base_miller` correctly, e.g. `2542-100`, `2858-111`, `643-111`.
   - `01_build_addh_master_with_outlier_drop.py` and `build_addh_master_from_ml2_layout.py` now also write bare-slab paths:
     - `bare_contcar_path`
     - `bare_vasprun_path`
     - `bare_oszicar_path`
   - These fields are used to build energy-difference-consistent graph embeddings.

2. **Bare slab information in graph embeddings**
   - New helper: `06_build_dual_graph_embeddings.py`.
   - It builds:
     ```text
     eq_emb_dual = concat(addH_emb, bare_emb, addH_emb - bare_emb)
     ```
   - This aligns the model input with the target definition `E(addH) - E(slab) - E(H)`.

3. **addH-out target-domain graph-text self-supervised alignment**
   - `04_make_multiview_data_cv_multimodal.py` now supports:
     ```bash
     --include-addhout-in-clip
     ```
   - With this flag, addH-out rows are appended only to `clip_train.pkl`, with `target` overwritten to `0.0`.
   - They are **not** included in regression training, validation, or test labels.

4. **Strict group OOF + metadata preservation for fusion**
   - `04_make_multiview_data_cv_multimodal.py` now preserves optional columns such as:
     - `text_structured`, `text_raw`
     - `family_base`, `family_base_miller`, `miller`, `dopant`
     - `site_type`, `anchor_count`, `slab_formula`
     - `data_source`, `w_domain`, `outlier_flag_target`
   - This makes `--concat-text-cols` and `--sample-weight-col w_domain` effective in the staged regression script.

Other fix:
- `make_target_domain_weighted_train_table_mild.py` fixed a file-opening typo: `p.open("rb", "rb")` -> `p.open("rb")`.

Recommended entry point:

```bash
bash RUNME_addH_full_multimodal_route.sh
```

Before running, set these environment variables if needed:

```bash
export ADDH_ROOT=/path/to/addH
export ADDH2_ROOT=/path/to/addH-2
export ADDHOUT_ROOT=/path/to/addH-out
export MODEL_DIR=/path/to/equiformer_v2_31m_allmd
export REPO_ROOT=/path/to/multi-view
export PYTHON=python
export DEVICE=cuda
```
