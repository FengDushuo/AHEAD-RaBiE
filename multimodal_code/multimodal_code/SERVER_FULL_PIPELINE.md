# AddH Server Full Pipeline

This is the recommended server route for full AddH/AddH-2 training and AddH-out prediction.

## Files

- `run_addh_server_full_pipeline.sh`: full server pipeline controller.
- `submit_addh_server_full_pipeline.slurm`: Slurm submit template.
- `22_fuse_llm_strict_blind_server.py`: final conservative blend of LLM/element predictions and heavy strict-blind graph/multiview predictions.

## Minimal Slurm Run

Submit from `/data/home/terminator/RL/multi-view`:

```bash
cd /data/home/terminator/RL/multi-view

export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export PY_FAIRCHEM=/data/home/terminator/anaconda3/envs/fairchem/bin/python
export FAIRCHEM_MODEL_DIR=/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd
export GPU_ID=2
export CUDA_VISIBLE_DEVICES=2

sbatch submit_addh_server_full_pipeline.slurm
```

Use GPU 3 instead:

```bash
export GPU_ID=3
export CUDA_VISIBLE_DEVICES=3
sbatch submit_addh_server_full_pipeline.slurm
```

The FAIR-Chem model directory must contain:

- `checkpoint.pt`
- `config.yml`

## Run With SCNet DeepSeek-V4-Pro Priors

Do not write the API key into any script. Set it only in the environment:

```bash
cd /data/home/terminator/RL/multi-view

export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export PY_FAIRCHEM=/data/home/terminator/anaconda3/envs/fairchem/bin/python
export FAIRCHEM_MODEL_DIR=/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd
export GPU_ID=2
export CUDA_VISIBLE_DEVICES=2
export SCNET_MODEL=DeepSeek-V4-Pro
export RUN_SCNET_LLM=1
read -s -p "SCNET API key: " SCNET_API_KEY; export SCNET_API_KEY; echo

sbatch submit_addh_server_full_pipeline.slurm
```

The SCNet call uses:

- `--prompt-style compact`
- `--disable-thinking`
- `--no-response-format`

This is the stable setting verified locally for DeepSeek-V4-Pro.

## Useful Fast/Debug Runs

Only rebuild LLM/element route, no heavy graph/multiview:

```bash
RUN_GRAPH_GRID=0 RUN_MULTIVIEW_GRID=0 RUN_STRICT_BLIND=0 \
RUN_SERVER_FINAL_BLEND=0 bash run_addh_server_full_pipeline.sh
```

Reuse existing master tables and embeddings:

```bash
RUN_DATA_PREP=0 RUN_EMBEDDINGS=0 bash run_addh_server_full_pipeline.sh
```

Run graph-only heavy training and skip multimodal:

```bash
RUN_MULTIVIEW_GRID=0 bash run_addh_server_full_pipeline.sh
```

Run a complete post-hoc audit after prediction:

```bash
RUN_AUDIT=1 bash run_addh_server_full_pipeline.sh
```

Audit labels are not used for training, model selection, or final blending.

## Important Environment Variables

- `ROOT`: repository root. Defaults to current directory.
- `PY_MM`: multiview Python interpreter. Use `/data/home/terminator/anaconda3/envs/multiview/bin/python`.
- `PY_FAIRCHEM`: FAIR-Chem Python interpreter. Use `/data/home/terminator/anaconda3/envs/fairchem/bin/python`.
- `FAIRCHEM_MODEL_DIR`: default `/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd`.
- `GPU_ID`: default `2`; use `3` if GPU 2 is busy.
- `CUDA_VISIBLE_DEVICES`: default follows `GPU_ID`.
- `RUN_ID`: run name. Defaults to timestamp or Slurm job id.
- `RUN_ROOT`: output root. Defaults to `runs_addh_server/$RUN_ID`.
- `ADDH_DIR`: default `addH`.
- `ADDH2_DIR`: default `addH-2`.
- `ADDHOUT_DIR`: default `addH-out`.
- `ADDHOUT_EXCEL`: default `addH-out/氢吸附能.xlsx`.
- `ADDH2_BASE_MILLER_MAP`: default `2542=100,2858=111,643=111`.
- `ADDHOUT_MILLER_MAP`: default `CeO2=111,ZnO=100`.
- `SEEDS`: default `42,52,62,72,82,92,102,112,122,132`.
- `MULTIVIEW_SCOPE`: `targeted`, `all`, or `off`; default `targeted`.
- `RUN_SCNET_LLM`: `1` to call SCNet, `0` to reuse existing priors or deterministic priors.

## Final Outputs

Main strict final:

- `runs_addh_server/<RUN_ID>/outputs_addh_strict_blind_final/strict_blind_strategy_ensemble_predictions.csv`

LLM/element final:

- `runs_addh_server/<RUN_ID>/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv`

Recommended final server blend:

- `runs_addh_server/<RUN_ID>/outputs_addh_server_final_blend/server_final_addhout_predictions.csv`
- `runs_addh_server/<RUN_ID>/outputs_addh_server_final_blend/server_final_addhout_predictions.xlsx`

Final prediction column:

- `pred_addh_server_final_blend`

The final blend automatically ignores heavy strict-blind predictions if their mean shift from the clean source training target distribution is too large.
