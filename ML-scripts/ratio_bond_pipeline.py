#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ratio_bond_pipeline.py

End-to-end pipeline to predict "Ratio of Bond (O→M)" from CIF structures using ML.

Features:
- Parse CIFs with pymatgen to compute structure-aware descriptors and composition stats
- Merge with labels from an Excel/CSV (target default: "Ratio of Bond (O→M)")
- Train multiple regressors with cross-validated hyperparameter search
- Define a ZnO-related test set via element filter or regex on keys
- Maximize test R² via model selection (choose best on CV, then report Test R²)
- Save: features.csv, best_model.joblib, scaler.joblib, columns.json, and plots

Usage examples:

1) Build features + Train (Zn-containing as test set by default):
   python ratio_bond_pipeline.py \
       featurize-train \
       --excel /mnt/data/RaBiE-ML-with-ZnO.xlsx \
       --excel-sheet 0 \
       --key-col basename \
       --target "Ratio of Bond (O→M)" \
       --cif-glob "/mnt/data/*.cif" \
       --outdir /mnt/data/rabie_out \
       --test-filter-elements Zn \
       --test-filter-mode elements \
       --cv-folds 5

2) Predict for new CIFs:
   python ratio_bond_pipeline.py \
       predict \
       --model-dir /mnt/data/rabie_out \
       --cif-paths /mnt/data/ZnO-Au-out.cif /mnt/data/353-011-Au-out.cif

Notes:
- Key matching: by default we use CIF basename (without extension) as 'key'. Ensure Excel has a column that can match it,
  or use --key-col to point to such a column. If your Excel has a 'cif' column with filenames, set --key-col cif and
  --key-normalize basename to normalize filenames to basenames.
- If pymatgen is not installed: pip install pymatgen
"""

import os, sys, re, json, math, argparse, warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pymatgen imports
try:
    from pymatgen.core import Structure, Element
    from pymatgen.analysis.local_env import CrystalNN
except Exception as e:
    raise SystemExit(
        "pymatgen is required. Please install it first: pip install pymatgen\n"
        f"Original error: {e}"
    )


# -------------------------------
# Utilities
# -------------------------------

def _resolve_cif_path_from_key(key: str, cif_root: str) -> Optional[str]:
    """
    Try to map a key (e.g., basename) to a CIF file under cif_root.
    Resolution order:
      1) exact match: cif_root/key
      2) append .cif: cif_root/key + ".cif"
      3) case-insensitive search by basename without ext
    Return absolute path or None if not found.
    """
    from glob import glob as _glob
    root = os.path.abspath(cif_root)
    cand1 = os.path.join(root, key)
    if os.path.isfile(cand1):
        return cand1
    if not cand1.lower().endswith(".cif"):
        cand2 = cand1 + ".cif"
        if os.path.isfile(cand2):
            return cand2
    # fallback: search
    base = key
    if "." in base:
        base = base[: base.rfind(".")]
    patt = os.path.join(root, "**", "*.cif")
    for fp in _glob(patt, recursive=True):
        b = os.path.basename(fp)
        b0 = b[: b.rfind(".")] if "." in b else b
        if b0.lower() == base.lower():
            return os.path.abspath(fp)
    return None


def _basename_no_ext(path: str) -> str:
    b = os.path.basename(path)
    if "." in b:
        return b[: b.rfind(".")]
    return b

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def _make_outdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# -------------------------------
# Feature Engineering
# -------------------------------

@dataclass
class FeatureConfig:
    neighbor_cutoff: float = 3.0  # initial distance threshold for O-M neigh stats if CrystalNN fails
    use_crystalnn: bool = True

def _composition_features(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    feats = {}
    # element fractions
    for el, frac in comp.items():
        el_str = str(el)
        feats[f"frac_{el_str}"] = float(frac)

    # composition stats based on elemental properties
    els = [el for el, _ in comp.items()]
    if not els:
        return feats

    def el_prop(el: Element, attr: str, default=np.nan):
        try:
            return getattr(el, attr)
        except Exception:
            return default

    props = {
        "Z": [el.number for el in els],
        "nelectrons": [el.Z for el in els],  # same as number
        "mendeleev_no": [el.mendeleev_no for el in els],
        "atomic_mass": [float(el.atomic_mass) if el.atomic_mass else np.nan for el in els],
        "atomic_radius": [el.atomic_radius if el.atomic_radius else np.nan for el in els],
        "en_pauling": [el.X if el.X else np.nan for el in els],
        "row": [el.row for el in els],
        "group": [el.group for el in els],
        "block_idx": [ {"s":0,"p":1,"d":2,"f":3}.get(el.block, np.nan) for el in els ],
    }

    weights = [float(comp[el]) for el in els]
    w = np.array(weights, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)

    for name, arr in props.items():
        arr = np.array(arr, dtype=float)
        wmean = np.nansum(w * arr)
        wstd  = np.sqrt(np.nansum(w * (arr - wmean) ** 2))
        feats[f"wmean_{name}"] = float(wmean)
        feats[f"wstd_{name}"]  = float(wstd)
    return feats

def _global_structure_features(struct: Structure) -> Dict[str, float]:
    feats = {}
    try:
        feats["n_sites"] = float(len(struct))
        feats["volume"] = float(struct.lattice.volume)
        feats["density"] = float(struct.density)
        feats["vol_per_atom"] = feats["volume"] / feats["n_sites"] if feats["n_sites"] > 0 else np.nan
        feats["lattice_a"] = float(struct.lattice.a)
        feats["lattice_b"] = float(struct.lattice.b)
        feats["lattice_c"] = float(struct.lattice.c)
        feats["lattice_alpha"] = float(struct.lattice.alpha)
        feats["lattice_beta"]  = float(struct.lattice.beta)
        feats["lattice_gamma"] = float(struct.lattice.gamma)
    except Exception:
        pass
    return feats

def _om_neighbor_features(struct: Structure, cfg: FeatureConfig) -> Dict[str, float]:
    """
    Compute O->M (non-oxygen element) neighbor statistics:
    - For each O site, find metal neighbors (non-O) using CrystalNN (preferred) or distance cutoff
    - Collect all O-M bond lengths
    - Summarize: count, min, max, mean, std, 25/50/75 percentiles
    Also compute coordination number stats of oxygen atoms.
    """
    feats = {
        "OM_n_bonds": 0.0,
        "OM_len_min": np.nan,
        "OM_len_max": np.nan,
        "OM_len_mean": np.nan,
        "OM_len_std": np.nan,
        "OM_len_p25": np.nan,
        "OM_len_p50": np.nan,
        "OM_len_p75": np.nan,
        "O_coord_mean": np.nan,
        "O_coord_std": np.nan,
        "O_frac": 0.0,
    }
    try:
        cnn = CrystalNN() if cfg.use_crystalnn else None
    except Exception:
        cnn = None

    om_lengths = []
    o_coords = []
    n_sites = len(struct)

    # fraction of oxygen atoms
    o_count = sum(1 for s in struct if s.specie.symbol == "O")
    feats["O_frac"] = float(o_count) / float(n_sites) if n_sites > 0 else 0.0

    for i, site in enumerate(struct.sites):
        if site.specie.symbol != "O":
            continue
        # get neighbors
        neighbors = []
        if cnn is not None:
            try:
                cn_list = cnn.get_nn_info(struct, i)
                # neighbors as (index, distance)
                for entry in cn_list:
                    j = entry["site_index"]
                    sp = struct[j].specie
                    if sp.symbol != "O":
                        dist = struct.get_distance(i, j)
                        neighbors.append((j, float(dist)))
            except Exception:
                neighbors = []  # fallback below

        if not neighbors:
            # distance-based fallback
            for j, nb_site in enumerate(struct.sites):
                if j == i:
                    continue
                sp = nb_site.specie
                if sp.symbol == "O":
                    continue
                d = struct.get_distance(i, j)
                if d <= cfg.neighbor_cutoff:
                    neighbors.append((j, float(d)))

        o_coords.append(float(len(neighbors)))
        for (_, d) in neighbors:
            om_lengths.append(d)

    if om_lengths:
        arr = np.array(om_lengths, dtype=float)
        feats["OM_n_bonds"] = float(len(arr))
        feats["OM_len_min"] = float(np.min(arr))
        feats["OM_len_max"] = float(np.max(arr))
        feats["OM_len_mean"] = float(np.mean(arr))
        feats["OM_len_std"] = float(np.std(arr))
        feats["OM_len_p25"] = float(np.percentile(arr, 25))
        feats["OM_len_p50"] = float(np.percentile(arr, 50))
        feats["OM_len_p75"] = float(np.percentile(arr, 75))

    if o_coords:
        o_arr = np.array(o_coords, dtype=float)
        feats["O_coord_mean"] = float(np.mean(o_arr))
        feats["O_coord_std"]  = float(np.std(o_arr))
    return feats

def featurize_cif(path: str, cfg: FeatureConfig) -> Dict[str, float]:
    struct = Structure.from_file(path)
    feats = {}
    feats["key"] = _basename_no_ext(path)
    feats["path"] = os.path.abspath(path)

    feats.update(_composition_features(struct))
    feats.update(_global_structure_features(struct))
    feats.update(_om_neighbor_features(struct, cfg))

    # Simple electronegativity difference between average metal and oxygen (if present)
    try:
        comp = struct.composition.fractional_composition
        els = [el for el in comp.keys()]
        o_frac = float(comp.get(Element("O"), 0.0))
        metals = [el for el in els if el.symbol != "O"]
        if metals:
            en_metal = np.nanmean([el.X for el in metals if el.X is not None])
        else:
            en_metal = np.nan
        en_ox = Element("O").X
        feats["en_diff_m_ox"] = float(en_metal - en_ox) if (en_metal is not None and en_ox is not None) else np.nan
    except Exception:
        feats["en_diff_m_ox"] = np.nan

    return feats


# -------------------------------
# Data IO & Merge
# -------------------------------

def load_labels_table(excel_path: str, sheet: Optional[str|int] = None) -> pd.DataFrame:
    if excel_path.lower().endswith(".csv"):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path, sheet_name=sheet)
    # normalize col names
    df.columns = [str(c).strip() for c in df.columns]
    return df

def normalize_key_col(df: pd.DataFrame, key_col: str, mode: str) -> pd.DataFrame:
    if mode == "basename":
        df[key_col] = df[key_col].astype(str).apply(lambda s: _basename_no_ext(s))
    elif mode == "lower":
        df[key_col] = df[key_col].astype(str).str.lower()
    elif mode == "strip":
        df[key_col] = df[key_col].astype(str).str.strip()
    else:
        pass
    return df

def build_feature_table(cif_paths: List[str], cfg: FeatureConfig) -> pd.DataFrame:
    rows = []
    for p in cif_paths:
        try:
            row = featurize_cif(p, cfg)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}", file=sys.stderr)
    df = pd.DataFrame(rows)
    return df


# -------------------------------
# Train & Evaluate
# -------------------------------

def build_models_and_grids(random_state: int = 42):
    models = {
        "ETR": ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
        "RFR": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GBR": GradientBoostingRegressor(random_state=random_state),
        "HGBR": HistGradientBoostingRegressor(random_state=random_state),
        "KRR": KernelRidge(),
        "ENet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 2, 30), max_iter=20000, cv=5, n_jobs=None),
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 3, 50), cv=5),
        "Huber": HuberRegressor(),
    }
    grids = {
        "ETR": {"etr__n_estimators": [400, 800, 1200], "etr__max_depth": [None, 12, 20]},
        "RFR": {"rfr__n_estimators": [400, 800, 1200], "rfr__max_depth": [None, 12, 20]},
        "GBR": {"gbr__n_estimators": [400, 800], "gbr__learning_rate": [0.03, 0.06], "gbr__max_depth": [2, 3]},
        "HGBR": {"hgbr__max_iter": [800, 1200], "hgbr__learning_rate": [0.03, 0.06], "hgbr__max_depth": [None, 8]},
        "KRR": {"krr__alpha": np.logspace(-4, 2, 10), "krr__kernel": ["rbf", "laplacian"], "krr__gamma": np.logspace(-3, 1, 8)},
        # ENet/RidgeCV/Huber are internally CV'd or robust; no GridSearch wrapper needed for them
    }
    return models, grids

def pick_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    bad = {"key", "path", target}
    cols = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def element_set_contains_zno(comp_row: Dict[str, float]) -> bool:
    # simple helper: return True if both Zn and O appear with positive fraction
    return (comp_row.get("frac_Zn", 0.0) > 0) and (comp_row.get("frac_O", 0.0) > 0)

def split_train_test(df: pd.DataFrame, mode: str = "regex", test_filter: str = "(?i)zno", include_test_in_train: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train/test split.
    - mode "regex": select rows where 'key' matches regex test_filter as test (default: '(?i)zno')
    - mode "elements": legacy support using element fractions
    If include_test_in_train is True, ZnO-related samples are **also** merged into training,
    but we **keep** the test set for evaluation (i.e., training和测试均包含ZnO样本，存在信息泄漏，请谨慎解读R²)。 
    """
    if mode == "regex":
        pat = re.compile(test_filter)
        mask = df["key"].astype(str).apply(lambda s: bool(pat.search(s)))
    elif mode == "elements":
        req = [x.strip() for x in test_filter.split(",") if x.strip()]
        def is_test_row(row):
            for el in req:
                if el == "O":
                    if row.get("frac_O", 0.0) <= 0:
                        return False
                else:
                    if row.get(f"frac_{el}", 0.0) <= 0:
                        return False
            return True
        mask = df.apply(is_test_row, axis=1)
    else:
        raise ValueError("Unknown test-filter-mode. Use 'regex' or 'elements'.")

    df_test = df[mask].copy()
    df_train = df[~mask].copy()

    if include_test_in_train:
        # Merge ZnO (test) into train but DO NOT clear df_test.
        df_train = pd.concat([df_train, df_test], axis=0).drop_duplicates(subset=["key"]).reset_index(drop=True)

    return df_train, df_test

def train_and_select(X: np.ndarray, y: np.ndarray, feature_names: List[str], outdir: str, cv_folds: int = 5, random_state: int = 42):
    models, grids = build_models_and_grids(random_state=random_state)

    results = []
    best_estimator = None
    best_cv_score = -np.inf
    best_name = None

    def save_cv_plot(name, y_true, y_pred, fold_idx):
        # keep simple; mainly final test plot is needed
        pass

    for name, base_model in models.items():
        if name in grids:
            steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
            key = name.lower()
            steps.append((key, base_model))
            pipe = Pipeline(steps)

            param_grid = grids[name]
            gs = GridSearchCV(pipe, param_grid, scoring="r2", cv=cv_folds, n_jobs=-1, refit=True, verbose=0)
            gs.fit(X, y)
            mean_cv = gs.best_score_
            est = gs.best_estimator_
        else:
            # For models internally doing CV (ElasticNetCV, RidgeCV) or robust ones
            steps = [("scaler", StandardScaler(with_mean=True, with_std=True)), (name.lower(), base_model)]
            pipe = Pipeline(steps)
            pipe.fit(X, y)  # not strictly CV but OK
            # quick KFold to estimate R2
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores = []
            for tr, va in kf.split(X):
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[va])
                scores.append(r2_score(y[va], pred))
            mean_cv = float(np.mean(scores))
            est = pipe

        results.append({"model": name, "cv_r2": float(mean_cv)})
        if mean_cv > best_cv_score:
            best_cv_score = mean_cv
            best_estimator = est
            best_name = name

    res_df = pd.DataFrame(results).sort_values("cv_r2", ascending=False)
    res_df.to_csv(os.path.join(outdir, "cv_results.csv"), index=False)
    return best_name, best_estimator, res_df

def evaluate_and_plot(y_true, y_pred, outpng: str):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)

    plt.figure(figsize=(5,5), dpi=180)
    plt.scatter(y_true, y_pred, s=28, alpha=0.8)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Test R²={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()
    return {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}


# -------------------------------
# CLI Subcommands
# -------------------------------

def cmd_featurize_train(args):
    outdir = _make_outdir(args.outdir)
    cfg = FeatureConfig(neighbor_cutoff=args.neighbor_cutoff, use_crystalnn=(not args.no_crystalnn))

    # Load labels
    labels_df = load_labels_table(args.excel, args.excel_sheet)
    labels_df.columns = [str(c).strip() for c in labels_df.columns]

    # Normalize key column
    key_col = args.key_col
    if args.key_normalize:
        labels_df = normalize_key_col(labels_df, key_col, args.key_normalize)

    if key_col not in labels_df.columns:
        for cand in ["key", "basename", "cif", "file", "name"]:
            if cand in labels_df.columns:
                key_col = cand
                print(f"[WARN] --key-col not found; using '{cand}' in labels.", file=sys.stderr)
                break

    # Keep only rows with non-null target
    if args.target not in labels_df.columns:
        print(f"[ERROR] Target column '{args.target}' not found in labels.", file=sys.stderr)
        sys.exit(4)
    labels_df = labels_df[~labels_df[args.target].isna()].copy()

    # Resolve CIF paths strictly from Excel keys (train only those present in Excel)
    keys = labels_df[key_col].astype(str).tolist()
    cif_paths = []
    resolved_keys = []
    for k in keys:
        rp = _resolve_cif_path_from_key(k, args.cif_root)
        if rp is not None:
            cif_paths.append(rp)
            resolved_keys.append(_basename_no_ext(rp))
        else:
            print(f"[WARN] CIF not found for key '{k}' under root '{args.cif_root}'. Skipped.", file=sys.stderr)

    if len(cif_paths) == 0:
        print("[ERROR] No CIFs resolved from Excel keys. Check --cif-root and --key-col/--key-normalize.", file=sys.stderr)
        sys.exit(2)

    # Featurize only those CIFs
    feat_df = build_feature_table(cif_paths, cfg)
    if feat_df.empty:
        print("[ERROR] Empty feature table.", file=sys.stderr)
        sys.exit(3)

    feat_df.to_csv(os.path.join(outdir, "features_raw.csv"), index=False)

    # Merge: only keys present in Excel with non-null target are considered
    merged = pd.merge(feat_df, labels_df, left_on="key", right_on=key_col, how="inner")
    merged.to_csv(os.path.join(outdir, "features_merged.csv"), index=False)

    # Pick features and target
    fcols = pick_feature_columns(merged, args.target)
    X_all = merged[fcols].values.astype(float)
    y_all = merged[args.target].values.astype(float)

    # Train/Test split: default by regex '(?i)zno' on key; option to include ZnO into train
    df_all = merged[["key", args.target] + fcols].copy()
    df_train, df_test = split_train_test(
        df_all,
        mode=args.test_filter_mode,
        test_filter=args.test_filter,
        include_test_in_train=args.include_zno_in_train
    )

    if len(df_train) == 0:
        print("[ERROR] Training set is empty after filtering.", file=sys.stderr)
        sys.exit(5)

    X_train = df_train[fcols].values.astype(float)
    y_train = df_train[args.target].values.astype(float)

    best_name, best_estimator, cv_df = train_and_select(X_train, y_train, fcols, outdir, cv_folds=args.cv_folds, random_state=args.random_state)
    print(f"[Best] {best_name} | CV R²={cv_df.iloc[0]['cv_r2']:.4f}")

    metrics = {}
    if len(df_test) > 0:
        X_test = df_test[fcols].values.astype(float)
        y_test = df_test[args.target].values.astype(float)
        yhat_test = best_estimator.predict(X_test)
        metrics = evaluate_and_plot(y_test, yhat_test, os.path.join(outdir, "test_scatter.png"))
        out_pred = df_test[["key", args.target]].copy()
        out_pred["yhat"] = yhat_test
        out_pred.to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)
        print(f"[Test] R²={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")
    else:
        print("[Info] ZnO-related samples included in training or no ZnO keys found; skipping test evaluation.")

    # Save artifacts
    joblib.dump(best_estimator, os.path.join(outdir, "best_model.joblib"))
    with open(os.path.join(outdir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": fcols, "target": args.target}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(outdir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}
")
        if metrics:
            f.write(f"Test R2: {metrics['r2']:.6f}
MAE: {metrics['mae']:.6f}
RMSE: {metrics['rmse']:.6f}
")
        f.write(f"Feature count: {len(fcols)}
")
        f.write("Files:
- features_raw.csv
- features_merged.csv
- cv_results.csv
- test_predictions.csv (if any)
- test_scatter.png (if any)
- best_model.joblib
- feature_columns.json
")

def _featurize_only_single(path: str, cfg: FeatureConfig) -> pd.DataFrame:
(path: str, cfg: FeatureConfig) -> pd.DataFrame:
    row = featurize_cif(path, cfg)
    df = pd.DataFrame([row])
    return df

def cmd_predict(args):
    model_dir = args.model_dir
    model_path = os.path.join(model_dir, "best_model.joblib")
    cols_path  = os.path.join(model_dir, "feature_columns.json")
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(cols_path), f"Feature column spec not found: {cols_path}"

    with open(cols_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fcols = meta["feature_columns"]

    cfg = FeatureConfig(neighbor_cutoff=args.neighbor_cutoff, use_crystalnn=(not args.no_crystalnn))

    rows = []
    for p in args.cif_paths:
        df = _featurize_only_single(p, cfg)
        # Ensure all columns present
        for c in fcols:
            if c not in df.columns:
                df[c] = np.nan
        X = df[fcols].values.astype(float)
        model = joblib.load(model_path)
        yhat = model.predict(X)
        rows.append({"key": df["key"].iloc[0], "path": df["path"].iloc[0], "yhat_ratio_O_to_M": float(yhat[0])})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(model_dir, "predictions_new_cifs.csv"), index=False)
    print(out.to_string(index=False))


# -------------------------------
# Main
# -------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Predict 'Ratio of Bond (O→M)' from CIFs via ML.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # featurize-train
    p_ft = sub.add_parser("featurize-train", help="Featurize CIFs, merge with labels, train & evaluate.")
    p_ft.add_argument("--excel", type=str, required=True, help="Excel/CSV file containing labels.")
    p_ft.add_argument("--excel-sheet", type=str, default=None, help="Sheet name or index for Excel.")
    p_ft.add_argument("--key-col", type=str, default="basename", help="Column in labels to join with CIF key (default CIF basename).")
    p_ft.add_argument("--key-normalize", type=str, default="basename", choices=["basename","lower","strip",None], help="Optional normalization for label keys.")
    p_ft.add_argument("--target", type=str, default="Ratio of Bond (O→M)", help="Target column in the labels table.")
    p_ft.add_argument("--cif-root", type=str, default="ML-vac-full-cif", help="Root directory containing CIF files. Only CIFs appearing in Excel are used.")
    # Deprecated options kept for backward compatibility but unused in strict Excel-driven mode
    p_ft.add_argument("--cif-glob", type=str, nargs="+", default=["*.cif"], help="(Deprecated in strict mode) CIF glob patterns.")
    p_ft.add_argument("--cif-paths", type=str, nargs="*", default=[], help="(Deprecated in strict mode) Explicit CIF paths to include.")
    p_ft.add_argument("--outdir", type=str, required=True, help="Output directory.")
    p_ft.add_argument("--test-filter-mode", type=str, default="elements", choices=["elements","regex"], help="How to build test set.")
    p_ft.add_argument("--test-filter", type=str, default="Zn,O", help="Filter (ZnO-related by default).")
    p_ft.add_argument("--cv-folds", type=int, default=5)
    p_ft.add_argument("--random-state", type=int, default=42)
    p_ft.add_argument("--neighbor-cutoff", type=float, default=3.0, help="Fallback O-M neighbor cutoff if CrystalNN fails.")
    p_ft.add_argument("--no-crystalnn", action="store_true", help="Disable CrystalNN neighbor detection.")
    p_ft.set_defaults(func=cmd_featurize_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict for new CIFs using a trained model.")
    p_pred.add_argument("--model-dir", type=str, required=True, help="Directory containing best_model.joblib and feature_columns.json.")
    p_pred.add_argument("--cif-paths", type=str, nargs="+", required=True, help="CIF files to predict.")
    p_pred.add_argument("--neighbor-cutoff", type=float, default=3.0)
    p_pred.add_argument("--no-crystalnn", action="store_true")
    p_pred.set_defaults(func=cmd_predict)

    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
