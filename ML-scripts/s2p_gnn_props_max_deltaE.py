#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s2p_gnn_props_deltaE.py
Two-CIF-dir joint featurization (Esurf from --full-dir, Evac from --out-dir) → predict ΔE with GPU & parallel support.

主要特性：
- F_*/O_* 两套特征 + D_* = F_ - O_ 差分特征
- 目标列可直接用 Excel 的 “ΔE/Delta E”，或由 Esurf-Evac 即算
- 仅使用 Excel 中目标非空 + 两侧均匹配到 CIF 的样本
- 测试集：名字包含 --test-id-contains（默认 "ZnO"）
- 可选把 ZnO 也并入训练 (--also-train-on-test)，注意可能信息泄漏
- 并行提特征 (--workers)，可选 GPU (--use-gpu)
- 关键修复：path→sid 映射改为绝对路径，并提供 realpath/basename 兜底；重复 sid 聚合为单行

依赖：
  pip install numpy pandas scikit-learn pymatgen matplotlib openpyxl
可选 GPU：
  pip install xgboost lightgbm catboost
"""

import os, re, sys, json, math, argparse, random, warnings, concurrent.futures
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.impute import SimpleImputer
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="No _symmetry_equiv_pos_as_xyz")
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")
warnings.filterwarnings("ignore", message="No oxidation states specified on sites")
warnings.filterwarnings("ignore", message="CrystalNN: cannot locate an appropriate radius")

# ---------- Pymatgen ----------
try:
    from pymatgen.core import Structure, Element
    from pymatgen.analysis.local_env import CrystalNN
except Exception as e:
    raise SystemExit("pymatgen is required. `pip install pymatgen`\n" + str(e))

# ---------- Optional GPU ----------
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

TARGET_DEFAULT = ["Delta E"]  # 如果 Excel 中没有该列，将用 Esurf - Evac 即算
DEF_ESURF = "Esurf"
DEF_EVAC  = "Evac"

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)

def is_cif(path: str) -> bool:
    return path.lower().endswith(".cif")

def parse_suffixes(s: Optional[str]) -> list:
    if s is None: return []
    s = str(s).strip()
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def normalize_id(name: str, strip_suffixes: list) -> str:
    base = os.path.splitext(str(name))[0].strip()
    for suf in strip_suffixes:
        if suf and base.endswith(suf):
            base = base[:-len(suf)]
    return base

def _basename_no_ext(path: str) -> str:
    b = os.path.basename(path)
    return b[: b.rfind(".")] if "." in b else b

def _rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def _normalize_sheet_arg(sheet):
    if sheet is None: return 0
    if isinstance(sheet, int): return sheet
    if isinstance(sheet, str):
        s = sheet.strip()
        if s == "" or s.lower() in {"none","na","null"}: return 0
        if s.lstrip("-").isdigit(): return int(s)
        return s
    return sheet

def load_table(path: str, sheet: Optional[str|int]=0) -> pd.DataFrame:
    sheet_norm = _normalize_sheet_arg(sheet)
    if path.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(path, sheet_name=sheet_norm)
    return pd.read_csv(path)

# ---------- Feature Engineering ----------

@dataclass
class FeatureConfig:
    neighbor_cutoff: float = 3.2
    use_crystalnn: bool = True

def _composition_features(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    feats = {}
    for el, frac in comp.items():
        el_str = str(el)
        feats[f"frac_{el_str}"] = float(frac)
    els = [el for el, _ in comp.items()]
    if not els: return feats

    props = {
        "Z": [el.number for el in els],
        "mendeleev_no": [el.mendeleev_no for el in els],
        "atomic_mass": [float(el.atomic_mass) if el.atomic_mass else np.nan for el in els],
        "atomic_radius": [el.atomic_radius if el.atomic_radius else np.nan for el in els],
        "en_pauling": [el.X if el.X else np.nan for el in els],
        "row": [el.row for el in els],
        "group": [el.group for el in els],
        "block_idx": [{"s":0,"p":1,"d":2,"f":3}.get(el.block, np.nan) for el in els],
    }
    weights = [float(comp[el]) for el in els]
    w = np.array(weights, dtype=float); w = w/(w.sum() if w.sum()>0 else 1.0)
    for name, arr in props.items():
        arr = np.array(arr, dtype=float)
        wmean = np.nansum(w * arr)
        wstd  = np.sqrt(np.nansum(w * (arr - wmean)**2))
        feats[f"wmean_{name}"] = float(wmean)
        feats[f"wstd_{name}"]  = float(wstd)
    return feats

def _global_structure_features(struct: Structure) -> Dict[str, float]:
    feats = {}
    try:
        feats["n_sites"] = float(len(struct))
        feats["volume"] = float(struct.lattice.volume)
        feats["density"] = float(struct.density)
        feats["vol_per_atom"] = feats["volume"]/feats["n_sites"] if feats["n_sites"]>0 else np.nan
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
    feats = {
        "OM_n_bonds": 0.0, "OM_len_min": np.nan, "OM_len_max": np.nan, "OM_len_mean": np.nan,
        "OM_len_std": np.nan, "OM_len_p25": np.nan, "OM_len_p50": np.nan, "OM_len_p75": np.nan,
        "O_coord_mean": np.nan, "O_coord_std": np.nan, "O_frac": 0.0,
    }
    try:
        cnn = CrystalNN() if cfg.use_crystalnn else None
    except Exception:
        cnn = None

    om_lengths, o_coords = [], []
    n_sites = len(struct)
    o_count = sum(1 for s in struct if s.specie.symbol == "O")
    feats["O_frac"] = float(o_count)/float(n_sites) if n_sites>0 else 0.0

    for i, site in enumerate(struct.sites):
        if site.specie.symbol != "O": continue
        neighbors = []
        if cnn is not None:
            try:
                cn_list = cnn.get_nn_info(struct, i)
                for entry in cn_list:
                    j = entry["site_index"]; sp = struct[j].specie
                    if sp.symbol != "O":
                        dist = struct.get_distance(i, j)
                        neighbors.append((j, float(dist)))
            except Exception:
                neighbors = []
        if not neighbors:
            for j, nb_site in enumerate(struct.sites):
                if j == i: continue
                if nb_site.specie.symbol == "O": continue
                d = struct.get_distance(i, j)
                if d <= cfg.neighbor_cutoff:
                    neighbors.append((j, float(d)))
        o_coords.append(float(len(neighbors)))
        for _, d in neighbors: om_lengths.append(d)

    if om_lengths:
        arr = np.array(om_lengths, dtype=float)
        feats.update({
            "OM_n_bonds": float(len(arr)),
            "OM_len_min": float(np.min(arr)),
            "OM_len_max": float(np.max(arr)),
            "OM_len_mean": float(np.mean(arr)),
            "OM_len_std": float(np.std(arr)),
            "OM_len_p25": float(np.percentile(arr, 25)),
            "OM_len_p50": float(np.percentile(arr, 50)),
            "OM_len_p75": float(np.percentile(arr, 75)),
        })
    if o_coords:
        o_arr = np.array(o_coords, dtype=float)
        feats["O_coord_mean"] = float(np.mean(o_arr))
        feats["O_coord_std"]  = float(np.std(o_arr))
    return feats

def featurize_cif(path: str, cfg: FeatureConfig) -> Dict[str, float]:
    s = Structure.from_file(path)
    feats = {"key": _basename_no_ext(path), "path": os.path.abspath(path)}
    feats.update(_composition_features(s))
    feats.update(_global_structure_features(s))
    feats.update(_om_neighbor_features(s, cfg))
    try:
        comp = s.composition.fractional_composition
        els = [el for el in comp.keys()]
        metals = [el for el in els if el.symbol != "O"]
        en_m = np.nanmean([el.X for el in metals if el.X is not None]) if metals else np.nan
        en_o = Element("O").X
        feats["en_diff_m_ox"] = float(en_m - en_o) if (en_m is not None and en_o is not None) else np.nan
    except Exception:
        feats["en_diff_m_ox"] = np.nan
    return feats

# ---------- Top-level worker for multiprocessing ----------
def _featurize_worker(args):
    path, cfg_dict = args
    cfg = FeatureConfig(
        neighbor_cutoff=cfg_dict.get("neighbor_cutoff", 3.2),
        use_crystalnn=cfg_dict.get("use_crystalnn", True),
    )
    try:
        return featurize_cif(path, cfg)
    except Exception as e:
        print(f"[WARN] featurize failed: {path}: {e}", file=sys.stderr)
        return None

def build_feature_table(paths: List[str], cfg: FeatureConfig, workers: int = 0) -> pd.DataFrame:
    rows = []
    if workers and workers > 0:
        cfg_dict = {"neighbor_cutoff": cfg.neighbor_cutoff, "use_crystalnn": cfg.use_crystalnn}
        tasks = [(p, cfg_dict) for p in paths]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_featurize_worker, tasks, chunksize=4):
                if r is not None: rows.append(r)
    else:
        for p in paths:
            try: rows.append(featurize_cif(p, cfg))
            except Exception as e: print(f"[WARN] featurize failed: {p}: {e}", file=sys.stderr)
    return pd.DataFrame(rows)

# ---------- Matching ----------
def _build_file_index(in_dir: str, fname_strip_suffixes: str) -> Dict[str,str]:
    files = [f for f in os.listdir(in_dir) if is_cif(f)]
    f_suffixes = parse_suffixes(fname_strip_suffixes)
    fmap: Dict[str,str] = {}
    dup = set()
    for f in files:
        nid = normalize_id(f, f_suffixes)
        if nid in fmap: dup.add(nid)
        else: fmap[nid] = os.path.join(in_dir, f)  # 注意：相对路径
    if dup: print(f"[WARN] {len(dup)} normalized id collisions (e.g. {list(dup)[:3]})")
    return fmap

def _match_one(fmap: Dict[str,str], sid: str, mode: str) -> Optional[str]:
    sid_l = sid.lower(); keys = list(fmap.keys()); keys_l = [k.lower() for k in keys]
    if mode == "exact":  return fmap.get(sid, None)
    if mode == "prefix":
        idx = [i for i,k in enumerate(keys_l) if k.startswith(sid_l)]
        if not idx: return None
        if len(idx)==1: return fmap[keys[idx[0]]]
        best = sorted([keys[i] for i in idx], key=len, reverse=True)[0]
        return fmap[best]
    if mode == "contain":
        idx = [i for i,k in enumerate(keys_l) if sid_l in k]
        if not idx: return None
        if len(idx)==1: return fmap[keys[idx[0]]]
        best = sorted([keys[i] for i in idx], key=len, reverse=True)[0]
        return fmap[best]
    raise ValueError("unknown match mode")

def _match_ids_to_two_dirs(sid: str, fmap_full: Dict[str,str], fmap_out: Dict[str,str], mode: str) -> Tuple[Optional[str], Optional[str]]:
    def _auto(fmap, sid):
        p = _match_one(fmap, sid, "exact")
        if p is None: p = _match_one(fmap, sid, "prefix")
        if p is None: p = _match_one(fmap, sid, "contain")
        return p
    if mode=="auto": return _auto(fmap_full, sid), _auto(fmap_out, sid)
    return _match_one(fmap_full, sid, mode), _match_one(fmap_out, sid, mode)

# ---------- Mapping helpers (关键修复：路径归一化 + 兜底) ----------
def _make_path2sid(rows: List[Dict]) -> Dict[str, str]:
    """
    rows: [{"sid":..., "p_full":..., "p_out":...}, ...]  # 可能是相对路径
    返回：以 abs/real 路径为键的映射；同时返回 basename->sid 备用
    """
    p2s_abs = {}
    p2s_real = {}
    b2s = {}
    for r in rows:
        for k in ("p_full", "p_out"):
            p_rel = r[k]
            sid = r["sid"]
            p_abs = os.path.abspath(p_rel)
            p_real = os.path.realpath(p_rel)
            p2s_abs[p_abs] = sid
            p2s_real[p_real] = sid
            b2s[_basename_no_ext(p_rel)] = sid
            b2s[_basename_no_ext(p_abs)] = sid
            b2s[_basename_no_ext(p_real)] = sid
    return p2s_abs, p2s_real, b2s

def _apply_sid_map(feat_df: pd.DataFrame, p2s_abs: Dict[str,str], p2s_real: Dict[str,str], b2s: Dict[str,str], tag: str) -> pd.DataFrame:
    """
    先用 abs 匹配，再 realpath 兜底，最后 basename 兜底。
    打印前后统计和若干样例，便于排查。
    """
    # 先尝试 abs
    path_abs = feat_df["path"]
    sid = path_abs.map(p2s_abs)

    # realpath 兜底
    miss_mask = sid.isna()
    if miss_mask.any():
        rp = path_abs.apply(os.path.realpath)
        sid2 = rp.map(p2s_real)
        sid.loc[miss_mask] = sid2.loc[miss_mask]

    # basename 兜底
    miss_mask = sid.isna()
    if miss_mask.any():
        base = path_abs.apply(_basename_no_ext)
        sid3 = base.map(b2s)
        sid.loc[miss_mask] = sid3.loc[miss_mask]

    feat_df = feat_df.copy()
    feat_df["sid"] = sid

    # 日志
    total = len(feat_df)
    miss = int(feat_df["sid"].isna().sum())
    if miss:
        print(f"[WARN] {tag}: sid mapping missing {miss}/{total}. Examples:")
        ex = feat_df.loc[feat_df["sid"].isna(), ["path"]].head(5)
        print(ex.to_string(index=False))
    else:
        print(f"[OK] {tag}: sid mapping 100% matched ({total}/{total}).")
    # 打印成功映射样例
    print(f"[CHECK] {tag} example mapping:")
    print(feat_df[["path","sid"]].head(3).to_string(index=False))
    return feat_df

# ---------- Modeling ----------
def build_models_and_grids(random_state: int = 42, use_gpu: bool = False):
    models = {
        "ETR": ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
        "RFR": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GBR": GradientBoostingRegressor(random_state=random_state),
        "HGBR": HistGradientBoostingRegressor(random_state=random_state),
        "KRR": KernelRidge(),
        "ENet": ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4,2,30), max_iter=100000, cv=5),
        "RidgeCV": RidgeCV(alphas=np.logspace(-4,3,50), cv=5),
        "Huber": HuberRegressor(max_iter=500),
    }
    if use_gpu and XGBRegressor is not None:
        models["XGB"] = XGBRegressor(
            tree_method="gpu_hist", predictor="gpu_predictor",
            random_state=random_state, n_estimators=900, max_depth=8,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.8
        )
    if use_gpu and LGBMRegressor is not None:
        models["LGBM"] = LGBMRegressor(
            device="gpu", random_state=random_state, n_estimators=1200,
            learning_rate=0.05, num_leaves=127
        )
    if use_gpu and CatBoostRegressor is not None:
        models["CAT"] = CatBoostRegressor(
            task_type="GPU", devices="0", random_seed=random_state,
            depth=8, learning_rate=0.05, iterations=1200, verbose=False
        )
    grids = {
        "ETR": {"etr__n_estimators":[400,800,1200], "etr__max_depth":[None,12,20]},
        "RFR": {"rfr__n_estimators":[400,800,1200], "rfr__max_depth":[None,12,20]},
        "GBR": {"gbr__n_estimators":[400,800], "gbr__learning_rate":[0.03,0.06], "gbr__max_depth":[2,3]},
        "HGBR":{"hgbr__max_iter":[800,1200], "hgbr__learning_rate":[0.03,0.06], "hgbr__max_depth":[None,8]},
        "KRR": {"krr__alpha":np.logspace(-4,2,10), "krr__kernel":["rbf","laplacian"], "krr__gamma":np.logspace(-3,1,8)},
    }
    return models, grids

def train_cv_select(X: np.ndarray, y: np.ndarray, feature_names: List[str], outdir: str,
                    cv_folds: int = 5, random_state: int = 42, use_gpu: bool = False):
    models, grids = build_models_and_grids(random_state=random_state, use_gpu=use_gpu)
    results = []; best_estimator=None; best_cv=-1e9; best_name=None

    def make_steps(model_name: str, base_model):
        return [("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (model_name.lower(), base_model)]

    for name, base_model in models.items():
        steps = make_steps(name, base_model)
        pipe = Pipeline(steps)

        if name in grids:
            gs = GridSearchCV(pipe, grids[name], scoring="r2", cv=cv_folds, n_jobs=-1, refit=True, verbose=0, error_score="raise")
            gs.fit(X, y)
            mean_cv = gs.best_score_; est = gs.best_estimator_
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores=[]
            for tr, va in kf.split(X):
                pipe.fit(X[tr], y[tr]); yp = pipe.predict(X[va])
                scores.append(r2_score(y[va], yp))
            mean_cv = float(np.mean(scores)); est = pipe

        results.append({"model": name, "cv_r2": float(mean_cv)})
        if mean_cv > best_cv:
            best_cv=mean_cv; best_estimator=est; best_name=name

    pd.DataFrame(results).sort_values("cv_r2", ascending=False).to_csv(os.path.join(outdir, "cv_results.csv"), index=False)
    return best_name, best_estimator, results

def evaluate_and_plot(y_true, y_pred, outpng: str, title_prefix="Test"):
    r2 = r2_score(y_true, y_pred); mae = mean_absolute_error(y_true, y_pred); rmse = _rmse(y_true, y_pred)
    plt.figure(figsize=(4.6,4.6), dpi=160)
    plt.scatter(y_true, y_pred, s=18, alpha=0.8)
    lo = float(min(np.min(y_true), np.min(y_pred))); hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo,hi],[lo,hi],'--')
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(f"{title_prefix} R²={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")
    plt.tight_layout(); plt.savefig(outpng); plt.close()
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}

def pick_feature_columns_df(df: pd.DataFrame, target: str, max_nan_frac: float = 0.95) -> List[str]:
    bad = {"key","path","sid", target}
    cols = []
    for c in df.columns:
        if c in bad: continue
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        if float(df[c].isna().mean()) >= max_nan_frac: continue
        cols.append(c)
    return cols

# ---------- Pipeline ----------
def cmd_train(args):
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    target_name = [c.strip() for c in (args.target_cols or ",".join(TARGET_DEFAULT)).split(",")][0]

    # Excel
    df = load_table(args.excel, args.sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if args.id_col not in df.columns:
        raise SystemExit(f"[ERR] id column '{args.id_col}' not in Excel.")
    if target_name not in df.columns:
        if args.esurf_col in df.columns and args.evac_col in df.columns:
            df[target_name] = df[args.esurf_col].astype(float) - df[args.evac_col].astype(float)
            print(f"[INFO] target '{target_name}' derived as {args.esurf_col} - {args.evac_col}")
        else:
            raise SystemExit(f"[ERR] target '{target_name}' not found and cannot derive.")
    if args.skip_na_targets:
        df = df[~pd.isna(df[target_name])].copy()

    # 文件索引
    fmap_full = _build_file_index(args.full_dir, args.full_strip_suffixes)
    fmap_out  = _build_file_index(args.out_dir,  args.out_strip_suffixes)
    e_suffixes = parse_suffixes(args.excel_strip_suffixes)

    # 匹配
    rows = []
    miss = 0
    for _, r in df.iterrows():
        sid_raw = str(r[args.id_col]).strip()
        sid = normalize_id(sid_raw, e_suffixes)
        p_full, p_out = _match_ids_to_two_dirs(sid, fmap_full, fmap_out, args.match_mode)
        if (p_full is None) or (p_out is None):
            miss += 1; continue
        rows.append({"sid": sid, "y": float(r[target_name]), "p_full": p_full, "p_out": p_out})
    if miss: print(f"[WARN] {miss} rows skipped (not matched in both dirs).")
    print(f"[OK] matched {len(rows)} paired samples")
    if len(rows)==0: raise SystemExit("[ERR] no paired samples.")

    # ---------- 关键修复：构建多路映射，并进行三重兜底 ----------
    p2s_abs, p2s_real, b2s = _make_path2sid(rows)

    # 特征（并行）
    cfg = FeatureConfig(neighbor_cutoff=args.neighbor_cutoff, use_crystalnn=(not args.no_crystalnn))
    paths_full = [r["p_full"] for r in rows]
    paths_out  = [r["p_out"]  for r in rows]
    feat_full = build_feature_table(paths_full, cfg, workers=args.workers)
    feat_out  = build_feature_table(paths_out , cfg, workers=args.workers)

    # 应用映射：abs → real → basename
    feat_full = _apply_sid_map(feat_full, p2s_abs, p2s_real, b2s, tag="full")
    feat_out  = _apply_sid_map(feat_out , p2s_abs, p2s_real, b2s, tag="out")

    # 1) 丢弃空 sid
    n_drop_full = int(feat_full["sid"].isna().sum())
    n_drop_out  = int(feat_out ["sid"].isna().sum())
    if n_drop_full or n_drop_out:
        print(f"[WARN] drop rows with empty sid → full:{n_drop_full} out:{n_drop_out}")
    feat_full = feat_full[~feat_full["sid"].isna()].copy()
    feat_out  = feat_out [~feat_out ["sid"].isna()].copy()

    # 2) 重复 sid → 数值列按均值聚合为单行
    def _dedup_by_sid_mean(df_in: pd.DataFrame, tag: str) -> pd.DataFrame:
        if df_in["sid"].nunique() == len(df_in):
            return df_in
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if "sid" not in num_cols:
            num_cols = ["sid"] + num_cols
        before = len(df_in)
        g = df_in[num_cols].groupby("sid", as_index=False).mean(numeric_only=True)
        after = len(g)
        print(f"[WARN] {tag}: duplicated sid detected → {before-after} rows collapsed by mean (unique sid={after})")
        return g

    feat_full = _dedup_by_sid_mean(feat_full, "full")
    feat_out  = _dedup_by_sid_mean(feat_out , "out")

    print(f"[CHECK] full rows={len(feat_full)} unique_sid={feat_full['sid'].nunique()} | "
          f"out rows={len(feat_out)} unique_sid={feat_out['sid'].nunique()}")

    # 仅数值 + 前缀；返回中保留 sid 列
    def numeric_feats(df_in: pd.DataFrame, prefix: str) -> pd.DataFrame:
        keep = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        out = df_in[["sid"] + [c for c in keep if c != "sid"]].copy()
        rename = {c: f"{prefix}{c}" for c in out.columns if c not in {"sid"}}
        out = out.rename(columns=rename)
        return out

    F = numeric_feats(feat_full, "F_")
    O = numeric_feats(feat_out , "O_")

    # 3) 现在按 sid 一对一 inner join
    all_df = pd.merge(F, O, on="sid", how="inner")
    print(f"[CHECK] merged rows={len(all_df)} (should be ~= number of unique sid)")

    # 差分特征
    f_cols = [c for c in all_df.columns if c.startswith("F_")]
    o_cols = [c for c in all_df.columns if c.startswith("O_")]
    commons = []
    for fc in f_cols:
        base = fc[2:]; oc = f"O_{base}"
        if oc in o_cols: commons.append(base)
    for base in commons:
        all_df[f"D_{base}"] = all_df[f"F_{base}"] - all_df[f"O_{base}"]

    # 合并 y
    sid2y = {r["sid"]: r["y"] for r in rows}
    all_df[target_name] = all_df["sid"].map(sid2y)

    # 如果仍是 0 行，直接报错并给诊断
    if len(all_df) == 0:
        print("[ERR] merged feature table is empty. Diagnostics:")
        print(f"  - feat_full rows: {len(feat_full)}")
        print(f"  - feat_out  rows: {len(feat_out)}")
        print("  - Example full paths (after abspath in featurize):")
        print(feat_full[['path','sid']].head(5).to_string(index=False))
        print("  - Example out  paths:")
        print(feat_out[['path','sid']].head(5).to_string(index=False))
        raise SystemExit("No samples left after merging by 'sid'. Check --match-mode / *-strip-suffixes / Excel id.")

    # 导出与 NaN 报告
    ensure_dir(args.save_dir)
    all_df.to_csv(os.path.join(args.save_dir, "features_all.csv"), index=False)
    all_df.isna().mean().sort_values(ascending=False).to_csv(os.path.join(args.save_dir, "nan_ratio_by_feature.csv"))
    print("[INFO] saved NaN report: nan_ratio_by_feature.csv")

    # 训练/验证/测试划分 + 清洗 y
    def _clean_y(df_in: pd.DataFrame) -> pd.DataFrame:
        # 确保目标列一定存在（即使是空表）
        if target_name not in df_in.columns:
            df_in = df_in.copy(); df_in[target_name] = np.nan
        y = df_in[target_name]
        mask = y.notna() & np.isfinite(y.astype(float))
        bad = (~mask).sum()
        if bad:
            ex = df_in.loc[~mask, ["sid", target_name]].head(5)
            print(f"[WARN] drop {bad} rows with invalid y. examples:\n{ex}")
        return df_in[mask].copy()

    ids = all_df["sid"].astype(str).tolist()
    needle = (args.test_id_contains or "").lower()
    mask_te = [needle in s.lower() for s in ids] if needle else [False]*len(ids)
    df_test = _clean_y(all_df[mask_te].copy())
    df_trva = _clean_y(all_df[~np.array(mask_te)].copy())

    if len(df_trva)==0 and not args.also_train_on_test:
        raise RuntimeError("No samples left for train/val after test filter.")
    if len(df_trva)==0 and args.also_train_on_test:
        df_train, df_val = train_test_split(df_test, test_size=args.val_size, random_state=args.seed, shuffle=True)
    else:
        df_train, df_val = train_test_split(df_trva, test_size=args.val_size, random_state=args.seed, shuffle=True)

    if args.also_train_on_test and len(df_test)>0:
        df_train = pd.concat([df_train, df_test], axis=0).drop_duplicates(subset=["sid"]).reset_index(drop=True)
        print(f"[INFO] also_train_on_test=True → train size={len(df_train)} (train + test)")

    print(f"[SPLIT] train={len(df_train)}  val={len(df_val)}  test={len(df_test)} | test filter='{args.test_id_contains}'")

    # 选特征列（基于 train）
    fcols = pick_feature_columns_df(df_train, target_name, args.max_nan_frac)
    with open(os.path.join(args.save_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": fcols, "targets": [target_name]}, f, ensure_ascii=False, indent=2)

    # 训练与选模
    def fit_once(random_state_offset=0):
        X_tr = df_train[fcols].values.astype(float); y_tr = df_train[target_name].values.astype(float)
        best_name, best_est, _ = train_cv_select(
            X_tr, y_tr, fcols, args.save_dir,
            cv_folds=args.cv_folds, random_state=args.seed+random_state_offset,
            use_gpu=args.use_gpu
        )
        print(f"[Best] {best_name}")
        return best_name, best_est

    models = []
    _, est = fit_once(random_state_offset=0)
    models.append(est); joblib.dump(est, os.path.join(args.save_dir, "best_model_1.joblib"))
    for k in range(1, args.ensemble):
        _, est_k = fit_once(random_state_offset=k*17)
        models.append(est_k); joblib.dump(est_k, os.path.join(args.save_dir, f"best_model_{k+1}.joblib"))

    # 评估
    def eval_split(df, tag):
        if len(df)==0:
            print(f"[{tag}] no rows."); return
        # 缺列补 NaN（管道里有 imputer）
        for c in fcols:
            if c not in df.columns: df[c] = np.nan
        X = df[fcols].values.astype(float); y = df[target_name].values.astype(float)
        yh_stack = [m.predict(X) for m in models]
        yhat = np.mean(np.stack(yh_stack, axis=0), axis=0) if len(models)>1 else yh_stack[0]
        mets = evaluate_and_plot(y, yhat, os.path.join(args.save_dir, f"scatter_{tag}.png"), title_prefix=tag)
        out = df[["sid", target_name]].copy(); out["pred"] = yhat
        out.to_csv(os.path.join(args.save_dir, f"{tag}_preds_vs_true{'_ens' if len(models)>1 else ''}.csv"), index=False)
        print(f"[{tag}] R2={mets['R2']:.4f} RMSE={mets['RMSE']:.4f} MAE={mets['MAE']:.4f}")

    eval_split(df_val,  "val")
    eval_split(df_test, "test")

    joblib.dump(models[0], os.path.join(args.save_dir, "best_model.joblib"))
    print(f"[OK] saved models & artifacts to: {args.save_dir}")

def cmd_predict(args):
    model_path = os.path.join(args.model_dir, "best_model.joblib")
    cols_path  = os.path.join(args.model_dir, "feature_columns.json")
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(cols_path),  f"Feature column spec not found: {cols_path}"
    model = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fcols = meta["feature_columns"]; target_name = meta.get("targets", TARGET_DEFAULT)[0]

    fmap_full = _build_file_index(args.full_dir, args.full_strip_suffixes)
    fmap_out  = _build_file_index(args.out_dir,  args.out_strip_suffixes)
    ids = sorted(list(set(fmap_full.keys()) & set(fmap_out.keys())))
    if len(ids)==0: raise SystemExit("[ERR] no paired ids found in full_dir & out_dir.")

    cfg = FeatureConfig(neighbor_cutoff=args.neighbor_cutoff, use_crystalnn=(not args.no_crystalnn))
    paths_full = [fmap_full[i] for i in ids]; paths_out = [fmap_out[i] for i in ids]
    feat_full = build_feature_table(paths_full, cfg, workers=args.workers)
    feat_out  = build_feature_table(paths_out , cfg, workers=args.workers)

    # 以 abspath/realpath/basename 映射 id，保持与训练一致
    rows = [{"sid": i, "p_full": fmap_full[i], "p_out": fmap_out[i]} for i in ids]
    p2s_abs, p2s_real, b2s = _make_path2sid(rows)
    feat_full = _apply_sid_map(feat_full, p2s_abs, p2s_real, b2s, tag="full(pred)")
    feat_out  = _apply_sid_map(feat_out , p2s_abs, p2s_real, b2s, tag="out(pred)")

    # 丢空、去重
    feat_full = feat_full[~feat_full["sid"].isna()].copy()
    feat_out  = feat_out [~feat_out ["sid"].isna()].copy()

    def _dedup_by_sid_mean(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in["sid"].nunique() == len(df_in):
            return df_in
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if "sid" not in num_cols:
            num_cols = ["sid"] + num_cols
        return df_in[num_cols].groupby("sid", as_index=False).mean(numeric_only=True)

    feat_full = _dedup_by_sid_mean(feat_full)
    feat_out  = _dedup_by_sid_mean(feat_out)

    def numeric_feats(df_in: pd.DataFrame, prefix: str) -> pd.DataFrame:
        keep = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        out = df_in[["sid"] + [c for c in keep if c != "sid"]].copy()
        rename = {c: f"{prefix}{c}" for c in out.columns if c not in {"sid"}}
        out = out.rename(columns=rename)
        return out

    F = numeric_feats(feat_full, "F_"); O = numeric_feats(feat_out, "O_")
    all_df = pd.merge(F, O, on="sid", how="inner")

    f_cols = [c for c in all_df.columns if c.startswith("F_")]
    o_cols = [c for c in all_df.columns if c.startswith("O_")]
    commons = []
    for fc in f_cols:
        base = fc[2:]; oc = f"O_{base}"
        if oc in o_cols: commons.append(base)
    for base in commons:
        all_df[f"D_{base}"] = all_df[f"F_{base}"] - all_df[f"O_{base}"]

    for c in fcols:
        if c not in all_df.columns: all_df[c] = np.nan
    X = all_df[fcols].values.astype(float)
    yhat = model.predict(X)

    out_df = pd.DataFrame({"sid": all_df["sid"], target_name: yhat})
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote predictions: {args.out_csv} ({len(out_df)} rows)")

# ---------- CLI ----------
def build_parser():
    p = argparse.ArgumentParser(description="Two-dir CIF featurization → ΔE (GPU & parallel)")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("train", help="train ΔE model")
    st.add_argument("--full-dir", type=str, required=True)
    st.add_argument("--out-dir",  type=str, required=True)
    st.add_argument("--excel",    type=str, required=True)
    st.add_argument("--sheet",    type=str, default="0")
    st.add_argument("--id-col",   type=str, default="name")
    st.add_argument("--target-cols", type=str, default=",".join(TARGET_DEFAULT))
    st.add_argument("--esurf-col", type=str, default=DEF_ESURF)
    st.add_argument("--evac-col",  type=str, default=DEF_EVAC)
    st.add_argument("--full-strip-suffixes", type=str, nargs="?", const="-out", default="-out")
    st.add_argument("--out-strip-suffixes",  type=str, nargs="?", const="-out", default="-out")
    st.add_argument("--excel-strip-suffixes",type=str, nargs="?", const="", default="")
    st.add_argument("--match-mode", type=str, choices=["auto","exact","prefix","contain"], default="auto")
    st.add_argument("--skip-na-targets", action="store_true", default=True)
    st.add_argument("--neighbor-cutoff", type=float, default=3.2)
    st.add_argument("--no-crystalnn", action="store_true")
    st.add_argument("--save-dir", type=str, required=True)
    st.add_argument("--cv-folds", type=int, default=5)
    st.add_argument("--val-size", type=float, default=0.15)
    st.add_argument("--test-id-contains", type=str, default="ZnO")
    st.add_argument("--also-train-on-test", action="store_true")
    st.add_argument("--seed", type=int, default=2025)
    st.add_argument("--ensemble", type=int, default=1)
    st.add_argument("--max-nan-frac", type=float, default=0.95)
    st.add_argument("--use-gpu", action="store_true")
    st.add_argument("--workers", type=int, default=0)
    st.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict", help="predict ΔE")
    sp.add_argument("--model-dir", type=str, required=True)
    sp.add_argument("--full-dir",  type=str, required=True)
    sp.add_argument("--out-dir",   type=str, required=True)
    sp.add_argument("--out-csv",   type=str, default="preds_deltaE.csv")
    sp.add_argument("--neighbor-cutoff", type=float, default=3.2)
    sp.add_argument("--no-crystalnn", action="store_true")
    sp.add_argument("--full-strip-suffixes", type=str, nargs="?", const="-out", default="-out")
    sp.add_argument("--out-strip-suffixes",  type=str, nargs="?", const="-out", default="-out")
    sp.add_argument("--workers", type=int, default=0)
    sp.set_defaults(func=cmd_predict)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

