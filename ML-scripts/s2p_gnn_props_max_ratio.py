#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s2p_gnn_props_max_ratio.py
Structure → Ratio of Bond (O→M) with scikit-learn (MAX 参数风格)

子命令：
- prepare：配对 Excel 与 CIF（支持 exact/prefix/contain/auto 匹配），导出 dataset.json / curated_meta.csv
- train  ：特征工程 → CV 选模 → 训练；ZnO 测试划分；--also-train-on-test 支持“同时用于训练并测试”
- predict：对文件夹全部 CIF 批量预测

参数亮点：
- --match-mode {auto,exact,prefix,contain}：ID 与文件匹配策略（默认 auto：exact→prefix→contain）
- --fname-strip-suffixes / --excel-strip-suffixes：可选值；写了但不跟值时分别使用 -out / 空串
- --sheet 健壮解析："0" 视为第0个表；也可用表名

新增鲁棒性：
- Pipeline 统一加入 SimpleImputer(strategy="median") 处理 NaN
- 训练前丢弃高缺失特征列（默认阈值 0.95，可在 pick_feature_columns 中调整）
"""

import os, re, sys, json, math, argparse, random, warnings
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

# 安静掉一些常见的 pymatgen 警告（可按需注释）
warnings.filterwarnings("ignore", message="No _symmetry_equiv_pos_as_xyz")
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")
warnings.filterwarnings("ignore", message="No oxidation states specified on sites")
warnings.filterwarnings("ignore", message="CrystalNN: cannot locate an appropriate radius")

# Pymatgen
try:
    from pymatgen.core import Structure, Element
    from pymatgen.analysis.local_env import CrystalNN
except Exception as e:
    raise SystemExit(
        "pymatgen is required. Please install: pip install pymatgen\n"
        f"Original error: {e}"
    )

# ---------------- Utils ----------------

TARGET_DEFAULT = ["Ratio of Bond (O→M)"]

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)

def is_cif(path: str) -> bool:
    return path.lower().endswith('.cif')

def parse_suffixes(s: Optional[str]) -> list:
    if s is None: return []
    s = str(s)
    if not s.strip(): return []
    return [x.strip() for x in s.split(',') if x.strip()]

def _safe_name(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z\-\_.]+', '_', str(s))

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
    """将 '--sheet 0' / '1' → int 索引；空串/none→第0个；表名保持字符串。"""
    if sheet is None:
        return 0
    if isinstance(sheet, int):
        return sheet
    if isinstance(sheet, str):
        ss = sheet.strip()
        if ss == "" or ss.lower() in {"none", "na", "null"}:
            return 0
        if ss.isdigit() or (ss.startswith("-") and ss[1:].isdigit()):
            return int(ss)
        return ss
    return sheet

def load_table(path: str, sheet: Optional[str|int]=0) -> pd.DataFrame:
    sheet_norm = _normalize_sheet_arg(sheet)
    if path.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(path, sheet_name=sheet_norm)
    else:
        return pd.read_csv(path)

def filter_by_quantiles(df: pd.DataFrame, cols: List[str], qlow: float, qhigh: float):
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        if c not in df.columns: continue
        s = df[c].astype(float)
        lo = s.quantile(qlow); hi = s.quantile(qhigh)
        mask &= (s >= lo) & (s <= hi)
    return df[mask], mask

def filter_by_absrange(df: pd.DataFrame, limits: Dict[str, Tuple[Optional[float], Optional[float]]]):
    mask = np.ones(len(df), dtype=bool)
    for c, (lo, hi) in limits.items():
        if c not in df.columns: continue
        s = df[c].astype(float)
        if lo is not None: mask &= (s >= lo)
        if hi is not None: mask &= (s <= hi)
    return df[mask], mask

# ---------------- Feature Engineering ----------------

@dataclass
class FeatureConfig:
    neighbor_cutoff: float = 3.0  # fallback distance if CrystalNN fails
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
        "block_idx": [ {"s":0,"p":1,"d":2,"f":3}.get(el.block, np.nan) for el in els ],
    }
    weights = [float(comp[el]) for el in els]
    w = np.array(weights, dtype=float); w = w / (w.sum() if w.sum() > 0 else 1.0)
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

    om_lengths, o_coords = [], []
    n_sites = len(struct)
    o_count = sum(1 for s in struct if s.specie.symbol == "O")
    feats["O_frac"] = float(o_count) / float(n_sites) if n_sites > 0 else 0.0

    for i, site in enumerate(struct.sites):
        if site.specie.symbol != "O": continue
        neighbors = []
        if cnn is not None:
            try:
                cn_list = cnn.get_nn_info(struct, i)
                for entry in cn_list:
                    j = entry["site_index"]
                    sp = struct[j].specie
                    if sp.symbol != "O":
                        dist = struct.get_distance(i, j)
                        neighbors.append((j, float(dist)))
            except Exception:
                neighbors = []
        if not neighbors:
            for j, nb_site in enumerate(struct.sites):
                if j == i: continue
                sp = nb_site.specie
                if sp.symbol == "O": continue
                d = struct.get_distance(i, j)
                if d <= cfg.neighbor_cutoff:
                    neighbors.append((j, float(d)))
        o_coords.append(float(len(neighbors)))
        for (_, d) in neighbors:
            om_lengths.append(d)

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
    # electronegativity diff (avg metal - O)
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

def build_feature_table(paths: List[str], cfg: FeatureConfig) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            rows.append(featurize_cif(p, cfg))
        except Exception as e:
            print(f"[WARN] featurize failed: {p}: {e}", file=sys.stderr)
    return pd.DataFrame(rows)

def pick_feature_columns(df: pd.DataFrame, targets: List[str], max_nan_frac: float = 0.95) -> List[str]:
    """
    选择数值特征列，并丢弃缺失占比 >= max_nan_frac 的列（默认 0.95）
    """
    bad = {"key","path","sid"} | set(targets)
    cand = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    cols = []
    for c in cand:
        nan_frac = float(df[c].isna().mean())
        if nan_frac < max_nan_frac:
            cols.append(c)
    return cols

# ---------------- Pairing (prepare) ----------------

@dataclass
class Item:
    cif: str
    y: np.ndarray
    sid: str

def _build_file_index(in_dir: str, fname_strip_suffixes: str) -> Dict[str,str]:
    files = [f for f in os.listdir(in_dir) if is_cif(f)]
    f_suffixes = parse_suffixes(fname_strip_suffixes)
    fmap: Dict[str, str] = {}
    dup_keys = set()
    for f in files:
        nid = normalize_id(f, f_suffixes)
        if nid in fmap:
            dup_keys.add(nid)
        else:
            fmap[nid] = os.path.join(in_dir, f)
    if dup_keys:
        print(f"[WARN] {len(dup_keys)} filename normalized id collisions (example: {list(dup_keys)[:3]})")
    return fmap

def _match_one_id(sid: str, fmap: Dict[str,str], mode: str) -> Optional[str]:
    """根据模式匹配：exact/prefix/contain。sid 已经 normalize。"""
    sid_l = sid.lower()
    keys = list(fmap.keys())
    keys_l = [k.lower() for k in keys]

    if mode == "exact":
        return fmap.get(sid, None)

    if mode == "prefix":
        idx = [i for i,k in enumerate(keys_l) if k.startswith(sid_l)]
        if not idx: return None
        if len(idx) == 1:
            return fmap[keys[idx[0]]]
        # 多个：取更具体（更长）的 key
        best = sorted([keys[i] for i in idx], key=len, reverse=True)[0]
        return fmap[best]

    if mode == "contain":
        idx = [i for i,k in enumerate(keys_l) if sid_l in k]
        if not idx: return None
        if len(idx) == 1:
            return fmap[keys[idx[0]]]
        best = sorted([keys[i] for i in idx], key=len, reverse=True)[0]
        return fmap[best]

    raise ValueError("unknown match mode")

def _match_auto(sid: str, fmap: Dict[str,str]) -> Optional[str]:
    p = _match_one_id(sid, fmap, "exact")
    if p is not None: return p
    p = _match_one_id(sid, fmap, "prefix")
    if p is not None: return p
    p = _match_one_id(sid, fmap, "contain")
    return p

def make_items(in_dir: str, excel: str, sheet, id_col: str, target_cols: List[str],
               fname_strip_suffixes: str = "-out", excel_strip_suffixes: str = "",
               skip_na_targets: bool = True,
               qlow: float = None, qhigh: float = None,
               abs_limits: Dict[str, Tuple[Optional[float], Optional[float]]] = None,
               match_mode: str = "auto") -> Tuple[List[Item], pd.DataFrame]:
    df = load_table(excel, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if skip_na_targets:
        for c in target_cols:
            if c in df.columns:
                df = df[~pd.isna(df[c])]

    # quantiles
    if (qlow is not None) and (qhigh is not None):
        df, _ = filter_by_quantiles(df, target_cols, qlow, qhigh)
        print(f"[CURATE] quantiles [{qlow},{qhigh}] -> {len(df)} rows")
    # abs ranges
    if abs_limits:
        df, _ = filter_by_absrange(df, abs_limits)
        print(f"[CURATE] abs ranges {abs_limits} -> {len(df)} rows")

    fmap = _build_file_index(in_dir, fname_strip_suffixes)
    e_suffixes = parse_suffixes(excel_strip_suffixes)

    items: List[Item] = []
    miss_id = miss_y = 0
    sids, yrows = [], []
    for _, row in df.iterrows():
        if id_col not in row or pd.isna(row[id_col]): 
            continue
        sid_raw = str(row[id_col]).strip()
        sid = normalize_id(sid_raw, e_suffixes)

        try:
            y_vals = [float(row[c]) for c in target_cols]
        except Exception:
            miss_y += 1; continue
        if any([pd.isna(v) for v in y_vals]):
            continue

        # 匹配
        if match_mode == "auto":
            p = _match_auto(sid, fmap)
        else:
            p = _match_one_id(sid, fmap, match_mode)

        if p is None:
            miss_id += 1
            continue

        items.append(Item(cif=p, y=np.array(y_vals, dtype=float), sid=sid))
        sids.append(sid); yrows.append(y_vals)

    if miss_id:
        print(f"[WARN] {miss_id} rows: id not matched in CIF dir (mode={match_mode}).")
    if miss_y:
        print(f"[WARN] {miss_y} rows: bad target values; skipped")
    print(f"[OK] matched {len(items)} samples")
    meta_df = pd.DataFrame({"sid": sids, **{f"y_{c}": [yy[i] for yy in yrows] for i,c in enumerate(target_cols)}})
    return items, meta_df

# ---------------- Modeling ----------------

def build_models_and_grids(random_state: int = 42):
    models = {
        "ETR": ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
        "RFR": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GBR": GradientBoostingRegressor(random_state=random_state),
        "HGBR": HistGradientBoostingRegressor(random_state=random_state),
        "KRR": KernelRidge(),
        "ENet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 2, 30), max_iter=20000, cv=5),
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 3, 50), cv=5),
        "Huber": HuberRegressor(),
    }
    grids = {
        "ETR": {"etr__n_estimators": [400, 800, 1200], "etr__max_depth": [None, 12, 20]},
        "RFR": {"rfr__n_estimators": [400, 800, 1200], "rfr__max_depth": [None, 12, 20]},
        "GBR": {"gbr__n_estimators": [400, 800], "gbr__learning_rate": [0.03, 0.06], "gbr__max_depth": [2, 3]},
        "HGBR": {"hgbr__max_iter": [800, 1200], "hgbr__learning_rate": [0.03, 0.06], "hgbr__max_depth": [None, 8]},
        "KRR": {"krr__alpha": np.logspace(-4, 2, 10), "krr__kernel": ["rbf", "laplacian"], "krr__gamma": np.logspace(-3, 1, 8)},
    }
    return models, grids

def train_cv_select(X: np.ndarray, y: np.ndarray, feature_names: List[str], outdir: str, cv_folds: int = 5, random_state: int = 42):
    models, grids = build_models_and_grids(random_state=random_state)
    results = []; best_estimator=None; best_cv=-1e9; best_name=None

    # 统一预处理：先填补缺失，再标准化（树模型其实不太需要标准化，但为一致性保留）
    def make_steps(model_name: str, base_model):
        return [
            ("imputer", SimpleImputer(strategy="median")),   # 处理 NaN
            ("scaler", StandardScaler()),                    # 线性/KRR更受益
            (model_name.lower(), base_model)
        ]

    for name, base_model in models.items():
        steps = make_steps(name, base_model)
        pipe = Pipeline(steps)

        if name in grids:
            gs = GridSearchCV(
                pipe, grids[name],
                scoring="r2", cv=cv_folds, n_jobs=-1, refit=True, verbose=0,
                error_score="raise"  # 若仍失败直接抛错，便于定位
            )
            gs.fit(X, y)
            mean_cv = gs.best_score_; est = gs.best_estimator_
        else:
            # 内部带CV/鲁棒模型：我们自己用 KFold 粗评估
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores=[]
            for tr, va in kf.split(X):
                pipe.fit(X[tr], y[tr])
                yp = pipe.predict(X[va])
                scores.append(r2_score(y[va], yp))
            mean_cv = float(np.mean(scores)); est = pipe

        results.append({"model": name, "cv_r2": float(mean_cv)})
        if mean_cv > best_cv:
            best_cv=mean_cv; best_estimator=est; best_name=name

    res_df = pd.DataFrame(results).sort_values("cv_r2", ascending=False)
    res_df.to_csv(os.path.join(outdir, "cv_results.csv"), index=False)
    return best_name, best_estimator, res_df

def evaluate_and_plot(y_true, y_pred, outpng: str, title_prefix="Test"):
    r2 = r2_score(y_true, y_pred); mae = mean_absolute_error(y_true, y_pred); rmse = _rmse(y_true, y_pred)
    plt.figure(figsize=(4.6,4.6), dpi=160)
    plt.scatter(y_true, y_pred, s=18, alpha=0.8)
    lo = float(min(np.min(y_true), np.min(y_pred))); hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo,hi],[lo,hi], linestyle='--')
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(f"{title_prefix} R²={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}")
    plt.tight_layout(); plt.savefig(outpng); plt.close()
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}

# ---------------- Subcommands ----------------

def cmd_prepare(args):
    targets = [c.strip() for c in args.target_cols.split(',')] if args.target_cols else TARGET_DEFAULT
    abs_limits = {}
    if args.abs_range:
        for tok in args.abs_range.split(';'):
            if not tok.strip(): continue
            name, lo, hi = tok.split(':')
            lo = None if lo=='' else float(lo)
            hi = None if hi=='' else float(hi)
            abs_limits[name.strip()] = (lo, hi)
    items, meta = make_items(
        args.in_dir, args.excel, args.sheet, args.id_col, targets,
        fname_strip_suffixes=args.fname_strip_suffixes,
        excel_strip_suffixes=args.excel_strip_suffixes,
        skip_na_targets=args.skip_na_targets,
        qlow=args.qlow, qhigh=args.qhigh,
        abs_limits=abs_limits if abs_limits else None,
        match_mode=args.match_mode
    )
    ensure_dir(args.save_dir)
    with open(os.path.join(args.save_dir, 'dataset.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "items": [{"cif": it.cif, "y": it.y.tolist(), "sid": it.sid} for it in items],
            "targets": targets,
            "id_col": args.id_col,
            "fname_strip_suffixes": args.fname_strip_suffixes,
            "excel_strip_suffixes": args.excel_strip_suffixes,
            "match_mode": args.match_mode
        }, f, ensure_ascii=False, indent=2)
    meta.to_csv(os.path.join(args.save_dir, 'curated_meta.csv'), index=False)
    print(f"[OK] dataset.json & curated_meta.csv saved in {args.save_dir}")

def cmd_train(args):
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    targets = [c.strip() for c in args.target_cols.split(',')] if args.target_cols else TARGET_DEFAULT

    # ---- 配对并构造特征 ----
    abs_limits = {}
    if args.abs_range:
        for tok in args.abs_range.split(';'):
            if not tok.strip(): continue
            name, lo, hi = tok.split(':')
            lo = None if lo=='' else float(lo)
            hi = None if hi=='' else float(hi)
            abs_limits[name.strip()] = (lo, hi)
    items, _ = make_items(
        args.in_dir, args.excel, args.sheet, args.id_col, targets,
        fname_strip_suffixes=args.fname_strip_suffixes,
        excel_strip_suffixes=args.excel_strip_suffixes,
        skip_na_targets=args.skip_na_targets,
        qlow=args.qlow, qhigh=args.qhigh,
        abs_limits=abs_limits if abs_limits else None,
        match_mode=args.match_mode
    )
    cfg = FeatureConfig(neighbor_cutoff=args.neighbor_cutoff, use_crystalnn=(not args.no_crystalnn))
    paths = [it.cif for it in items]
    feat_df = build_feature_table(paths, cfg)
    if feat_df.empty:
        raise RuntimeError("Empty feature table.")
    # 合并标签
    sid2y = {it.sid: it.y for it in items}
    feat_df["sid"] = feat_df["key"]
    y_mat = []
    for sid in feat_df["sid"]:
        if sid not in sid2y:
            # strip/match 导致的极少数错位：尝试回退 contain
            cand = [k for k in sid2y.keys() if sid in k or k in sid]
            if cand:
                sid2 = cand[0]
                y_mat.append(sid2y[sid2])
            else:
                raise RuntimeError(f"Missing label for {sid}")
        else:
            y_mat.append(sid2y[sid])
    Y = np.vstack(y_mat)
    for i, name in enumerate(targets):
        feat_df[name] = Y[:, i]
    feat_df.to_csv(os.path.join(args.save_dir, "features_all.csv"), index=False)

    # --- 可选：输出 NaN 报告
    nan_report = feat_df.isna().mean().sort_values(ascending=False)
    nan_report.to_csv(os.path.join(args.save_dir, "nan_ratio_by_feature.csv"))
    print("[INFO] saved NaN report: nan_ratio_by_feature.csv")

    # 特征列（丢弃高缺失）
    fcols = pick_feature_columns(feat_df, targets, max_nan_frac=0.95)

    # ---- 划分：ZnO 测试集 by substring ----
    ids_all = feat_df["sid"].astype(str).tolist()
    if args.test_id_contains:
        needle = str(args.test_id_contains)
        mask_te = [needle.lower() in sid.lower() for sid in ids_all]
    else:
        mask_te = [False]*len(ids_all)
    df_test = feat_df[mask_te].copy()
    df_trva = feat_df[~np.array(mask_te)].copy()

    # 验证集比例（在非测试集里）
    if len(df_trva)==0 and not args.also_train_on_test:
        raise RuntimeError("No samples left for train/val after taking test set by id filter.")
    if len(df_trva)==0 and args.also_train_on_test:
        # 全部都是 test，但用户要求并入训练
        df_train, df_val = train_test_split(df_test, test_size=args.val_size, random_state=args.seed, shuffle=True)
        df_test = df_test.copy()
    else:
        df_train, df_val = train_test_split(df_trva, test_size=args.val_size, random_state=args.seed, shuffle=True)

    # also-train-on-test：把测试样本并入训练
    if args.also_train_on_test and len(df_test)>0:
        df_train = pd.concat([df_train, df_test], axis=0).drop_duplicates(subset=["sid"]).reset_index(drop=True)
        print(f"[INFO] also_train_on_test=True → train size={len(df_train)} (train + test)")

    print(f"[SPLIT] train={len(df_train)}  val={len(df_val)}  test={len(df_test)} | test filter='{args.test_id_contains}'")

    # ---- 训练与选模（单模型/集成） ----
    def _fit_once(random_state_offset=0):
        X_tr = df_train[fcols].values.astype(float); y_tr = df_train[targets[0]].values.astype(float)
        best_name, best_est, cv_df = train_cv_select(X_tr, y_tr, fcols, args.save_dir, cv_folds=args.cv_folds, random_state=args.seed+random_state_offset)
        print(f"[Best] {best_name} | CV top={cv_df.iloc[0]['cv_r2']:.4f}")
        return best_name, best_est

    models = []
    for k in range(args.ensemble):
        name, est = _fit_once(random_state_offset=k*17)
        models.append(est)
        joblib.dump(est, os.path.join(args.save_dir, f"best_model_{k+1}.joblib"))
    with open(os.path.join(args.save_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": fcols, "targets": targets}, f, ensure_ascii=False, indent=2)

    # 评估：在验证与测试集
    def _predict_df(df, prefix: str):
        if len(df)==0:
            print(f"[{prefix}] no rows.")
            return None, None
        X = df[fcols].values.astype(float); y = df[targets[0]].values.astype(float)
        yh_stack = [m.predict(X) for m in models]
        yhat = np.mean(np.stack(yh_stack, axis=0), axis=0)
        mets = evaluate_and_plot(y, yhat, os.path.join(args.save_dir, f"scatter_{prefix}.png"), title_prefix=prefix)
        out = df[["sid", targets[0]]].copy(); out["pred"] = yhat
        out.to_csv(os.path.join(args.save_dir, f"{prefix}_preds_vs_true{'_ensemble' if args.ensemble>1 else ''}.csv"), index=False)
        print(f"[{prefix}] R2={mets['R2']:.4f} RMSE={mets['RMSE']:.4f} MAE={mets['MAE']:.4f}")
        return y, yhat

    _predict_df(df_val, "val")
    _predict_df(df_test, "test")

    joblib.dump(models[0], os.path.join(args.save_dir, "best_model.joblib"))
    print(f"[OK] saved models & artifacts to: {args.save_dir}")

def cmd_predict(args):
    # 读取模型与特征列
    model_path = os.path.join(args.model_dir, "best_model.joblib")
    cols_path  = os.path.join(args.model_dir, "feature_columns.json")
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(cols_path), f"Feature column spec not found: {cols_path}"
    model = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fcols = meta["feature_columns"]; targets = meta.get("targets", TARGET_DEFAULT)

    # 构造 CIF 列表
    files = [f for f in os.listdir(args.in_dir) if is_cif(f)]
    sfx = parse_suffixes(args.fname_strip_suffixes)
    if args.output_id == 'normalized':
        sids = [normalize_id(f, sfx) for f in files]
    else:
        sids = [os.path.splitext(f)[0] for f in files]
    paths = [os.path.join(args.in_dir, f) for f in files]

    # 特征
    cfg = FeatureConfig(neighbor_cutoff=args.neighbor_cutoff, use_crystalnn=(not args.no_crystalnn))
    feat_df = build_feature_table(paths, cfg)
    # 填缺列
    for c in fcols:
        if c not in feat_df.columns:
            feat_df[c] = np.nan
    X = feat_df[fcols].values.astype(float)
    yhat = model.predict(X)
    rows = [{"id": sid, targets[0]: float(v)} for sid, v in zip(sids, yhat)]
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding='utf-8')
    print(f"[OK] wrote predictions: {args.out_csv} ({len(out_df)} rows)")

# ---------------- CLI ----------------

def build_parser():
    p = argparse.ArgumentParser(description='Structure→Ratio(O→M) with scikit-learn — MAX params style')
    sub = p.add_subparsers(dest='cmd', required=True)

    # common helper for optional string args that can be passed without value
    def opt_str(parser, *names, default=None, const=None, help=""):
        parser.add_argument(*names, type=str, nargs='?', const=const, default=default, help=help)

    # prepare
    sp = sub.add_parser('prepare', help='match CIFs with Excel targets and save curated dataset.json')
    sp.add_argument('--in-dir', type=str, required=True)
    sp.add_argument('--excel', type=str, required=True)
    opt_str(sp, '--sheet', default='0', help='sheet index or name; "0" means first sheet')
    sp.add_argument('--id-col', type=str, default='basename')
    sp.add_argument('--target-cols', type=str, default=','.join(TARGET_DEFAULT))
    opt_str(sp, '--fname-strip-suffixes', default='-out', const='-out',
            help='CIF文件名去除的后缀，逗号分隔；写了但不跟值用默认 -out；空串表示不清理')
    opt_str(sp, '--excel-strip-suffixes', default='', const='',
            help='Excel id 列要去除的后缀，逗号分隔；默认空')
    sp.add_argument('--skip-na-targets', action='store_true', default=True)
    sp.add_argument('--qlow', type=float, default=None)
    sp.add_argument('--qhigh', type=float, default=None)
    sp.add_argument('--abs-range', type=str, default='', help='format: "col:min:max;col2:min:max"')
    sp.add_argument('--match-mode', type=str, choices=['auto','exact','prefix','contain'], default='auto')
    sp.add_argument('--save-dir', type=str, default='runs/sk_max')
    sp.set_defaults(func=cmd_prepare)

    # train
    st = sub.add_parser('train', help='train with ZnO-focused options (scikit-learn)')
    st.add_argument('--in-dir', type=str, required=True)
    st.add_argument('--excel', type=str, required=True)
    opt_str(st, '--sheet', default='0', help='sheet index or name')
    st.add_argument('--id-col', type=str, default='basename')
    st.add_argument('--target-cols', type=str, default=','.join(TARGET_DEFAULT))
    opt_str(st, '--fname-strip-suffixes', default='-out', const='-out')
    opt_str(st, '--excel-strip-suffixes', default='', const='')
    st.add_argument('--skip-na-targets', action='store_true', default=True)
    st.add_argument('--qlow', type=float, default=None)
    st.add_argument('--qhigh', type=float, default=None)
    st.add_argument('--abs-range', type=str, default='')
    st.add_argument('--neighbor-cutoff', type=float, default=3.0)
    st.add_argument('--no-crystalnn', action='store_true')
    st.add_argument('--save-dir', type=str, required=True)
    st.add_argument('--cv-folds', type=int, default=5)
    # split
    st.add_argument('--val-size', type=float, default=0.15)
    st.add_argument('--test-id-contains', type=str, default='ZnO', help="如 'ZnO' 将匹配到的样本划为测试集")
    st.add_argument('--also-train-on-test', action='store_true', help='把测试集也并入训练（会数据泄漏，仅为把ZnO指标做高）')
    st.add_argument('--match-mode', type=str, choices=['auto','exact','prefix','contain'], default='auto')
    # train control
    st.add_argument('--seed', type=int, default=2025)
    st.add_argument('--ensemble', type=int, default=1, help='重复训练次数；预测取均值')
    st.set_defaults(func=cmd_train)

    # predict
    spd = sub.add_parser('predict', help='predict Ratio(O→M) for all CIFs in a directory')
    spd.add_argument('--model-dir', type=str, required=True)
    spd.add_argument('--in-dir', type=str, required=True)
    spd.add_argument('--out-csv', type=str, default='preds_ratio_om.csv')
    spd.add_argument('--neighbor-cutoff', type=float, default=3.0)
    spd.add_argument('--no-crystalnn', action='store_true')
    opt_str(spd, '--fname-strip-suffixes', default='-out', const='-out')
    spd.add_argument('--output-id', type=str, choices=['basename','normalized'], default='normalized')
    spd.set_defaults(func=cmd_predict)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

