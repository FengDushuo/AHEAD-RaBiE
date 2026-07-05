#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blend_singleview_oof_fixed_v2.py

Fixes:
- If global CatBoost files already contain fold/seed, do not call fillna(None)
- NN paths may contain only fold, not seed
- If one side lacks seed, merge falls back to (id, fold)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-root", required=True)
    ap.add_argument("--nn-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--blend-method", default="ridge", choices=["mean", "weighted", "ridge"])
    ap.add_argument("--cat-weight", type=float, default=0.6)
    ap.add_argument("--nn-weight", type=float, default=0.4)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--use-calibration", action="store_true")
    ap.add_argument("--calibration-mode", default="bias_only", choices=["bias_only", "affine"])
    return ap.parse_args()


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def metrics_from_df(df: pd.DataFrame, pred_col: str = "pred") -> Dict[str, float]:
    y_true = df["target"].to_numpy()
    y_pred = df[pred_col].to_numpy()
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_calibration(y_pred: np.ndarray, y_true: np.ndarray, mode: str):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[finite]
    y_true = y_true[finite]
    if len(y_pred) == 0:
        return {"mode": mode, "a": 1.0, "b": 0.0}
    if mode == "bias_only":
        b = float(np.mean(y_true - y_pred))
        return {"mode": mode, "a": 1.0, "b": b}
    if len(y_pred) < 2 or float(np.std(y_pred)) < 1e-12:
        b = float(np.mean(y_true - y_pred))
        return {"mode": "affine", "a": 1.0, "b": b}
    a, b = np.polyfit(y_pred, y_true, deg=1)
    return {"mode": "affine", "a": float(a), "b": float(b)}


def apply_calibration(pred: np.ndarray, calib: Dict[str, float]) -> np.ndarray:
    return calib["a"] * np.asarray(pred, dtype=float) + calib["b"]


def parse_run_tag(text: str) -> Tuple[Optional[int], Optional[int]]:
    s = str(text)
    m = re.search(r"fold[_\-]?(\d+).*?seed[_\-]?(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"fold[_\-]?(\d+)", s)
    if m:
        return int(m.group(1)), None
    return None, None


def _maybe_fillna(series: pd.Series, value):
    if value is None:
        return series
    return series.fillna(value)


def _normalize_test_pred_df(df: pd.DataFrame, source_name: str, origin: str) -> pd.DataFrame:
    out = df.copy()
    if "pred" not in out.columns:
        raise ValueError(f"{origin}: expected 'pred' column")
    if "id" not in out.columns:
        raise ValueError(f"{origin}: expected 'id' column")
    if "target" not in out.columns:
        raise ValueError(f"{origin}: expected 'target' column for OOF/test blending")

    p_fold, p_seed = parse_run_tag(origin)

    if "fold" not in out.columns:
        out["fold"] = p_fold
    else:
        out["fold"] = _maybe_fillna(out["fold"], p_fold)

    if "seed" not in out.columns:
        out["seed"] = p_seed
    else:
        out["seed"] = _maybe_fillna(out["seed"], p_seed)

    if out["fold"].isna().all():
        raise ValueError(f"{origin}: could not infer fold")

    out["fold"] = out["fold"].astype(int)
    out["seed"] = out["seed"].astype("Int64")
    out["model_source"] = source_name
    out["origin"] = origin

    cols = ["id", "target", "pred", "fold", "seed", "model_source", "origin"]
    extra = [c for c in ["run_tag", "run_weight"] if c in out.columns]
    return out[cols + extra].copy()


def _normalize_out_pred_df(df: pd.DataFrame, source_name: str, origin: str) -> pd.DataFrame:
    out = df.copy()
    if "pred" not in out.columns or "id" not in out.columns:
        raise ValueError(f"{origin}: expected id/pred columns")

    p_fold, p_seed = parse_run_tag(origin)

    if "fold" not in out.columns:
        out["fold"] = p_fold
    else:
        out["fold"] = _maybe_fillna(out["fold"], p_fold)

    if "seed" not in out.columns:
        out["seed"] = p_seed
    else:
        out["seed"] = _maybe_fillna(out["seed"], p_seed)

    if out["fold"].isna().all():
        raise ValueError(f"{origin}: could not infer fold")

    out["fold"] = out["fold"].astype(int)
    out["seed"] = out["seed"].astype("Int64")
    out["model_source"] = source_name
    out["origin"] = origin

    cols = ["id", "pred", "fold", "seed", "model_source", "origin"]
    extra = [c for c in ["run_tag", "run_weight"] if c in out.columns]
    return out[cols + extra].copy()


def load_cat_predictions(cat_root: Path):
    test_global = cat_root / "test_pred_all_runs.csv"
    out_global = cat_root / "addH_out_pred_all_runs.csv"

    test_parts = []
    out_parts = []

    if test_global.exists():
        df = pd.read_csv(test_global)
        test_parts.append(_normalize_test_pred_df(df, source_name="catboost", origin=str(test_global)))
    else:
        for fp in cat_root.rglob("test_pred.csv"):
            df = pd.read_csv(fp)
            test_parts.append(_normalize_test_pred_df(df, source_name="catboost", origin=str(fp)))

    if out_global.exists():
        df = pd.read_csv(out_global)
        out_parts.append(_normalize_out_pred_df(df, source_name="catboost", origin=str(out_global)))
    else:
        for fp in cat_root.rglob("addH_out_pred.csv"):
            df = pd.read_csv(fp)
            out_parts.append(_normalize_out_pred_df(df, source_name="catboost", origin=str(fp)))

    if not test_parts:
        raise FileNotFoundError(f"No CatBoost test predictions found under {cat_root}")

    test_df = pd.concat(test_parts, axis=0, ignore_index=True).drop_duplicates(subset=["id", "fold", "seed", "model_source"])
    if out_parts:
        out_df = pd.concat(out_parts, axis=0, ignore_index=True).drop_duplicates(subset=["id", "fold", "seed", "model_source"])
    else:
        out_df = pd.DataFrame(columns=["id", "pred", "fold", "seed", "model_source", "origin"])
    return test_df, out_df


def load_nn_predictions(nn_root: Path):
    test_parts = []
    out_parts = []

    for fp in nn_root.rglob("predictions.csv"):
        lower = str(fp).lower()
        df = pd.read_csv(fp)
        if ("addhout" in lower) or ("out" in lower and "test" not in lower):
            out_parts.append(_normalize_out_pred_df(df, source_name="nn", origin=str(fp)))
        else:
            if "target" in df.columns:
                test_parts.append(_normalize_test_pred_df(df, source_name="nn", origin=str(fp)))

    if not test_parts:
        raise FileNotFoundError(f"No NN test predictions found under {nn_root}")

    test_df = pd.concat(test_parts, axis=0, ignore_index=True).drop_duplicates(subset=["id", "fold", "seed", "model_source"])
    if out_parts:
        out_df = pd.concat(out_parts, axis=0, ignore_index=True).drop_duplicates(subset=["id", "fold", "seed", "model_source"])
    else:
        out_df = pd.DataFrame(columns=["id", "pred", "fold", "seed", "model_source", "origin"])
    return test_df, out_df


def _merge_predictions_with_seed_fallback(left_df: pd.DataFrame, right_df: pd.DataFrame, left_pred_name: str, right_pred_name: str, require_target: bool = True) -> pd.DataFrame:
    left = left_df.copy()
    right = right_df.copy()

    if require_target:
        left = left.rename(columns={"target": "target_left", "pred": left_pred_name})
        right = right.rename(columns={"target": "target_right", "pred": right_pred_name})
    else:
        left = left.rename(columns={"pred": left_pred_name})
        right = right.rename(columns={"pred": right_pred_name})

    has_left_seed = left["seed"].notna().all()
    has_right_seed = right["seed"].notna().all()

    if has_left_seed and has_right_seed:
        merged = left.merge(right, on=["id", "fold", "seed"], how="inner")
    else:
        left2 = left.drop(columns=["seed"]).copy()
        right2 = right.drop(columns=["seed"]).copy()
        merged = left2.merge(right2, on=["id", "fold"], how="inner")
        merged["seed"] = pd.NA

    if merged.empty:
        raise ValueError("No overlapping rows found after merge")

    if require_target:
        merged["target"] = merged["target_left"]
        both = merged[["target_left", "target_right"]].dropna()
        if len(both) > 0:
            max_abs_diff = np.max(np.abs(both["target_left"].to_numpy() - both["target_right"].to_numpy()))
            if max_abs_diff > 1e-6:
                print(f"[WARN] target mismatch after merge; max abs diff = {max_abs_diff:.6g}")
        keep_cols = ["id", "fold", "seed", "target", left_pred_name, right_pred_name]
    else:
        keep_cols = ["id", "fold", "seed", left_pred_name, right_pred_name]
    return merged[keep_cols].copy()


def make_wide_pair_df(cat_df: pd.DataFrame, nn_df: pd.DataFrame) -> pd.DataFrame:
    return _merge_predictions_with_seed_fallback(
        left_df=cat_df[["id", "target", "pred", "fold", "seed"]].copy(),
        right_df=nn_df[["id", "target", "pred", "fold", "seed"]].copy(),
        left_pred_name="pred_cat",
        right_pred_name="pred_nn",
        require_target=True,
    )


def make_wide_pair_out_df(cat_df: pd.DataFrame, nn_df: pd.DataFrame) -> pd.DataFrame:
    return _merge_predictions_with_seed_fallback(
        left_df=cat_df[["id", "pred", "fold", "seed"]].copy(),
        right_df=nn_df[["id", "pred", "fold", "seed"]].copy(),
        left_pred_name="pred_cat",
        right_pred_name="pred_nn",
        require_target=False,
    )


def fit_ridge_oof(df_wide: pd.DataFrame, alpha: float) -> Ridge:
    X = df_wide[["pred_cat", "pred_nn"]].to_numpy(dtype=float)
    y = df_wide["target"].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, y)
    return model


def apply_blend(df_wide: pd.DataFrame, method: str, cat_weight: float, nn_weight: float, ridge_model: Optional[Ridge] = None) -> np.ndarray:
    X_cat = df_wide["pred_cat"].to_numpy(dtype=float)
    X_nn = df_wide["pred_nn"].to_numpy(dtype=float)

    if method == "mean":
        return 0.5 * X_cat + 0.5 * X_nn
    if method == "weighted":
        w_sum = float(cat_weight + nn_weight)
        if w_sum <= 0:
            raise ValueError("cat_weight + nn_weight must be > 0")
        wc = float(cat_weight) / w_sum
        wn = float(nn_weight) / w_sum
        return wc * X_cat + wn * X_nn
    if method == "ridge":
        if ridge_model is None:
            raise ValueError("ridge_model is required for blend-method=ridge")
        X = df_wide[["pred_cat", "pred_nn"]].to_numpy(dtype=float)
        return ridge_model.predict(X)
    raise ValueError(f"Unsupported blend method: {method}")


def aggregate_by_id_mean(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    return df.groupby("id", as_index=False)[pred_col].mean()


def main():
    args = parse_args()
    cat_root = Path(args.cat_root).resolve()
    nn_root = Path(args.nn_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_test, cat_out = load_cat_predictions(cat_root)
    nn_test, nn_out = load_nn_predictions(nn_root)

    cat_test.to_csv(out_dir / "cat_test_all_runs.csv", index=False)
    nn_test.to_csv(out_dir / "nn_test_all_runs.csv", index=False)

    test_wide = make_wide_pair_df(cat_test, nn_test)
    test_wide.to_csv(out_dir / "blend_oof_pairs.csv", index=False)

    ridge_model = None
    ridge_info = None
    if args.blend_method == "ridge":
        ridge_model = fit_ridge_oof(test_wide, alpha=float(args.ridge_alpha))
        ridge_info = {
            "coef": ridge_model.coef_.tolist(),
            "intercept": float(ridge_model.intercept_),
            "alpha": float(args.ridge_alpha),
        }
        save_json(out_dir / "ridge_info.json", ridge_info)

    test_wide["pred_blend_raw"] = apply_blend(
        test_wide,
        method=args.blend_method,
        cat_weight=float(args.cat_weight),
        nn_weight=float(args.nn_weight),
        ridge_model=ridge_model,
    )

    calib = {"mode": args.calibration_mode, "a": 1.0, "b": 0.0}
    if args.use_calibration:
        calib = fit_calibration(
            y_pred=test_wide["pred_blend_raw"].to_numpy(dtype=float),
            y_true=test_wide["target"].to_numpy(dtype=float),
            mode=args.calibration_mode,
        )
    save_json(out_dir / "blend_calibration.json", calib)

    test_wide["pred_blend"] = apply_calibration(test_wide["pred_blend_raw"].to_numpy(dtype=float), calib)
    test_wide.to_csv(out_dir / "blend_oof_pairs_with_pred.csv", index=False)

    oof_final = aggregate_by_id_mean(test_wide[["id", "target", "pred_blend"]].rename(columns={"pred_blend": "pred"}), pred_col="pred")
    target_final = test_wide.groupby("id", as_index=False)["target"].mean()
    oof_final = target_final.merge(oof_final, on="id", how="inner")
    oof_final.to_csv(out_dir / "blend_oof_ensemble.csv", index=False)
    oof_metrics = metrics_from_df(oof_final, pred_col="pred")
    save_json(out_dir / "blend_oof_metrics.json", oof_metrics)

    if len(cat_out) > 0 and len(nn_out) > 0:
        cat_out.to_csv(out_dir / "cat_addH_out_all_runs.csv", index=False)
        nn_out.to_csv(out_dir / "nn_addH_out_all_runs.csv", index=False)

        out_wide = make_wide_pair_out_df(cat_out, nn_out)
        out_wide["pred_blend_raw"] = apply_blend(
            out_wide,
            method=args.blend_method,
            cat_weight=float(args.cat_weight),
            nn_weight=float(args.nn_weight),
            ridge_model=ridge_model,
        )
        out_wide["pred_blend"] = apply_calibration(out_wide["pred_blend_raw"].to_numpy(dtype=float), calib)
        out_wide.to_csv(out_dir / "blend_addH_out_pairs_with_pred.csv", index=False)

        out_final = aggregate_by_id_mean(out_wide[["id", "pred_blend"]].rename(columns={"pred_blend": "pred"}), pred_col="pred")
        out_final.to_csv(out_dir / "blend_addH_out_ensemble.csv", index=False)
        out_final.sort_values("pred").head(args.topk).to_csv(out_dir / "blend_addH_out_top20_low.csv", index=False)
        out_final.sort_values("pred", ascending=False).head(args.topk).to_csv(out_dir / "blend_addH_out_top20_high.csv", index=False)

    summary = {
        "blend_method": args.blend_method,
        "cat_weight": float(args.cat_weight),
        "nn_weight": float(args.nn_weight),
        "use_calibration": bool(args.use_calibration),
        "calibration_mode": args.calibration_mode,
        "oof_metrics": oof_metrics,
        "ridge_info": ridge_info,
    }
    save_json(out_dir / "blend_summary.json", summary)

    print("[DONE] blend finished ->", out_dir)
    print("[INFO] OOF metrics =", oof_metrics)
    if ridge_info is not None:
        print("[INFO] ridge coef/intercept =", ridge_info)


if __name__ == "__main__":
    main()
