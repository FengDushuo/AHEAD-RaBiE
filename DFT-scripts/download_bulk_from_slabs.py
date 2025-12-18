#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 slabs_host_doped/*.cif
- 从文件名解析 mp-id（例如 mp-1336_...）
- 用 MPRester 下载对应体相（原胞/标准结构）
- 保存到 bulk_cells_out/<mp-id>.cif
- 写 manifest.csv

依赖：mp-api, pymatgen, numpy, pandas
环境：export MAPI_KEY=你的 Materials Project API key
"""

import os, re, argparse
import pandas as pd
from glob import glob
from typing import Optional
from mp_api.client import MPRester

MPID_RE = re.compile(r"(mp-\d+)")

def parse_mpid_from_name(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = MPID_RE.search(base)
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab_dir", type=str, default="slabs_host_doped",
                    help="含 slab CIF 的目录")
    ap.add_argument("--outdir", type=str, default="bulk_cells_out",
                    help="原胞 CIF 的输出目录")
    ap.add_argument("--which", type=str, default="conventional",
                    choices=["as_is","primitive","symm_primitive","conventional"],
                    help="下载后输出哪种标准（默认 conventional）")
    ap.add_argument("--overwrite", action="store_true",
                    help="已存在同名 CIF 时强制覆盖")
    ap.add_argument("--max_files", type=int, default=0,
                    help="最多处理多少个 slab 文件（0=不限制）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob(os.path.join(args.slab_dir, "*.cif")))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY. 请先 export MAPI_KEY=你的key")

    # 去重：同一个 mp-id 只下载一次
    ids = []
    file_map = []
    for f in files:
        mid = parse_mpid_from_name(f)
        file_map.append((f, mid))
        if mid and mid not in ids:
            ids.append(mid)

    rows = []

    if not ids:
        print("[WARN] 没解析到任何 mp-id。请确认文件名里包含 mp-XXXX。")
    else:
        print(f"[INFO] 待下载 mp-ids 数量：{len(ids)}")

    with MPRester(api_key) as mpr:
        # 批量拉 summary，后续取 structure
        # 也可分批 search；这里直接逐个拉结构，简单稳妥
        downloaded = {}

        for mid in ids:
            out_cif = os.path.join(args.outdir, f"{mid}.cif")
            if (not args.overwrite) and os.path.exists(out_cif):
                rows.append({"mp_id": mid, "outfile": out_cif, "status": "exists"})
                downloaded[mid] = out_cif
                continue
            try:
                # 直接取 summary 返回的 structure（已是 pymatgen Structure）
                # 如果你想更严格，可用 materials API 拉最稳定条目；这里简化为 mp-id 指定条目
                doc = mpr.summary.search(material_ids=[mid], fields=["material_id","structure","formula_pretty"])
                if not doc:
                    rows.append({"mp_id": mid, "outfile": "", "status": "not_found"})
                    continue
                struct = doc[0].structure

                # 选择输出的标准形
                mode = args.which
                if mode == "primitive":
                    struct = struct.get_primitive_structure()
                elif mode == "symm_primitive":
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                        sga = SpacegroupAnalyzer(struct, symprec=1e-3, angle_tolerance=5)
                        struct = sga.get_primitive_standard_structure()
                    except Exception:
                        pass
                elif mode == "conventional":
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                        sga = SpacegroupAnalyzer(struct, symprec=1e-3, angle_tolerance=5)
                        struct = sga.get_conventional_standard_structure()
                    except Exception:
                        pass
                # as_is 则不变

                struct.to(fmt="cif", filename=out_cif)
                rows.append({"mp_id": mid, "outfile": out_cif, "status": "ok"})
                downloaded[mid] = out_cif
                print("[OK]", mid, "→", out_cif)
            except Exception as e:
                rows.append({"mp_id": mid, "outfile": "", "status": f"error:{e}"})
                print("[ERR]", mid, e)

    # 附带记录 slab→mp-id→bulk 的映射
    for f, mid in file_map:
        rows.append({
            "slab_file": f,
            "mp_id": mid or "",
            "outfile": downloaded.get(mid, "") if mid else "",
            "status": ("mapped" if mid and mid in downloaded else ("no_mpid" if not mid else "not_downloaded"))
        })

    man = os.path.join(args.outdir, "manifest.csv")
    pd.DataFrame(rows).to_csv(man, index=False)
    print("[DONE] manifest:", man)

if __name__ == "__main__":
    main()
