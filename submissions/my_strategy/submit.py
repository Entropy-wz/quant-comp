"""
批量推理脚本 v2 — 生成 submission.csv (Public Phase).
逐分区处理，每 500 time_id 报告进度，边跑边写避免内存问题。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRATEGY_DIR))
from main import Model


def _visible_cols(frame: pd.DataFrame) -> list[str]:
    forbidden = {"weight", "target", "timestamp", "symbol"}
    return [c for c in frame.columns
            if c not in forbidden and not str(c).startswith("responder_")]


def manifest_files(data_root: Path, split: str) -> list[Path]:
    manifest = data_root / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        files = info.get("files", {}).get(split, [])
        if files:
            return [data_root / str(f) for f in files]
    return sorted((data_root / split).glob("*.parquet"))


def process_partition(path: Path, model: Model, part_idx: int, total: int) -> Path:
    """处理一个分区，返回临时 CSV 路径。"""
    out = path.parent / f"_submit_part{part_idx}.csv"
    frame = pd.read_parquet(path)
    cols = _visible_cols(frame)
    groups = list(frame.groupby("time_id", sort=False))
    n_groups = len(groups)

    print(f"[{part_idx+1}/{total}] {path.name} ({len(frame):,}行, {n_groups} time_ids)", flush=True)

    t0 = time.perf_counter()
    with open(out, "w") as f:
        f.write("row_id,target\n")
        for gi, (time_id, chunk) in enumerate(groups):
            try:
                test = chunk[cols].copy()
                pred = model.predict(test)
                for rid, p in zip(chunk["row_id"], pred):
                    f.write(f"{rid},{p:.8f}\n")
            except Exception as e:
                print(f"\n  ! time_id={time_id}: {e}", flush=True)
                for rid in chunk["row_id"]:
                    f.write(f"{rid},0.0\n")

            if (gi + 1) % 500 == 0:
                pct = (gi + 1) / n_groups * 100
                elapsed = time.perf_counter() - t0
                speed = (gi + 1) / elapsed if elapsed > 0 else 0
                eta = (n_groups - gi - 1) / speed if speed > 0 else 0
                print(f"  {gi+1}/{n_groups} ({pct:.0f}%) {speed:.0f} tids/s, 预计剩余 {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  完成 ({elapsed:.0f}s)", flush=True)
    return out


def concat_parts(parts: list[Path], output: Path):
    """合并所有临时文件。"""
    print(f"\n合并 {len(parts)} 个临时文件 → {output} ...", flush=True)
    with open(output, "w") as out:
        out.write("row_id,target\n")
        for p in parts:
            with open(p) as f:
                next(f)  # 跳表头
                for line in f:
                    out.write(line)
    # 清理临时文件
    for p in parts:
        p.unlink(missing_ok=True)


def generate(data_root: str | Path, output: str | Path) -> Path:
    data_root = Path(data_root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("载入模型...", flush=True)
    t0 = time.perf_counter()
    model = Model()
    print(f"  完成 ({time.perf_counter() - t0:.1f}s)", flush=True)

    files = manifest_files(data_root, "test")
    print(f"测试分区: {len(files)} 个\n", flush=True)

    parts = []
    for i, path in enumerate(files):
        p = process_partition(path, model, i, len(files))
        parts.append(p)

    concat_parts(parts, output)

    df = pd.read_csv(output)
    print(f"\n最终: {len(df):,} 行, target ∈ [{df.target.min():.4f}, {df.target.max():.4f}]", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="批量推理生成 submission.csv")
    parser.add_argument("--data-root", default="../public_release_20260630/data")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    generate(args.data_root, args.output)


if __name__ == "__main__":
    main()
