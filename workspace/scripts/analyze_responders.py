from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.io import list_partition_files, load_partitions
from contest.paths import load_paths
from contest.responders import rank_responders_by_target_corr


def main() -> None:
    paths = load_paths()
    cfg_path = ROOT / "configs" / "train_lgb_v1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    files = list_partition_files(paths["data_root"], "train")
    max_parts = cfg.get("max_train_partitions", 1)
    if max_parts is not None:
        files = files[-int(max_parts) :]
    else:
        files = files[-1:]

    # Lean read: target/weight/responders only
    import pyarrow.parquet as pq

    names = list(pq.ParquetFile(files[0]).schema_arrow.names)
    cols = ["target", "weight", *[c for c in names if c.startswith("responder_")]]
    df = load_partitions(files, columns=cols)
    ranked = rank_responders_by_target_corr(df, top_k=15)
    payload = {
        "partitions": [p.name for p in files],
        "rows": int(len(df)),
        "ranked": [{"responder": n, "weighted_corr": c} for n, c in ranked],
    }
    out = paths["experiments_dir"] / "responder_corr.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
