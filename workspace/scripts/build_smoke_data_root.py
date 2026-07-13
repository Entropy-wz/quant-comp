from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contest.paths import load_paths

DEFAULT_N_TIME_IDS = 300
DEFAULT_COLUMN_CHUNK_WIDTH = 32


def _time_id_mask(path: Path, threshold: int) -> pa.BooleanArray:
    """Cheap projected read of a single column to build the row mask.

    Reading only ``time_id`` keeps peak memory tiny (a few MB) regardless of
    how many feature columns the file has, which is what lets this run on a
    host where a full ``pd.read_parquet`` of the same file raises
    ``MemoryError``.
    """

    ids = pq.read_table(path, columns=["time_id"]).column("time_id")
    return pc.less_equal(ids, threshold)


def _read_filtered_table(path: Path, mask: pa.BooleanArray, chunk_width: int) -> pa.Table:
    """Read every column but decode only ``chunk_width`` columns at a time.

    The source test partitions are stored as a single Parquet row group, so
    ``ParquetFile.iter_batches`` still has to materialize the whole row group
    for any column it touches. Reading a narrow slice of columns at a time
    (instead of all ~326 at once) bounds decode memory to roughly
    ``chunk_width / n_columns`` of a full-width read, then the boolean mask
    immediately drops everything outside the sampled time_ids before the next
    chunk is read.
    """

    schema = pq.read_schema(path)
    columns = list(schema.names)
    filtered_columns: dict[str, pa.ChunkedArray] = {}
    for start in range(0, len(columns), chunk_width):
        chunk_cols = columns[start : start + chunk_width]
        chunk_table = pq.read_table(path, columns=chunk_cols).filter(mask)
        for name in chunk_cols:
            filtered_columns[name] = chunk_table.column(name)
        del chunk_table
    return pa.table(filtered_columns).select(columns)


def build_smoke_data_root(
    *,
    data_root: Path,
    out_root: Path,
    n_time_ids: int = DEFAULT_N_TIME_IDS,
    column_chunk_width: int = DEFAULT_COLUMN_CHUNK_WIDTH,
    force: bool = False,
) -> Path:
    """Build (or reuse) a tiny local test data-root for smoke-testing the
    official runner without loading a full test partition into memory.

    Samples the first ``n_time_ids`` distinct ``time_id`` values (and all
    assets present at those times) from the real first test partition and
    writes them to ``out_root/test/test_partition_000.parquet`` with a
    matching ``manifest.json`` so the unmodified official runner can consume
    it exactly like the real data_root.
    """

    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    rel_path = manifest["files"]["test"][0]
    src_path = data_root / rel_path

    out_test_dir = out_root / "test"
    out_path = out_test_dir / "test_partition_000.parquet"
    manifest_path = out_root / "manifest.json"

    if out_path.exists() and manifest_path.exists() and not force:
        return out_root

    out_test_dir.mkdir(parents=True, exist_ok=True)

    tid_column = pq.read_table(src_path, columns=["time_id"]).column("time_id")
    unique_ids = sorted(set(tid_column.to_pylist()))
    if not unique_ids:
        raise ValueError(f"no time_id values found in {src_path}")
    n_take = min(n_time_ids, len(unique_ids))
    threshold = unique_ids[n_take - 1]
    del tid_column

    mask = _time_id_mask(src_path, threshold)
    matched_rows = int(pc.sum(mask).as_py() or 0)
    if matched_rows == 0:
        raise ValueError("smoke sample produced zero rows; check n_time_ids/threshold")

    table = _read_filtered_table(src_path, mask, column_chunk_width)
    pq.write_table(table, out_path, compression="zstd")

    smoke_manifest = {
        "competition": manifest.get("competition"),
        "version": f"{manifest.get('version', 'unknown')}-smoke",
        "description": (
            "Local smoke data root: first N time_ids sampled from the real "
            "test_partition_000 for fast official-runner gating on "
            "memory-constrained hosts. Not for scoring."
        ),
        "source": {
            "release_test_partition": rel_path,
            "n_time_ids_sampled": n_take,
            "n_time_ids_available": len(unique_ids),
            "time_id_threshold": threshold,
        },
        "files": {"test": ["test/test_partition_000.parquet"]},
        "rows": {"test": int(table.num_rows)},
        "counts": manifest.get("counts", {}),
        "columns": manifest.get("columns", {}),
        "format": manifest.get("format", {}),
    }
    manifest_path.write_text(json.dumps(smoke_manifest, indent=2), encoding="utf-8")
    return out_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small local smoke test data-root sampled from the real test partition 0."
    )
    parser.add_argument("--n-time-ids", type=int, default=DEFAULT_N_TIME_IDS)
    parser.add_argument("--column-chunk-width", type=int, default=DEFAULT_COLUMN_CHUNK_WIDTH)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the smoke data root already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths()
    out_root = paths["artifacts_dir"] / "smoke_data_root"
    result = build_smoke_data_root(
        data_root=paths["data_root"],
        out_root=out_root,
        n_time_ids=args.n_time_ids,
        column_chunk_width=args.column_chunk_width,
        force=args.force,
    )
    print(f"smoke data root ready at {result}")


if __name__ == "__main__":
    main()
