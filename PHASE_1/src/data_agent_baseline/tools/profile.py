"""One-shot profiler for a task's whole context directory.

Instead of forcing the agent to spend many steps listing files and reading each
one, `profile_context` walks every file once and returns a compact structural
summary: per-column stats for CSVs, record keys for JSONs, text previews for
docs (which often hold the schema / foreign-key notes), and table schema for
sqlite databases. One call gives the model the full data landscape.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import list_context_tree
from data_agent_baseline.tools.sqlite import inspect_sqlite_schema

_CSV_EXTS = {".csv", ".tsv"}
_JSON_EXTS = {".json"}
_DOC_EXTS = {".md", ".txt", ".text", ".rst"}
_DB_EXTS = {".db", ".sqlite", ".sqlite3"}

_MAX_SAMPLES = 5
_DOC_PREVIEW_CHARS = 800


def _jsonable(value: object) -> object:
    """Coerce numpy / pandas scalars into JSON-serializable Python values."""
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:  # pragma: no cover - numpy always present, defensive only
        pass
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _profile_csv(path: Path) -> dict[str, object]:
    import pandas as pd

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    columns: list[dict[str, object]] = []
    for name in df.columns:
        series = df[name]
        col: dict[str, object] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique(dropna=True)),
            "n_null": int(series.isna().sum()),
            "samples": [_jsonable(v) for v in series.dropna().unique()[:_MAX_SAMPLES]],
        }
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            col["min"] = _jsonable(series.min())
            col["max"] = _jsonable(series.max())
        columns.append(col)
    return {"kind": "csv", "row_count": int(len(df)), "columns": columns}


def _profile_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        data = json.load(handle)

    if isinstance(data, dict) and isinstance(data.get("records"), list):
        records = data["records"]
    elif isinstance(data, list):
        records = data
    else:
        keys = list(data.keys())[:50] if isinstance(data, dict) else None
        return {"kind": "json", "shape": "object", "top_level_keys": keys}

    keys = list(records[0].keys()) if records and isinstance(records[0], dict) else []
    return {
        "kind": "json",
        "record_count": len(records),
        "keys": keys,
        "sample": records[0] if records else None,
    }


def _profile_doc(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    return {"kind": "doc", "chars": len(text), "preview": text[:_DOC_PREVIEW_CHARS]}


def profile_context(task: PublicTask, *, max_depth: int = 6) -> dict[str, object]:
    tree = list_context_tree(task, max_depth=max_depth)
    files: dict[str, object] = {}
    for entry in tree.get("entries", []):
        if entry.get("kind") != "file":
            continue
        rel_path = str(entry["path"])
        path = task.context_dir / rel_path
        ext = path.suffix.lower()
        try:
            if ext in _CSV_EXTS:
                profile = _profile_csv(path)
            elif ext in _JSON_EXTS:
                profile = _profile_json(path)
            elif ext in _DOC_EXTS:
                profile = _profile_doc(path)
            elif ext in _DB_EXTS:
                profile = {"kind": "sqlite", **inspect_sqlite_schema(path)}
            else:
                profile = {"kind": "other", "size": entry.get("size")}
        except Exception as exc:  # keep profiling the rest even if one file fails
            profile = {"kind": ext.lstrip(".") or "unknown", "error": f"{type(exc).__name__}: {exc}"}
        files[rel_path] = profile

    return {"root": tree.get("root"), "file_count": len(files), "files": files}
