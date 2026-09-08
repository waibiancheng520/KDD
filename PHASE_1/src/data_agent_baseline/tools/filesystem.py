from __future__ import annotations

import csv
import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 20) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": len(data_rows),
    }


def _window(text: str, relative_path: str, offset: int, max_chars: int) -> dict[str, object]:
    """A slice of `text` plus the paging info needed to reach the rest of it.

    Every knowledge.md in this dataset is larger than one window, and the section
    that resolves column ambiguities sits at the end -- so a reader that cannot
    page past the first window simply never sees it.
    """
    total = len(text)
    start = max(0, min(offset, total))
    end = min(start + max_chars, total)
    result: dict[str, object] = {
        "path": relative_path,
        "preview": text[start:end],
        "offset": start,
        "next_offset": end if end < total else None,
        "total_chars": total,
        "truncated": end < total,
    }
    if end < total:
        result["hint"] = (
            f"{total - end} of {total} chars not shown. Call this tool again with "
            f"offset={end} to continue reading."
        )
    return result


def read_json_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 8000, offset: int = 0
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text())
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return _window(preview, relative_path, offset, max_chars)


def read_doc_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 8000, offset: int = 0
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    return _window(text, relative_path, offset, max_chars)
