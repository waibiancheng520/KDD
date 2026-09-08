"""Cross-source SQL over ALL of a task's data files, powered by DuckDB.

CSV, JSON records, and sqlite/db tables are normally queried through three
different mechanisms (hand-written `execute_python` loops for csv/json,
`execute_context_sql` for sqlite), which is where join / filter / aggregation
bugs creep in. This tool registers every csv, json, and sqlite table in the
context as a DuckDB table (named by the file stem, or the sqlite table name) so
the model can express the whole thing as one SQL statement -- even joining a CSV
against a sqlite table:

    SELECT m.first_name, m.last_name
    FROM member m JOIN zip_code z ON m.zip = z.zip_code
    WHERE z.state = 'Illinois'

DuckDB's sqlite scanner is bundled and loads offline; if it is unavailable for
any reason, sqlite tables are loaded via the Python standard library instead, so
the tool degrades gracefully rather than failing.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sqlite3
from decimal import Decimal
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask

_CSV_EXTS = {".csv", ".tsv"}
_JSON_EXTS = {".json"}
_DB_EXTS = {".db", ".sqlite", ".sqlite3"}


def _safe_ident(stem: str) -> str:
    name = re.sub(r"\W+", "_", stem).strip("_")
    if not name:
        name = "t"
    if name[0].isdigit():
        name = "t_" + name
    return name


def _records(data: object) -> list | None:
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    if isinstance(data, list):
        return data
    return None


def _cell(value: object) -> object:
    """Coerce a DuckDB result cell into a JSON-serializable value."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time, Decimal)):
        return str(value)
    return str(value)


def _try_load_sqlite(con) -> bool:
    """Load DuckDB's sqlite scanner. Returns True if attaching sqlite will work."""
    for statement in ("LOAD sqlite;", "INSTALL sqlite; LOAD sqlite;"):
        try:
            con.execute(statement)
            return True
        except Exception:
            continue
    return False


class _Registrar:
    def __init__(self, con) -> None:
        self.con = con
        self.held: list = []  # keep DataFrames alive while the connection is open
        self.mapping: dict[str, str] = {}
        self._used: set[str] = set()

    def _unique(self, base: str) -> str:
        candidate = base
        counter = 2
        while candidate in self._used:
            candidate = f"{base}_{counter}"
            counter += 1
        self._used.add(candidate)
        return candidate

    def add_csv(self, path: Path, rel: str) -> None:
        name = self._unique(_safe_ident(path.stem))
        escaped = str(path).replace("'", "''")
        self.con.execute(f'CREATE VIEW "{name}" AS SELECT * FROM read_csv_auto(\'{escaped}\')')
        self.mapping[name] = rel

    def add_json(self, path: Path, rel: str) -> None:
        import pandas as pd

        try:
            with path.open() as handle:
                data = json.load(handle)
        except Exception:
            return
        records = _records(data)
        if records is None:
            return
        frame = pd.DataFrame(records)
        self.held.append(frame)
        name = self._unique(_safe_ident(path.stem))
        self.con.register(name, frame)
        self.mapping[name] = rel

    def add_sqlite_native(self, path: Path, rel: str) -> None:
        escaped = str(path).replace("'", "''")
        alias = self._unique(_safe_ident(path.stem) + "_db")
        self.con.execute(f'ATTACH \'{escaped}\' AS "{alias}" (TYPE sqlite, READ_ONLY)')
        tables = self.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog = ?",
            [alias],
        ).fetchall()
        for (table_name,) in tables:
            view = self._unique(_safe_ident(table_name))
            self.con.execute(f'CREATE VIEW "{view}" AS SELECT * FROM "{alias}"."{table_name}"')
            self.mapping[view] = f"{rel}:{table_name}"

    def add_sqlite_fallback(self, path: Path, rel: str) -> None:
        import pandas as pd

        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (table_name,) in tables:
                frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                self.held.append(frame)
                view = self._unique(_safe_ident(table_name))
                self.con.register(view, frame)
                self.mapping[view] = f"{rel}:{table_name}"
        finally:
            conn.close()


def _register_all(con, context_dir: Path) -> tuple[dict[str, str], list]:
    registrar = _Registrar(con)
    sqlite_native = _try_load_sqlite(con)

    for path in sorted(context_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        rel = path.relative_to(context_dir).as_posix()
        try:
            if ext in _CSV_EXTS:
                registrar.add_csv(path, rel)
            elif ext in _JSON_EXTS:
                registrar.add_json(path, rel)
            elif ext in _DB_EXTS:
                if sqlite_native:
                    registrar.add_sqlite_native(path, rel)
                else:
                    registrar.add_sqlite_fallback(path, rel)
        except Exception:
            # Skip any single file that fails to register; keep the rest usable.
            continue

    return registrar.mapping, registrar.held


def query_files(task: PublicTask, sql: str, *, limit: int = 200) -> dict[str, object]:
    import duckdb

    normalized = sql.lstrip().lower()
    if not normalized.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    con = duckdb.connect()
    try:
        mapping, _held = _register_all(con, task.context_dir)  # _held keeps frames alive
        try:
            cursor = con.execute(sql)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "available_tables": mapping,
            }
        columns = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        limited = rows[:limit]
        return {
            "columns": columns,
            "rows": [[_cell(cell) for cell in row] for row in limited],
            "row_count": len(limited),
            "truncated": truncated,
            "available_tables": mapping,
        }
    finally:
        con.close()
