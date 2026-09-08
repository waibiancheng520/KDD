from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.duckdb_files import query_files
from data_agent_baseline.tools.profile import profile_context
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30

# Name of the reasoning argument injected into every native tool schema. It is
# stripped from the arguments before a handler runs.
THOUGHT_ARG = "thought"

# Window size for document reads. Must stay <= the observation budget in
# react.py, otherwise the tail the model asked for is cut off before it sees it.
DOC_WINDOW_CHARS = 8000


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    # Real JSON Schema for the arguments, used for native function calling. The
    # `input_schema` above stays as the human-readable example rendered into the
    # text prompt for models/back-ends without tool support.
    parameters: dict[str, Any] | None = None


_ANSWER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One column name per attribute the question asks for.",
        },
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": ["string", "number", "boolean", "null"]},
            },
            "description": "Result rows; each row must have exactly len(columns) cells.",
        },
        "evidence_step": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Step number of the successful tool observation that contains every value "
                "submitted in this answer. Failed observations cannot be cited."
            ),
        },
    },
    "required": ["columns", "rows", "evidence_step"],
}


def _params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", DOC_WINDOW_CHARS))
    offset = int(action_input.get("offset", 0))
    return ToolExecutionResult(
        ok=True, content=read_json_preview(task, path, max_chars=max_chars, offset=offset)
    )


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", DOC_WINDOW_CHARS))
    offset = int(action_input.get("offset", 0))
    return ToolExecutionResult(
        ok=True, content=read_doc_preview(task, path, max_chars=max_chars, offset=offset)
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _profile_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 6))
    return ToolExecutionResult(ok=True, content=profile_context(task, max_depth=max_depth))


def _query_files(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    content = query_files(task, sql, limit=limit)
    return ToolExecutionResult(ok="error" not in content, content=content)


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Tool definitions for the OpenAI-compatible function-calling API.

        Every tool gets a required `thought` argument. Without it the model
        answers a tool call with empty content and skips its reasoning entirely,
        which measurably degrades answer quality; making the reasoning part of
        the structured call keeps chain-of-thought while keeping the schema.
        """
        schemas: list[dict[str, Any]] = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            base = spec.parameters or {"type": "object", "properties": {}, "required": []}
            properties = {
                THOUGHT_ARG: {
                    "type": "string",
                    "description": (
                        "Your reasoning for this step: what you learned from the last observation "
                        "and why this call is the right next move. Think it through here BEFORE acting."
                    ),
                },
                **base.get("properties", {}),
            }
            parameters = {
                "type": "object",
                "properties": properties,
                "required": [THOUGHT_ARG, *base.get("required", [])],
            }
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": parameters,
                    },
                }
            )
        return schemas

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
                "evidence_step": 3,
            },
            parameters=_ANSWER_PARAMETERS,
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
            parameters=_params(
                {
                    "path": {"type": "string", "description": "Path relative to the context dir."},
                    "sql": {"type": "string", "description": "Read-only SELECT/WITH/PRAGMA statement."},
                    "limit": {"type": "integer", "default": 200},
                },
                ["path", "sql"],
            ),
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
            parameters=_params(
                {"code": {"type": "string", "description": "Python source to execute; print results."}},
                ["code"],
            ),
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
            parameters=_params({"path": {"type": "string"}}, ["path"]),
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
            parameters=_params({"max_depth": {"type": "integer", "default": 4}}, []),
        ),
        "profile_context": ToolSpec(
            name="profile_context",
            description=(
                "Profile the ENTIRE context directory in ONE call. For each CSV: row count plus, per "
                "column, dtype, unique-value count, null count, sample values, and min/max for numeric "
                "columns. For each JSON: record count, keys, and one sample record. For each doc "
                "(.md/.txt): a text preview that usually contains the schema and foreign-key notes. For "
                "each sqlite/db: the full table schema. Call this FIRST to understand every file at once "
                "instead of listing and reading them one by one."
            ),
            input_schema={"max_depth": 6},
            parameters=_params({"max_depth": {"type": "integer", "default": 6}}, []),
        ),
        "query_files": ToolSpec(
            name="query_files",
            description=(
                "Run one read-only DuckDB SQL query across ALL data files in context at once -- csv, "
                "json, AND sqlite/db together. Every CSV, JSON (in `{\"records\":[...]}` form), and "
                "sqlite table is auto-registered as a table: CSV/JSON by file stem (`csv/member.csv` -> "
                "`member`, `json/zip_code.json` -> `zip_code`), sqlite by its table name (`db/atom.db` "
                "-> `atom`, `connected`). You can JOIN/GROUP BY/aggregate across ALL of them -- even a "
                "CSV joined to a sqlite table -- in a single statement instead of hand-writing python "
                "loops. The result lists `available_tables` (name -> source). Prefer this over "
                "execute_python and execute_context_sql for any querying."
            ),
            input_schema={"sql": "SELECT ... FROM table_a JOIN table_b ...", "limit": 200},
            parameters=_params(
                {
                    "sql": {"type": "string", "description": "Read-only DuckDB SELECT/WITH statement."},
                    "limit": {"type": "integer", "default": 200},
                },
                ["sql"],
            ),
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
            parameters=_params(
                {"path": {"type": "string"}, "max_rows": {"type": "integer", "default": 20}},
                ["path"],
            ),
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a window of a text document inside context. Documents here are LONGER than one "
                "window and the sections that resolve column/metric ambiguities sit near the END, so "
                "when the result says `truncated`, call again with the returned `next_offset` until you "
                "have read the whole file."
            ),
            input_schema={"path": "relative/path/to/file.md", "max_chars": DOC_WINDOW_CHARS, "offset": 0},
            parameters=_params(
                {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": DOC_WINDOW_CHARS},
                    "offset": {"type": "integer", "default": 0, "description": "Start character; use next_offset to page."},
                },
                ["path"],
            ),
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a window of a JSON file inside context; page with `offset` when truncated.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": DOC_WINDOW_CHARS, "offset": 0},
            parameters=_params(
                {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": DOC_WINDOW_CHARS},
                    "offset": {"type": "integer", "default": 0},
                },
                ["path"],
            ),
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "profile_context": _profile_context,
        "query_files": _query_files,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
