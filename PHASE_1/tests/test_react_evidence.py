from pathlib import Path

from data_agent_baseline.agents.react import (
    ReActAgent,
    ReActAgentConfig,
    _answer_evidence_error,
    _build_compacted_memory,
)
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


class _SequenceModel:
    supports_native_tools = False

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def complete(self, messages: list[object], tools: list[dict] | None = None) -> str:
        return next(self._responses)


def _step(
    index: int,
    *,
    ok: bool,
    content: object = None,
    error: str | None = None,
) -> StepRecord:
    observation = {"ok": ok, "tool": "query_files"}
    if error is not None:
        observation["error"] = error
    else:
        observation["content"] = content
    return StepRecord(
        step_index=index,
        thought="test",
        action="query_files",
        action_input={"sql": "SELECT 1"},
        raw_response="",
        observation=observation,
        ok=ok,
        evidence_status="verified" if ok else "error",
        provenance_id=f"step:{index}:tool:query_files",
    )


def test_answer_accepts_exact_table_from_verified_step() -> None:
    steps = [_step(1, ok=True, content={"columns": ["value"], "rows": [[42]]})]

    error = _answer_evidence_error(
        {"columns": ["value"], "rows": [[42]], "evidence_step": 1}, steps
    )

    assert error is None


def test_answer_accepts_equivalent_types_order_and_column_alias() -> None:
    steps = [
        _step(
            1,
            ok=True,
            content={"columns": ["count_star()"], "rows": [[42], [7]]},
        )
    ]

    error = _answer_evidence_error(
        {"columns": ["count"], "rows": [["7.0"], ["42"]], "evidence_step": 1},
        steps,
    )

    assert error is None


def test_answer_accepts_supported_projection() -> None:
    steps = [
        _step(
            1,
            ok=True,
            content={
                "columns": ["name", "district", "phone"],
                "rows": [["School A", "District A", "555-0100"]],
            },
        )
    ]

    error = _answer_evidence_error(
        {"columns": ["Phone"], "rows": [["555-0100"]], "evidence_step": 1},
        steps,
    )

    assert error is None


def test_answer_rejects_failed_or_mismatched_evidence() -> None:
    failed = [_step(1, ok=False, error="query failed")]
    answer = {"columns": ["value"], "rows": [[42]], "evidence_step": 1}
    assert "not verified" in (_answer_evidence_error(answer, failed) or "")

    verified = [_step(1, ok=True, content={"columns": ["value"], "rows": [[41]]})]
    assert "not supported" in (_answer_evidence_error(answer, verified) or "")


def test_answer_requires_observation_ok_even_if_step_flag_is_inconsistent() -> None:
    inconsistent = _step(1, ok=True, content={"columns": ["value"], "rows": [[42]]})
    inconsistent.observation["ok"] = False

    error = _answer_evidence_error(
        {"columns": ["value"], "rows": [[42]], "evidence_step": 1}, [inconsistent]
    )

    assert "not verified" in (error or "")


def test_compacted_memory_separates_evidence_from_failures() -> None:
    memory = _build_compacted_memory(
        [
            _step(1, ok=True, content={"columns": ["value"], "rows": [[42]]}),
            _step(2, ok=False, error="bad column"),
        ]
    )

    assert "VERIFIED EVIDENCE" in memory
    assert "step:1:tool:query_files" in memory
    assert "UNRESOLVED ISSUES" in memory
    assert "bad column" in memory
    assert "FAILED (not evidence)" in memory


def test_agent_salvages_unsupported_answer_instead_of_returning_empty(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    task = PublicTask(
        record=TaskRecord(task_id="task_test", difficulty="demo", question="Return value."),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    unsupported_answer = (
        '{"thought":"guess","action":"answer","action_input":'
        '{"columns":["value"],"rows":[[999]],"evidence_step":1}}'
    )
    model = _SequenceModel(
        [
            '{"thought":"inspect","action":"list_context","action_input":{}}',
            unsupported_answer,
            unsupported_answer,
        ]
    )
    agent = ReActAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=2),
    )

    result = agent.run(task)

    assert result.answer is not None
    assert result.answer.rows == [[999]]
    assert result.failure_reason is None
    assert result.steps[0].evidence_status == "verified"
    assert result.steps[1].evidence_status == "error"
    assert "UNSUPPORTED ANSWER" in result.steps[1].observation["error"]


def test_agent_writes_sentinel_when_no_answer_or_table_exists(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    task = PublicTask(
        record=TaskRecord(task_id="task_test", difficulty="demo", question="Return value."),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    list_action = '{"thought":"inspect","action":"list_context","action_input":{}}'
    agent = ReActAgent(
        model=_SequenceModel([list_action, list_action]),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=1),
    )

    result = agent.run(task)

    assert result.answer is not None
    assert result.answer.rows == [["NO_VERIFIED_RESULT"]]
    assert result.failure_reason is None
