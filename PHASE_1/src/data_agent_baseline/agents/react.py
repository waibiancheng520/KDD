from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


_FULL_OBSERVATION_STEPS = 3
# Must be >= tools.registry.DOC_WINDOW_CHARS, or a document window the model
# explicitly asked for is cut off before it ever reaches the model.
_RECENT_OBSERVATION_CHARS = 9000
# A bounded, runtime-generated memory replaces the full dialogue for old steps.
# It keeps successful evidence separate from failures so errors cannot masquerade
# as facts merely because they appeared earlier in the conversation.
_MEMORY_ITEM_CHARS = 1800
_MEMORY_MAX_VERIFIED = 8
_MEMORY_MAX_ISSUES = 5
# Refuse to execute an action once it has already been issued this many times.
_REPEAT_REFUSE_AFTER = 2


def _action_signature(action: str, action_input: dict[str, object]) -> str:
    """A stable key for detecting when the model repeats the exact same action."""
    try:
        return action + "|" + json.dumps(action_input, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return action + "|" + str(action_input)


def _count_trailing_repeats(steps: list, signature: str) -> int:
    """How many of the immediately preceding (non-error) steps share this signature."""
    repeats = 0
    for step in reversed(steps):
        if step.action == "__error__":
            continue
        if _action_signature(step.action, step.action_input) == signature:
            repeats += 1
        else:
            break
    return repeats


def _short_json(value: object, max_chars: int = _MEMORY_ITEM_CHARS) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(rendered) <= max_chars:
        return rendered
    head = max_chars * 2 // 3
    tail = max_chars - head
    return rendered[:head] + "...[middle omitted]..." + rendered[-tail:]


def _build_compacted_memory(steps: list[StepRecord]) -> str:
    """Build bounded memory whose facts come only from successful tool calls."""
    verified: list[str] = []
    issues: list[str] = []
    for step in steps:
        provenance = step.provenance_id or f"step:{step.step_index}:tool:{step.action}"
        if (
            step.ok
            and step.observation.get("ok") is True
            and step.evidence_status == "verified"
            and step.action != "answer"
        ):
            content = step.observation.get("content")
            verified.append(
                f"- [{provenance}] input={_short_json(step.action_input, 500)} "
                f"result={_short_json(content)}"
            )
        elif step.evidence_status == "error":
            error = step.observation.get("error", step.observation.get("content"))
            issues.append(
                f"- [step:{step.step_index}:tool:{step.action}] FAILED (not evidence): "
                f"{_short_json(error, 700)}"
            )

    verified = verified[-_MEMORY_MAX_VERIFIED:]
    issues = issues[-_MEMORY_MAX_ISSUES:]
    verified_text = "\n".join(verified) if verified else "- (none)"
    issue_text = "\n".join(issues) if issues else "- (none)"
    return (
        "Runtime-generated working memory for older steps. Treat only VERIFIED EVIDENCE as facts.\n"
        "VERIFIED EVIDENCE:\n"
        f"{verified_text}\n"
        "UNRESOLVED ISSUES / FAILED ATTEMPTS:\n"
        f"{issue_text}\n"
        "Failed attempts explain what remains unresolved; they never support an answer."
    )


def _answer_evidence_error(
    action_input: dict[str, object], steps: list[StepRecord]
) -> str | None:
    """Return why an answer is unsupported, or None when its provenance is valid."""
    evidence_step = action_input.get("evidence_step")
    if not isinstance(evidence_step, int) or isinstance(evidence_step, bool):
        return "answer.evidence_step must name one successful prior tool step."
    source = next((step for step in steps if step.step_index == evidence_step), None)
    if source is None:
        return f"Evidence step {evidence_step} does not exist."
    if (
        not source.ok
        or source.observation.get("ok") is not True
        or source.evidence_status != "verified"
        or source.action == "answer"
    ):
        return f"Step {evidence_step} is not verified evidence."

    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return "Answer columns and rows must be lists."
    content = source.observation.get("content")
    if isinstance(content, dict):
        source_columns = content.get("columns")
        source_rows = content.get("rows")
        if isinstance(source_columns, list) and isinstance(source_rows, list):
            if source_columns == columns and source_rows == rows:
                return None
            return (
                f"Answer table does not exactly match structured evidence step {evidence_step}."
            )

    # Python/document tools may return a rendered table instead of structured rows.
    # In that case every submitted cell must still occur in the cited observation.
    rendered = json.dumps(content, ensure_ascii=False, default=str)
    for column in columns:
        if not isinstance(column, str) or column not in rendered:
            return (
                f"Column {column!r} is absent from cited evidence step {evidence_step}. "
                "Print the complete final table before answering."
            )
    for row in rows:
        if not isinstance(row, list):
            return "Each answer row must be a list."
        for cell in row:
            if cell is None:
                candidates = ("null", "None", "<NA>")
            else:
                candidates = (str(cell), json.dumps(cell, ensure_ascii=False, default=str))
            if not any(candidate in rendered for candidate in candidates):
                return (
                    f"Value {cell!r} is absent from cited evidence step {evidence_step}. "
                    "Run a tool that prints the complete final table, then cite that step."
                )
    return None


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder(strict=False)
    # Some models (e.g. DeepSeek) prepend a natural-language preamble before the
    # JSON action, like: "I need to explore the context.\n\n{...}". raw_decode from
    # char 0 would fail on that leading prose, so scan forward to the first '{' that
    # parses as a JSON object and decode from there. Any prose before or after the
    # object is ignored — the real observation is fed back separately by the runtime,
    # so trailing model text can never be mistaken for a tool result.
    start = 0
    last_error: Exception | None = None
    while True:
        brace = text.find("{", start)
        if brace == -1:
            if last_error is not None:
                raise last_error
            raise ValueError("Model response must contain a JSON object.")
        try:
            payload, _ = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError as exc:
            last_error = exc
            start = brace + 1
            continue
        if not isinstance(payload, dict):
            start = brace + 1
            continue
        return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
            native_tools=getattr(self.model, "supports_native_tools", False),
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        # Replace old dialogue with one bounded, typed memory block. Recent steps stay
        # verbatim so the model can repair the latest failure without losing detail.
        old_steps = state.steps[:-_FULL_OBSERVATION_STEPS]
        recent_steps = state.steps[-_FULL_OBSERVATION_STEPS:]
        if old_steps:
            messages.append(
                ModelMessage(role="user", content=_build_compacted_memory(old_steps))
            )
        for step in recent_steps:
            # Feed back ONLY the single action the model actually took, not its full
            # raw output. The raw output may contain hallucinated "observations" the
            # model wrote itself; replaying those would let it trust its own fabricated
            # data instead of the real tool results below.
            if step.action == "__error__":
                assistant_content = "(Previous response could not be parsed as a single JSON action.)"
            else:
                assistant_content = json.dumps(
                    {
                        "thought": step.thought,
                        "action": step.action,
                        "action_input": step.action_input,
                    },
                    ensure_ascii=False,
                )
            messages.append(ModelMessage(role="assistant", content=assistant_content))
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(
                        step.observation,
                        step_index=step.step_index,
                        max_steps=self.config.max_steps,
                        max_chars=_RECENT_OBSERVATION_CHARS,
                        question=task.question if step.step_index == len(state.steps) else None,
                    ),
                )
            )
        return messages

    def _forced_answer_turn(self, task: PublicTask, state: AgentRuntimeState) -> None:
        """One extra turn in which `answer` is the only action that will be run."""
        print("[收尾] 强制提交轮 (仅开放 answer)", file=sys.stderr, flush=True)
        messages = self._build_messages(task, state)
        messages.append(
            ModelMessage(
                role="user",
                content=(
                    "You are out of investigation steps. Respond NOW with the `answer` action and "
                    "nothing else, using the best result table you can assemble from observations you "
                    "have already seen. Any other action will be discarded. Follow the answer rules: "
                    "only the columns the question asks for, one source column per output column, and "
                    "full numeric precision. You MUST include `evidence_step`, citing the successful "
                    "observation that printed the complete table."
                ),
            )
        )
        answer_only = [
            schema
            for schema in self.tools.openai_tool_schemas()
            if schema.get("function", {}).get("name") == "answer"
        ]
        try:
            raw_response = self.model.complete(messages, answer_only)
            model_step = parse_model_step(raw_response)
            if model_step.action != "answer":
                return
            evidence_error = _answer_evidence_error(model_step.action_input, state.steps)
            if evidence_error is not None:
                print(f"[收尾] 强制提交证据不足: {evidence_error}", file=sys.stderr, flush=True)
                return
            tool_result = self.tools.execute(task, "answer", model_step.action_input)
        except Exception as exc:  # a failed rescue must not mask the original outcome
            print(f"[收尾] 强制提交失败: {exc}", file=sys.stderr, flush=True)
            return
        if tool_result.is_terminal and tool_result.answer is not None:
            state.answer = tool_result.answer
            state.steps.append(
                StepRecord(
                    step_index=len(state.steps) + 1,
                    thought="(forced answer turn)",
                    action="answer",
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation={"ok": True, "tool": "answer", "content": tool_result.content},
                    ok=True,
                    evidence_status="terminal",
                    provenance_id=f"step:{len(state.steps) + 1}:tool:answer",
                )
            )
            print("[收尾] 强制提交成功", file=sys.stderr, flush=True)

    def _salvage_answer(self, state: AgentRuntimeState) -> None:
        """Last resort: submit the most recent verified tabular observation.

        Failed observations are deliberately ineligible. This preserves the evidence
        boundary even when the model runs out of steps before calling `answer`.
        """
        for step in reversed(state.steps):
            if not step.ok or step.evidence_status != "verified" or step.action == "answer":
                continue
            content = (step.observation or {}).get("content")
            if not isinstance(content, dict):
                continue
            columns, rows = content.get("columns"), content.get("rows")
            if not isinstance(columns, list) or not columns:
                continue
            if not isinstance(rows, list) or not rows:
                continue
            state.answer = AnswerTable(
                columns=[str(column) for column in columns],
                rows=[list(row) for row in rows if isinstance(row, list)],
            )
            state.failure_reason = None
            print(
                f"[收尾] 打捞第{step.step_index}步的表格作为答案 ({len(rows)}行)",
                file=sys.stderr,
                flush=True,
            )
            return

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            print(f"[步骤 {step_index}/{self.config.max_steps}]", file=sys.stderr, flush=True)
            raw_response = self.model.complete(
                self._build_messages(task, state),
                self.tools.openai_tool_schemas(),
            )
            try:
                model_step = parse_model_step(raw_response)
                signature = _action_signature(model_step.action, model_step.action_input)
                prior_repeats = (
                    0
                    if model_step.action == "answer"
                    else _count_trailing_repeats(state.steps, signature)
                )

                # Death-loop guard. Warning alone provably does not work -- runs exist
                # where the model repeated one call through six escalating warnings --
                # so past the second repeat we refuse to run the tool at all. Returning
                # no new data is what actually forces a different move.
                if prior_repeats >= _REPEAT_REFUSE_AFTER:
                    tool_result = None
                    observation = {
                        "ok": False,
                        "tool": model_step.action,
                        "error": (
                            f"REFUSED: this is the same `{model_step.action}` call you already made "
                            f"{prior_repeats} times in a row. It was not executed, and repeating it "
                            "again will be refused too. Either take a genuinely DIFFERENT action, or "
                            "call `answer` now with the best result from what you have already seen."
                        ),
                    }
                else:
                    evidence_error = (
                        _answer_evidence_error(model_step.action_input, state.steps)
                        if model_step.action == "answer"
                        else None
                    )
                    if evidence_error is not None:
                        tool_result = None
                        observation = {
                            "ok": False,
                            "tool": "answer",
                            "error": "UNSUPPORTED ANSWER: " + evidence_error,
                        }
                    else:
                        tool_result = self.tools.execute(
                            task, model_step.action, model_step.action_input
                        )
                        observation = {
                            "ok": tool_result.ok,
                            "tool": model_step.action,
                            "content": tool_result.content,
                        }
                    if (
                        tool_result is not None
                        and not tool_result.is_terminal
                        and prior_repeats >= 1
                    ):
                        observation["repeat_warning"] = (
                            f"You have issued this SAME action {prior_repeats + 1} times in a row and are "
                            "stuck in a loop. Do NOT repeat it again. Take a DIFFERENT action, or call the "
                            "`answer` tool right now with the best result you can build from data you have "
                            "already observed."
                        )
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=bool(tool_result and tool_result.ok),
                    evidence_status=(
                        "terminal"
                        if tool_result is not None and tool_result.is_terminal
                        else "verified"
                        if tool_result is not None and tool_result.ok
                        else "error"
                    ),
                    provenance_id=f"step:{step_index}:tool:{model_step.action}",
                )
                state.steps.append(step_record)
                if tool_result is not None:
                    status_suffix = ""
                elif observation.get("tool") == "answer":
                    status_suffix = " [rejected: unsupported evidence]"
                else:
                    status_suffix = " [refused: repeat]"
                print(
                    f"    -> {model_step.action} ok={bool(tool_result and tool_result.ok)}"
                    + status_suffix,
                    file=sys.stderr,
                    flush=True,
                )
                if tool_result is not None and tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                        evidence_status="error",
                        provenance_id=f"step:{step_index}:tool:__error__",
                    )
                )

        # Running out of steps used to score a hard zero even when the result was
        # already sitting in the last observation. Two fallbacks now stand between
        # that and an empty submission.
        if state.answer is None:
            self._forced_answer_turn(task, state)
        if state.answer is None:
            self._salvage_answer(state)

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
