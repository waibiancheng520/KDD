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
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


_FULL_OBSERVATION_STEPS = 3
_RECENT_OBSERVATION_CHARS = 4000
_OLD_OBSERVATION_CHARS = 700


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
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        # Older observations are heavily truncated: replaying every full tool dump makes
        # the input grow quadratically, which blows the model's per-minute token quota.
        last_full_index = len(state.steps) - _FULL_OBSERVATION_STEPS
        for step in state.steps:
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
                        max_chars=(
                            _RECENT_OBSERVATION_CHARS
                            if step.step_index > last_full_index
                            else _OLD_OBSERVATION_CHARS
                        ),
                    ),
                )
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            print(f"[步骤 {step_index}/{self.config.max_steps}]", file=sys.stderr, flush=True)
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                print(f"    -> {model_step.action} ok={tool_result.ok}", file=sys.stderr, flush=True)
                if tool_result.is_terminal:
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
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
