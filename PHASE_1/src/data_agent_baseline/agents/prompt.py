from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Before answering, you MUST actually open and query the real data files (e.g. with `execute_python` or `execute_context_sql`). Reading only the documentation is NOT enough.
2. Every ID, value, and cell in your final answer MUST come directly from a tool observation in this run. NEVER invent, guess, or copy numbers from the examples. If you have not seen a value in a tool result, you may not put it in the answer.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.
8. When using `execute_python`, keep the code compact and use SINGLE quotes for Python strings. Any double quote or newline placed inside a JSON string value MUST be escaped as \" and \n, otherwise the JSON will be invalid and your step will fail.
9. If a step fails with a JSON or code error, fix the format and retry the SAME data query. A parse error is never a reason to give up and answer from memory. Each observation tells you how many steps remain; use that to pace your investigation.
10. Only call `answer` after you have computed the full result table from the actual data and printed it in a tool observation. The rows you submit must match what you printed.
11. Real data is often incomplete: a join may match only some rows, and some fields may be missing or null. That is a property of the data, not a bug to keep investigating. Report what the data supports (leaving unmatched fields empty) rather than spending steps trying to explain the gap.
12. Reference example solution approaches: If the context contains an example solution approach for this exact question, you may use it as a reference, unless the data clearly contradicts it.
13. Output ONE JSON action and then STOP. Never write "Observation:", tool results, or the outcome of your own action. The system runs the tool and gives you the real observation. Anything you write about a tool's result is a hallucination and will make your answer wrong.
14. Identify the entity the question asks about, then read each requested attribute from the table where that attribute is a defining property of the entity. When the same column name appears in several tables, prefer the one whose rows are one-per-entity over one whose rows are one-per-event or one-per-measurement; the latter belongs in filters and aggregates, not in the output columns.
15. Restrict the result to entities that satisfy every condition in the question. When combining tables, keep only rows present on both sides unless the question explicitly asks to include unmatched ones.
16. Return exactly the attributes the question names, one column per attribute, in the order asked. Do not merge two attributes into one column, split one into several, or add columns that were not requested.
17. Do not round off the numerical values,retain full precision.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you run Python (note: use single quotes only, keep it on one line, escape newlines as \n):
```json
{"thought":"Load the JSON and print the columns.","action":"execute_python","action_input":{"code":"import json\np = json.load(open('json/Patient.json'))\nprint(list(p['records'][0].keys()))"}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(
    observation: dict[str, object],
    *,
    step_index: int | None = None,
    max_steps: int | None = None,
    max_chars: int | None = None,
) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    if max_chars is not None and len(rendered) > max_chars:
        omitted = len(rendered) - max_chars
        rendered = rendered[:max_chars] + f"\n... [{omitted} chars truncated]"
    prompt = f"Observation:\n{rendered}"
    if step_index is not None and max_steps is not None:
        remaining = max_steps - step_index
        prompt += f"\n\n(Step {step_index}/{max_steps}. You have {remaining} step(s) left."
        if remaining <= 3:
            prompt += " Submit your best answer from the data you already have before they run out."
        prompt += ")"
    return prompt
