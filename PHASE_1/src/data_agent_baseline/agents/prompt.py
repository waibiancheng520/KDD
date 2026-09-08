from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
0. EVERY step must carry non-empty `thought`. Before you act, reason there in one or two sentences: what the last observation told you, and why this is the right next step. Never leave `thought` empty and never skip straight to the action — that reasoning is what keeps your answers correct.
1. Before answering, you MUST actually open and query the real data files (e.g. with `query_files` or `execute_python`). Reading only the documentation is NOT enough.
2. Every ID, value, and cell in your final answer MUST come directly from a tool observation in this run. NEVER invent, guess, or copy numbers from the examples. If you have not seen a value in a tool result, you may not put it in the answer.
3. The task is complete only when you call the `answer` tool. Only call it after you have computed the full result table from the actual data and printed it in a tool observation; the rows you submit must match what you printed.
4. If a step fails with an error, fix it and retry the SAME data query. An error is never a reason to give up and answer from memory. Each observation tells you how many steps remain; use that to pace your investigation.
5. Never write "Observation:", tool results, or the outcome of your own action. The system runs the tool and gives you the real observation. Anything you write about a tool's result is a hallucination and will make your answer wrong.
6. Real data is often incomplete: a join may match only some rows, and some fields may be missing or null. That is a property of the data, not a bug to keep investigating. Report what the data supports (leaving unmatched fields empty) rather than spending steps trying to explain the gap.
7. Reference example solution approaches: If the context contains an example solution approach for this exact question, you may use it as a reference, unless the data clearly contradicts it.
8. Identify the entity the question asks about, then read each requested attribute from the table where that attribute is a defining property of the entity. When the same column name appears in several tables, prefer the one whose rows are one-per-entity over one whose rows are one-per-event or one-per-measurement; the latter belongs in filters and aggregates, not in the output columns.
9. Restrict the result to entities that satisfy every condition in the question. When combining tables, keep only rows present on both sides unless the question explicitly asks to include unmatched ones.
10. Return exactly the attributes the question names, one column per attribute, in the order asked. Do not merge two attributes into one column, split one into several, or add columns that were not requested. Output the MINIMAL projection: only the attribute the question asks you to output. Entities the question uses to FILTER or GROUP are not output columns, and neither is a count you computed along the way, unless the question explicitly asks to list them too. Real examples that scored zero for one extra column:
    - "Tally the toxicology element of the 4th atom" -> output ONLY `element`. Not `element, count` -- "tally" describes the work, not a column.
    - "For all the people who paid more than 29.00 ... give their consumption status" -> output ONLY `Consumption`. Not `CustomerID, Consumption` -- "the people" is the filter, not the requested attribute.
    - "List all the withdrawals ... that client 3356 makes" -> output ONLY `trans_id`. Never `SELECT *`.
    When in doubt, output the single attribute named by the question's main verb and nothing else.
11. NEVER concatenate values from different source columns into a single cell. When the data stores a value across several columns (e.g. `first_name` and `last_name`, or separate home/away score columns), output each source column as its OWN column, in that order — even when the question uses a single word such as "name", "full name", or "score". Do not join them with a space, hyphen, comma, or any other separator. Example: if the members table has first_name='Annabella' and last_name='Warren', the answer is TWO columns with row ['Annabella','Warren'], NOT one cell 'Annabella Warren'. Likewise a paired score is ['1','1'], never '1-1'. This applies to values you read out of a DOCUMENT too, not just table columns: if the underlying dataset models a person's name as first name + last name, answer a "full name" as TWO columns ['Elijah','Allen'] even when the document prints it as the single string "Elijah Allen". When unsure, check how the structured data (csv/json/db) stores that attribute and mirror that split.
12. Do not round, truncate, or reformat numeric values. Copy them into the answer with the FULL precision exactly as printed in the tool observation. If a computed value is 52.173913, answer 52.173913 — never 52.17. Do not change units or reformat (e.g. do not turn a raw number of seconds into a "h:mm:ss" string).
13. Before you call `answer`, sanity-check the result table against the question:
    - Row count: does it match what the question implies? A question like "which three ..." or "who is the top ...?" expects a small, specific number of rows (often 1). If your table has dozens of rows, you almost certainly forgot a filter, a GROUP BY, or a DISTINCT — go back and fix the query instead of dumping the raw table.
    - Columns: include ONLY the attributes the question asks for. Drop helper/intermediate columns such as a GROUP BY key, an id, or a count that the question did not request.
    - Magnitude: if a single number looks off by orders of magnitude (e.g. millions when a small ratio or average was expected), you likely selected the wrong column or skipped a division — re-check before answering.
    Being forced to answer because you are low on steps is NOT a reason to submit a raw, unfiltered table; submit the filtered result you can best justify.

Keep reasoning concise and grounded in the observed data.
""".strip()

# Only sent when the endpoint cannot do native function calling. With tool calling
# the envelope is enforced by the API (and the arguments by each tool's JSON
# Schema), so shipping these formatting rules would just burn tokens every step.
TEXT_MODE_FORMAT_RULES = """
Response format:
- Always return exactly one JSON object with keys `thought`, `action`, and `action_input`, wrapped in exactly one fenced code block that starts with ```json and ends with ```. Do not output any text before or after that block.
- The `answer` action's `action_input` must be a table with `columns` and `rows`.
- When using `execute_python`, keep the code compact and use SINGLE quotes for Python strings. Any double quote or newline inside a JSON string value MUST be escaped as \\" and \\n, otherwise the JSON is invalid and your step fails.
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


def build_system_prompt(
    tool_descriptions: str,
    system_prompt: str | None = None,
    *,
    native_tools: bool = False,
) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    sections = [base_prompt, "Available tools:\n" + tool_descriptions]
    if not native_tools:
        sections.extend([TEXT_MODE_FORMAT_RULES, RESPONSE_EXAMPLES])
    return "\n\n".join(sections)


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
    question: str | None = None,
) -> str:
    # `repeat_warning` is lifted out before truncation: it is the shortest and most
    # urgent part of an observation, and burying it in a body that then gets cut is
    # why the loop guard could fire six times without the model ever reacting.
    warning = None
    if isinstance(observation, dict) and "repeat_warning" in observation:
        # Copy rather than pop: the stored observation is re-rendered on every later
        # step, and mutating it here would drop the warning after its first render.
        warning = observation["repeat_warning"]
        observation = {k: v for k, v in observation.items() if k != "repeat_warning"}
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    if max_chars is not None and len(rendered) > max_chars:
        # Keep BOTH ends. Tool output prints intermediates first and the computed
        # result last, so head-only truncation systematically deletes the answer.
        head = max_chars * 2 // 3
        tail = max_chars - head
        omitted = len(rendered) - max_chars
        rendered = (
            rendered[:head]
            + f"\n... [{omitted} chars omitted from the middle] ...\n"
            + rendered[-tail:]
        )
    prompt = f"Observation:\n{rendered}"
    if warning:
        prompt += f"\n\n!! {warning}"
    if step_index is not None and max_steps is not None:
        remaining = max_steps - step_index
        prompt += f"\n\n(Step {step_index}/{max_steps}. You have {remaining} step(s) left."
        if remaining <= 1:
            prompt += (
                " This is your LAST step. You MUST respond with the `answer` action now, using the best "
                "result you can assemble from what you have already observed. Do not run any more queries."
            )
        elif remaining <= 3:
            prompt += (
                " You are almost out of steps. Stop exploring, and call the `answer` tool with your best "
                "result from the data you already have."
            )
        prompt += ")"
    if question:
        # The question is otherwise stated once, tens of thousands of characters back;
        # restating it here is what keeps late steps aimed at what was actually asked.
        prompt += f'\n\nReminder -- the question you must answer: "{question}"'
    return prompt
