#!/usr/bin/env python3
"""为一次 run 的 prediction.csv 打分，对比 data/public/output/<task>/gold.csv。

用法:
    uv run python scripts/score.py artifacts/runs/<run_id>
    uv run python scripts/score.py artifacts/runs/<run_id> --verbose
    uv run python scripts/score.py artifacts/runs/<run_id> --gold-root data/public/output

评分规则（对列名宽松，对数值内容严格）:
  - 列名不参与比较（gold 里常是 `COUNT(*)` 这类 SQL 表达式，agent 无从得知）
  - 行顺序不参与比较（按排序后的多重集合比对）
  - 数字按数值比较（"4" == "4.0" == "4"），其余按去空白后的字符串比较
  - 空值统一视为空串（gold 的缺失 与 pred 的 "" / "None" / "nan" 等价）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_NULLS = {"", "none", "nan", "null", "na", "<na>", "n/a"}


def _norm_cell(value: str) -> str:
    """把单元格normalize成可比较的形式。"""
    text = str(value).strip()
    if text.lower() in _NULLS:
        return ""
    # 去掉数字里的千分位逗号后尝试按数值比较
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text
    if number == int(number):
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _read_table(path: Path) -> list[tuple[str, ...]]:
    """读 CSV，跳过表头，返回normalize后的数据行。"""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    body = rows[1:]  # 首行是表头，不参与比较
    table = []
    for row in body:
        cells = tuple(_norm_cell(cell) for cell in row)
        if any(cells):  # 丢掉完全空行
            table.append(cells)
    return table


def compare(gold: list[tuple[str, ...]], pred: list[tuple[str, ...]]) -> dict:
    """比较两张表，返回是否完全一致以及行级统计。"""
    gold_sorted = sorted(gold)
    pred_sorted = sorted(pred)
    exact = gold_sorted == pred_sorted

    gold_set = set(gold_sorted)
    pred_set = set(pred_sorted)
    matched = gold_set & pred_set
    precision = len(matched) / len(pred_set) if pred_set else 0.0
    recall = len(matched) / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "exact": exact,
        "gold_rows": len(gold),
        "pred_rows": len(pred),
        "matched_rows": len(matched),
        "missing": sorted(gold_set - pred_set)[:5],
        "extra": sorted(pred_set - gold_set)[:5],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def score_run(run_dir: Path, gold_root: Path) -> dict:
    results: list[dict] = []

    task_dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir()),
        key=lambda p: (len(p.name), p.name),
    )
    for task_dir in task_dirs:
        task_id = task_dir.name
        gold_path = gold_root / task_id / "gold.csv"
        pred_path = task_dir / "prediction.csv"
        trace_path = task_dir / "trace.json"

        entry: dict = {"task_id": task_id}

        if not gold_path.exists():
            entry.update(status="no_gold", exact=False)
            results.append(entry)
            continue

        if not pred_path.exists():
            reason = None
            if trace_path.exists():
                try:
                    reason = json.loads(trace_path.read_text()).get("failure_reason")
                except (json.JSONDecodeError, OSError):
                    reason = None
            entry.update(status="no_prediction", exact=False, failure_reason=reason)
            results.append(entry)
            continue

        try:
            gold = _read_table(gold_path)
            pred = _read_table(pred_path)
        except (OSError, csv.Error) as exc:
            entry.update(status="read_error", exact=False, failure_reason=str(exc))
            results.append(entry)
            continue

        entry.update(status="scored", **compare(gold, pred))
        results.append(entry)

    scored = [r for r in results if r["status"] == "scored"]
    correct = [r for r in scored if r["exact"]]
    total = len(results)

    return {
        "run_dir": str(run_dir),
        "total_tasks": total,
        "scored": len(scored),
        "no_prediction": sum(1 for r in results if r["status"] == "no_prediction"),
        "no_gold": sum(1 for r in results if r["status"] == "no_gold"),
        "exact_correct": len(correct),
        "accuracy": round(len(correct) / total, 4) if total else 0.0,
        # 分母是全部任务：没产出答案的题按 F1=0 计，否则失败越多分数反而越好。
        "avg_f1": round(sum(r.get("f1", 0.0) for r in results) / total, 4) if total else 0.0,
        "tasks": results,
    }


def _print_report(report: dict, verbose: bool) -> None:
    print(f"运行目录: {report['run_dir']}")
    print("=" * 68)
    print(f"任务总数    : {report['total_tasks']}")
    print(f"完全正确    : {report['exact_correct']}")
    print(f"未产出答案  : {report['no_prediction']}")
    if report["no_gold"]:
        print(f"缺标准答案  : {report['no_gold']}")
    print(f"准确率      : {report['accuracy']:.1%}")
    print(f"平均行级F1  : {report['avg_f1']:.4f}")
    print("=" * 68)

    for entry in report["tasks"]:
        task_id = entry["task_id"]
        if entry["status"] == "no_prediction":
            reason = (entry.get("failure_reason") or "无答案")[:60]
            print(f"  ✗ {task_id:<12} 未产出  {reason}")
            continue
        if entry["status"] in {"no_gold", "read_error"}:
            print(f"  ? {task_id:<12} {entry['status']}")
            continue

        mark = "✅" if entry["exact"] else "❌"
        print(
            f"  {mark} {task_id:<12} gold={entry['gold_rows']:<4} "
            f"pred={entry['pred_rows']:<4} F1={entry['f1']:.2f}"
        )
        if verbose and not entry["exact"]:
            if entry["missing"]:
                print(f"       漏掉: {entry['missing']}")
            if entry["extra"]:
                print(f"       多给: {entry['extra']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="给一次 run 的预测结果打分")
    parser.add_argument("run_dir", type=Path, help="artifacts/runs/<run_id>")
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=Path("data/public/output"),
        help="标准答案根目录（默认 data/public/output）",
    )
    parser.add_argument("--verbose", action="store_true", help="显示错题的差异明细")
    parser.add_argument("--json", type=Path, help="把完整报告写到这个 json 文件")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"错误: 运行目录不存在: {args.run_dir}", file=sys.stderr)
        return 1
    if not args.gold_root.is_dir():
        print(f"错误: 标准答案目录不存在: {args.gold_root}", file=sys.stderr)
        return 1

    report = score_run(args.run_dir, args.gold_root)
    _print_report(report, args.verbose)

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\n完整报告已写入: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
