"""汇总 Agent 工具行为评估，不调用模型。

结果文件中的每条记录至少包含：
{
  "task_id": "agent_001",
  "status": "completed",
  "tool_names": ["search_knowledge_base"],
  "approval_required": false,
  "sources": [{"filename": "..."}],
  "error": null
}

先运行 ``--template`` 生成记录模板，实际浏览器或 API 演示后补齐结果，
再运行脚本得到 JSON 和 CSV。这样 Agent 工具行为评估与 RAGAS 生成质量评估保持分离。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_metrics(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected_tools = list(task.get("expected_tools", []))
    actual_tools = list(result.get("tool_names", []))
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    forbidden = set(task.get("forbidden_tools", []))
    requires_approval = bool(task.get("requires_approval"))
    approval_required = bool(result.get("approval_required"))
    expected_sources = bool(task.get("expected_sources"))
    has_sources = bool(result.get("sources"))
    status_match = result.get("status") == task.get("expected_status")

    return {
        "task_id": task["id"],
        "tool_selection_correct": actual_set == expected_set,
        "tool_selection_precision": len(expected_set & actual_set) / len(actual_set) if actual_set else 0.0,
        "forbidden_tool_triggered": bool(forbidden & actual_set),
        "approval_trigger_correct": requires_approval == approval_required,
        "source_coverage_correct": expected_sources == has_sources,
        "task_completed_correctly": status_match,
        "tool_call_count": len(actual_tools),
        "error": bool(result.get("error")),
        "status": result.get("status", ""),
    }


def evaluate(tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    result_by_id = {str(item.get("task_id")): item for item in results}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        result = result_by_id.get(task["id"], {})
        rows.append(_task_metrics(task, result))

    count = len(rows)
    average = lambda key: round(sum(float(row[key]) for row in rows) / count, 4) if count else None
    return {
        "sample_count": count,
        "metrics": {
            "tool_selection_accuracy": average("tool_selection_correct"),
            "tool_selection_precision": average("tool_selection_precision"),
            "dangerous_tool_false_positive_rate": round(
                sum(row["forbidden_tool_triggered"] for row in rows) / count, 4
            ) if count else None,
            "approval_trigger_accuracy": average("approval_trigger_correct"),
            "source_coverage": average("source_coverage_correct"),
            "task_completion_rate": average("task_completed_correctly"),
            "average_tool_calls": round(sum(row["tool_call_count"] for row in rows) / count, 4) if count else None,
            "error_rate": average("error"),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总企业知识运营 Agent 工具行为评估")
    parser.add_argument("--tasks", default="eval_data/agent_tasks.json")
    parser.add_argument("--results", default="eval_outputs/agent_results.json")
    parser.add_argument("--output", default="eval_outputs/agent_evaluation.json")
    parser.add_argument("--csv-output", default="eval_outputs/agent_evaluation.csv")
    parser.add_argument("--template", action="store_true", help="生成待填写的结果模板")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    tasks = _load_json(tasks_path)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if args.template:
        template = [
            {
                "task_id": task["id"],
                "status": "",
                "tool_names": [],
                "approval_required": False,
                "sources": [],
                "error": None,
            }
            for task in tasks
        ]
        results_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已生成 Agent 评估结果模板：{results_path}")
        return

    if not results_path.exists():
        raise SystemExit(f"结果文件不存在：{results_path}。先运行 --template 生成模板并填写实际运行结果。")

    report = evaluate(tasks, _load_json(results_path))
    output_path = Path(args.output)
    csv_path = Path(args.csv_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["rows"][0].keys()) if report["rows"] else ["task_id"])
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"JSON：{output_path}")
    print(f"CSV：{csv_path}")


if __name__ == "__main__":
    main()
