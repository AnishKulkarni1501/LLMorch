"""Paired benchmark: direct tool use versus the real workflow orchestrator.

This is an API-style local tool benchmark. It does not make Gemini calls unless
explicitly requested through the command line; model usage is reported separately.
"""
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ..models import Status, TaskSpec, Workflow
from ..orchestration.engine import execute
from ..orchestration.llm_planner import configured_cost, configured_rates
from ..orchestration.planner import plan
from ..orchestration.tools import ToolRegistry


TASKS: list[dict[str, Any]] = [
    {"id": "search", "task": "Search research documents about MCP tools", "expected_tool": "local.search", "args": {"query": "MCP tools"}, "expected_contains": "MCP tools"},
    {"id": "sum", "task": "Calculate the sum of 4 and 5", "expected_tool": "local.analysis", "args": {"operation": "sum", "values": [4, 5]}, "expected": {"operation": "sum", "value": 9}},
    {"id": "report", "task": "Write a report on graph scheduling", "expected_tool": "local.document", "args": {"title": "Graph scheduling", "findings": ["DAG schedulers run independent tasks concurrently and wait for prerequisites."], "sources": []}, "expected_contains": "Graph scheduling"},
    {"id": "research_then_report", "task": "Research failure recovery then write a report based on the findings", "expected_tool": "local.document", "args": {}, "expected_contains": "Failure recovery"},
    {"id": "find_dag", "task": "Find information about DAG scheduling", "expected_tool": "local.search", "args": {"query": "DAG scheduling"}, "expected_contains": "DAG schedulers"},
    {"id": "mean", "task": "Calculate the average of 2 4 6", "expected_tool": "local.analysis", "args": {"operation": "mean", "values": [2, 4, 6]}, "expected": {"operation": "mean", "value": 4}},
    {"id": "text", "task": "Count words and find frequent terms in this text: DAG tasks can run concurrently", "expected_tool": "local.text", "args": {"text": "DAG tasks can run concurrently"}, "expected": {"word_count": 5}},
    {"id": "dataset", "task": "Summarize dataset values 3, 7, null, 10", "expected_tool": "local.dataset", "args": {"values": [3, 7, None, 10], "units": []}, "expected": {"count": 4, "missing_count": 1, "sum": 20}},
    {"id": "units", "task": "Convert 5 km to miles", "expected_tool": "local.units", "args": {"value": 5, "from_unit": "km", "to_unit": "mi"}, "expected_approx": {"path": ["value"], "value": 5 / 1.609344, "tolerance": 1e-6}},
    {"id": "files", "task": "List files in the user documents folder", "expected_tool": "local.files", "args": {"action": "list"}, "expected": {"documents": []}},
]

# Simulated API/MCP round-trip; this is explicitly not a real provider measurement.
SIMULATED_API_LATENCY_SECONDS = 0.025


class BenchmarkRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.invalid_once: set[str] = set()

    async def invoke(self, name: str, args: dict[str, Any], timeout: float = 10) -> Any:
        await asyncio.sleep(SIMULATED_API_LATENCY_SECONDS)
        if name in self.invalid_once:
            self.invalid_once.remove(name)
            self.calls += 1
            return {"malformed": True}
        return await super().invoke(name, args, timeout)


STRESS_CASES: list[dict[str, Any]] = [
    {"id": "fanout_fanin", "description": "Run three independent analyses, then create a report from all results.", "kind": "fanout_fanin"},
    {"id": "transient_timeout", "description": "Search and recover from one temporary tool timeout.", "kind": "transient_timeout"},
    {"id": "invalid_then_valid", "description": "Reject one malformed tool response and recover with a valid retry.", "kind": "invalid_then_valid"},
    {"id": "blocked_dependency", "description": "Do not create a report when its required search fails persistently.", "kind": "blocked_dependency"},
]


def _stress_nodes(case: dict[str, Any]) -> list[TaskSpec]:
    if case["kind"] == "fanout_fanin":
        return [
            TaskSpec(task_id="sum", description="sum", selected_tool="local.analysis", input_data={"operation": "sum", "values": [2, 3]}),
            TaskSpec(task_id="mean", description="mean", selected_tool="local.analysis", input_data={"operation": "mean", "values": [4, 6]}),
            TaskSpec(task_id="convert", description="convert", selected_tool="local.units", input_data={"value": 5, "from_unit": "km", "to_unit": "mi"}),
            TaskSpec(task_id="report", description="report from all branches", selected_tool="local.document", dependencies=["sum", "mean", "convert"], input_data={"title": "Combined results", "findings": [], "sources": []}),
        ]
    if case["kind"] in {"transient_timeout", "invalid_then_valid"}:
        return [TaskSpec(task_id="search", description="search MCP", selected_tool="local.search", input_data={"query": "MCP tools"})]
    return [
        TaskSpec(task_id="search", description="required search", selected_tool="local.search", input_data={"query": "MCP"}),
        TaskSpec(task_id="report", description="report using search findings", selected_tool="local.document", dependencies=["search"], input_data={"title": "Should be blocked", "findings": [], "sources": []}),
    ]


async def _run_stress_direct(case: dict[str, Any]) -> dict[str, Any]:
    """Naive serial pipeline: same steps/arguments, no validation or retries."""
    registry = BenchmarkRegistry()
    nodes = _stress_nodes(case)
    if case["kind"] == "transient_timeout":
        registry.fail_once.add("local.search")
    elif case["kind"] == "invalid_then_valid":
        registry.invalid_once.add("local.search")
    elif case["kind"] == "blocked_dependency":
        # A persistent failure models an unavailable required upstream service.
        registry.fail_once.add("local.search")
    started = time.perf_counter()
    results: dict[str, Any] = {}
    errors: list[str] = []
    for node in nodes:
        if any(dep not in results for dep in node.dependencies):
            errors.append(f"blocked: {node.task_id} prerequisite missing")
            continue
        args = dict(node.input_data)
        if node.selected_tool == "local.document" and node.dependencies:
            prior = [results[d] for d in node.dependencies]
            args["findings"] = [json.dumps(value, sort_keys=True, default=str) for value in prior]
        try:
            results[node.task_id] = await registry.invoke(node.selected_tool, args)
        except Exception as exc:
            errors.append(f"{node.task_id}: {type(exc).__name__}: {exc}")
            # Naive code continues without recording a usable prerequisite result.
    elapsed = time.perf_counter() - started
    result = results.get("report") if case["kind"] == "fanout_fanin" else results.get("search")
    if case["kind"] == "fanout_fanin":
        correct = isinstance(result, dict) and all(k in str(result.get("report", "")) for k in ('"value": 5', '"value": 5.0')) and "Combined results" in result.get("report", "")
        # Structured representation of all three successful prerequisite payloads is the expectation.
        correct = isinstance(result, dict) and all(token in result.get("report", "") for token in ('"operation": "sum"', '"operation": "mean"', '"to_unit": "mi"'))
    elif case["kind"] == "transient_timeout":
        correct = False
    elif case["kind"] == "invalid_then_valid":
        correct = isinstance(result, dict) and isinstance(result.get("results"), list)
    else:
        correct = not any(n.startswith("report:") for n in errors) and "report" not in results
    completed = not errors and len(results) == len(nodes)
    return {"completed": completed, "correct": correct, "latency_seconds": elapsed,
            "tool_calls": registry.calls, "recovery_attempted": case["kind"] in {"transient_timeout", "invalid_then_valid"},
            "recovery_success": False, "validation_failures": 0, "validation_checks": 0,
            "failure_propagated": case["kind"] == "blocked_dependency" and "report" not in results,
            "error_count": len(errors), "errors": errors}


async def _run_stress_orchestrator(case: dict[str, Any]) -> dict[str, Any]:
    registry = BenchmarkRegistry()
    nodes = _stress_nodes(case)
    workflow = Workflow(task=case["description"], nodes=nodes, max_concurrency=4, max_retries=1)
    if case["kind"] in {"transient_timeout", "blocked_dependency"}:
        registry.fail_once.add("local.search")
        if case["kind"] == "blocked_dependency":
            # The retry also fails: replace the registry invocation behavior to fail persistently.
            registry.fail_once.add("local.search")
            original_invoke = registry.invoke
            async def fail_search_twice(name: str, args: dict[str, Any], timeout: float = 10):
                if name == "local.search":
                    await asyncio.sleep(SIMULATED_API_LATENCY_SECONDS)
                    registry.calls += 1
                    raise TimeoutError("persistent injected failure")
                return await original_invoke(name, args, timeout)
            registry.invoke = fail_search_twice
    elif case["kind"] == "invalid_then_valid":
        registry.invalid_once.add("local.search")
    started = time.perf_counter()
    try:
        await execute(workflow, registry)
    except Exception:
        pass
    elapsed = time.perf_counter() - started
    if case["kind"] == "fanout_fanin":
        result = workflow.nodes[-1].output_data
        correct = workflow.status == Status.completed and isinstance(result, dict) and all(token in result.get("report", "") for token in ('"operation": "sum"', '"operation": "mean"', '"to_unit": "mi"'))
    elif case["kind"] == "transient_timeout":
        result = workflow.nodes[0].output_data
        correct = workflow.status == Status.completed and isinstance(result, dict) and isinstance(result.get("results"), list)
    elif case["kind"] == "invalid_then_valid":
        result = workflow.nodes[0].output_data
        correct = workflow.status == Status.completed and isinstance(result, dict) and isinstance(result.get("results"), list)
    else:
        correct = workflow.status == Status.failed and workflow.nodes[0].status == Status.failed and workflow.nodes[1].status == Status.failed and workflow.nodes[1].output_data is None
    checks = sum(1 + n.retry_count for n in workflow.nodes if n.status in {Status.completed, Status.failed})
    invalids = sum(1 for n in workflow.nodes if any("required output fields missing" in event for event in n.recovery_history))
    recovered = case["kind"] in {"transient_timeout", "invalid_then_valid"} and any(n.retry_count for n in workflow.nodes) and correct
    propagated = case["kind"] == "blocked_dependency" and workflow.nodes[1].output_data is None and workflow.nodes[1].status == Status.failed
    return {"completed": workflow.status == Status.completed, "correct": correct, "latency_seconds": elapsed,
            "tool_calls": registry.calls, "recovery_attempted": case["kind"] in {"transient_timeout", "invalid_then_valid"},
            "recovery_success": bool(recovered), "validation_failures": invalids, "validation_checks": checks,
            "failure_propagated": propagated, "error_count": sum(n.status == Status.failed for n in workflow.nodes),
            "workflow": workflow}


def _check(value: Any, expected: dict[str, Any] | None, contains: str | None, approx: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    if expected and any(value.get(key) != wanted for key, wanted in expected.items()):
        return False
    if contains and contains.lower() not in json.dumps(value, ensure_ascii=False).lower():
        return False
    if approx:
        actual: Any = value
        for key in approx["path"]:
            actual = actual.get(key) if isinstance(actual, dict) else None
        if not isinstance(actual, (int, float)) or abs(actual - approx["value"]) > approx["tolerance"]:
            return False
    return True


def _dependent_workflow(task: dict[str, Any], registry: ToolRegistry) -> Workflow:
    if task["id"] == "research_then_report":
        first = TaskSpec(task_id="search", description="Search failure recovery", selected_tool="local.search", input_data={"query": "failure recovery"})
        second = TaskSpec(task_id="report", description="Report search findings", selected_tool="local.document", dependencies=["search"], input_data={"title": "Failure recovery", "findings": [], "sources": []})
        return Workflow(task=task["task"], nodes=[first, second], max_concurrency=4, max_retries=1)
    nodes = plan(task["task"], registry)
    if task["id"] == "report":
        nodes[-1].input_data.update(task["args"])
    return Workflow(task=task["task"], nodes=nodes, max_concurrency=4, max_retries=1)


async def _run_baseline(item: dict[str, Any], *, inject_failure: bool = False) -> dict[str, Any]:
    """Direct equivalent API client: caller selects and invokes exact tool(s)."""
    registry = BenchmarkRegistry()
    if inject_failure:
        registry.fail_once.add(item["expected_tool"])
    started = time.perf_counter()
    validation_failures = 0
    calls_before = registry.calls
    recovery = False
    try:
        if item["id"] == "research_then_report":
            search = await registry.invoke("local.search", {"query": "failure recovery"})
            # Deliberately inject the report failure on its first attempt, mirroring the orchestrator case.
            report_args = {"title": "Failure recovery", "findings": [r["text"] for r in search["results"]], "sources": [r["title"] for r in search["results"]]}
            if inject_failure:
                registry.fail_once.add("local.document")
            try:
                report = await registry.invoke("local.document", report_args)
            except Exception:
                if not inject_failure:
                    raise
                recovery = True
                report = await registry.invoke("local.document", report_args)
            output = report
        elif inject_failure:
            try:
                output = await registry.invoke(item["expected_tool"], item["args"])
            except Exception:
                recovery = True
                output = await registry.invoke(item["expected_tool"], item["args"])
        else:
            output = await registry.invoke(item["expected_tool"], item["args"])
    except Exception:
        output = None
    elapsed = time.perf_counter() - started
    correct = _check(output, item.get("expected"), item.get("expected_contains"), item.get("expected_approx"))
    if item["id"] == "research_then_report":
        correct = isinstance(output, dict) and "Failure recovery" in str(output.get("report", ""))
    return {"completed": output is not None, "correct": correct, "latency_seconds": elapsed,
            "tool_calls": registry.calls - calls_before, "recovery_attempted": inject_failure,
            "recovery_success": recovery and correct, "validation_failures": validation_failures,
            "validation_checks": 0, "output": output, "observed_tool_latency_seconds": elapsed}


async def _run_orchestrator(item: dict[str, Any], *, inject_failure: bool = False, concurrency: int = 4) -> dict[str, Any]:
    registry = BenchmarkRegistry()
    workflow = _dependent_workflow(item, registry)
    workflow.max_concurrency = concurrency
    if inject_failure:
        target = "local.document" if item["id"] == "research_then_report" else item["expected_tool"]
        registry.fail_once.add(target)
    started = time.perf_counter()
    try:
        output = await execute(workflow, registry)
    except Exception:
        output = None
    elapsed = time.perf_counter() - started
    result = workflow.nodes[-1].output_data if workflow.nodes else output
    correct = _check(result, item.get("expected"), item.get("expected_contains"), item.get("expected_approx"))
    if item["id"] == "research_then_report":
        correct = isinstance(result, dict) and "Failure recovery" in str(result.get("report", ""))
    attempts = sum(1 + n.retry_count for n in workflow.nodes if n.status in {Status.completed, Status.failed})
    failures = sum(1 for n in workflow.nodes if any("required output fields missing" in event for event in n.recovery_history))
    recovered = inject_failure and any(n.retry_count > 0 and n.status == Status.completed for n in workflow.nodes)
    return {"completed": workflow.status == Status.completed, "correct": correct,
            "latency_seconds": elapsed, "tool_calls": registry.calls,
            "recovery_attempted": inject_failure, "recovery_success": bool(recovered and correct),
            "validation_failures": failures, "validation_checks": attempts,
            "output": result, "observed_tool_latency_seconds": sum(n.observed_latency for n in workflow.nodes),
            "workflow": workflow}


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    latencies = [r["latency_seconds"] for r in rows]
    completed = sum(bool(r["completed"]) for r in rows)
    correct = sum(bool(r["correct"]) for r in rows)
    attempted = [r for r in rows if r["recovery_attempted"]]
    recovered = sum(bool(r["recovery_success"]) for r in attempted)
    checks = sum(r["validation_checks"] for r in rows)
    fails = sum(r["validation_failures"] for r in rows)
    return {"task_count": count, "task_completion_rate": completed / count if count else 0,
            "correct_completion_rate": correct / count if count else 0,
            "recovery_success_rate": recovered / len(attempted) if attempted else None,
            "recovery_attempts": len(attempted), "recovered_failures": recovered,
            "mean_end_to_end_latency_seconds": statistics.mean(latencies) if latencies else 0,
            "median_end_to_end_latency_seconds": statistics.median(latencies) if latencies else 0,
            "p95_end_to_end_latency_seconds": sorted(latencies)[max(0, int(.95 * len(latencies) + .999) - 1)] if latencies else 0,
            "tool_invocations": sum(r["tool_calls"] for r in rows),
            "mean_tool_invocations_per_task": statistics.mean(r["tool_calls"] for r in rows) if rows else 0,
            "validation_failure_rate": fails / checks if checks else None,
            "validation_failures": fails, "validation_checks": checks,
            "model_tokens": 0, "model_cost_usd": 0.0,
            "tool_cost_usd": 0.0, "total_execution_cost_usd": 0.0,
            "cost_basis": "local tools; no model call; local tool cost set to zero"}


async def run_benchmarks(trials: int = 1, *, failure_trials: int = 1) -> dict[str, Any]:
    """Run paired baseline/orchestrator tasks, plus separate injected-failure cases."""
    rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    pair = 0
    for trial in range(max(1, trials)):
        for item in TASKS:
            pair += 1
            # Alternate run order across pairs to reduce systematic warm-up bias.
            actions = [("direct_api", _run_baseline), ("orchestrator", _run_orchestrator)]
            if pair % 2 == 0:
                actions.reverse()
            results: dict[str, dict[str, Any]] = {}
            for name, runner in actions:
                result = await runner(item)
                results[name] = result
                rows.append({"trial": trial, "task_id": item["id"], "task": item["task"], "scenario": "normal", "system": name,
                             **{k: v for k, v in result.items() if k not in {"output", "workflow"}}})
            # Failure recovery is a dedicated scenario, not folded into ordinary success rates.
            for name, runner in (("direct_api", _run_baseline), ("orchestrator", _run_orchestrator)):
                result = await runner(item, inject_failure=True)
                rows.append({"trial": trial, "task_id": item["id"], "task": item["task"], "scenario": "one_transient_failure", "system": name,
                             **{k: v for k, v in result.items() if k not in {"output", "workflow"}}})
        for case in STRESS_CASES:
            for system, runner in (("direct_api", _run_stress_direct), ("orchestrator", _run_stress_orchestrator)):
                result = await runner(case)
                stress_rows.append({"trial": trial, "case_id": case["id"], "case": case["description"], "system": system,
                                    **{k: v for k, v in result.items() if k not in {"workflow", "errors"}}})
    # Compare independent calls serially vs concurrently using simulated API latency.
    parallel_rows = []
    for trial in range(max(1, trials)):
        independent = [
            TaskSpec(task_id="a", description="calculate sum", selected_tool="local.analysis", input_data={"operation": "sum", "values": [4, 5]}),
            TaskSpec(task_id="b", description="convert units", selected_tool="local.units", input_data={"value": 5, "from_unit": "km", "to_unit": "mi"}),
        ]
        for concurrency in (1, 4):
            registry = BenchmarkRegistry()
            wf = Workflow(task="Two independent local operations", nodes=[n.model_copy(deep=True) for n in independent], max_concurrency=concurrency, max_retries=0)
            started = time.perf_counter(); await execute(wf, registry); elapsed = time.perf_counter() - started
            parallel_rows.append({"trial": trial, "concurrency": concurrency, "latency_seconds": elapsed,
                                  "completed": wf.status == Status.completed, "tool_calls": registry.calls})
    seq = [r["latency_seconds"] for r in parallel_rows if r["concurrency"] == 1]
    par = [r["latency_seconds"] for r in parallel_rows if r["concurrency"] == 4]
    speedups = [s / p for s, p in zip(seq, par) if p > 0]
    # Cost scope is explicit: these benchmark executions intentionally avoid the Gemini planner.
    summaries = {}
    for scenario in ("normal", "one_transient_failure"):
        for system in ("direct_api", "orchestrator"):
            subset = [r for r in rows if r["scenario"] == scenario and r["system"] == system]
            summaries[f"{scenario}.{system}"] = _agg(subset)
    normal_direct = summaries["normal.direct_api"]
    normal_orch = summaries["normal.orchestrator"]
    for name, summary in (("normal.direct_api", normal_direct), ("normal.orchestrator", normal_orch)):
        summary["parallel_speedup"] = statistics.mean(speedups) if speedups else None
        summary["parallel_sequential_mean_seconds"] = statistics.mean(seq) if seq else None
        summary["parallel_concurrent_mean_seconds"] = statistics.mean(par) if par else None
        summary["parallel_time_saved_seconds"] = (statistics.mean(seq) - statistics.mean(par)) if seq and par else None
    stress_summaries: dict[str, Any] = {}
    for case in STRESS_CASES:
        for system in ("direct_api", "orchestrator"):
            subset = [r for r in stress_rows if r["case_id"] == case["id"] and r["system"] == system]
            summary = _agg(subset)
            summary["failure_propagation_rate"] = statistics.mean(bool(r.get("failure_propagated", False)) for r in subset) if subset else 0
            summary["mean_error_count"] = statistics.mean(r.get("error_count", 0) for r in subset) if subset else 0
            stress_summaries[f"{case['id']}.{system}"] = summary
    result = {"title": "MCPorch orchestration stress benchmark", "version": 2,
              "methodology": {"systems": "direct_api invokes the same local tool registry directly with the known expected tool; orchestrator plans with the deterministic planner and executes through the DAG validator/scheduler.",
                              "tasks": len(TASKS), "trials": max(1, trials), "normal_task_runs_per_system": len(TASKS) * max(1, trials),
                              "failure_injection": "Separate paired case per task: fail the expected tool exactly once, then let each system retry/recover.",
                              "correctness": "Independent expected fields, expected substrings, or numeric tolerance; completion and correctness are separate.",
                              "model": "Gemini is not called in this benchmark; planner token and model cost are zero. This isolates orchestration overhead and tool execution.",
                              "simulated_api_latency_seconds_per_invocation": SIMULATED_API_LATENCY_SECONDS,
                              "limitations": "Synthetic local-tool benchmark with deterministic injected faults; fixed simulated API delay, not real provider performance or billing. The stress baseline is a simple serial pipeline with no validation or retry, which makes the measured protection benefits specific to that baseline."},
              "stress_cases": [{"id": c["id"], "description": c["description"]} for c in STRESS_CASES],
              "summaries": summaries, "stress_summaries": stress_summaries, "stress_runs": stress_rows,
              "parallel_experiment": {"purpose": "Compare serial and concurrent scheduling on two independent deterministic local tools.",
                                    "trials": max(1, trials), "serial_mean_seconds": statistics.mean(seq) if seq else None,
                                    "concurrent_mean_seconds": statistics.mean(par) if par else None,
                                    "mean_speedup": statistics.mean(speedups) if speedups else None,
                                    "mean_time_saved_seconds": statistics.mean(seq) - statistics.mean(par) if seq and par else None,
                                    "raw_runs": parallel_rows}, "runs": rows}
    return result


def save_results(data: dict[str, Any], folder: str = "data") -> None:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / "benchmark-results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    if data.get("runs"):
        with (path / "benchmark-results.csv").open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data["runs"][0]))
            writer.writeheader()
            writer.writerows(data["runs"])
    (path / "benchmark-report.md").write_text(render_report(data), encoding="utf8")


def render_report(data: dict[str, Any]) -> str:
    summaries = data["summaries"]
    stress_summaries = data.get("stress_summaries", {})
    rows = []
    labels = [
        ("Task completion rate", "task_completion_rate", "percent"),
        ("Correct completion rate", "correct_completion_rate", "percent"),
        ("Recovery success rate", "recovery_success_rate", "percent_or_na"),
        ("End-to-end latency (mean)", "mean_end_to_end_latency_seconds", "seconds"),
        ("End-to-end latency (median)", "median_end_to_end_latency_seconds", "seconds"),
        ("End-to-end latency (p95)", "p95_end_to_end_latency_seconds", "seconds"),
        ("Tool invocations", "tool_invocations", "number"),
        ("Mean tool calls per task", "mean_tool_invocations_per_task", "number"),
        ("Validation failure rate", "validation_failure_rate", "percent_or_na"),
        ("Validation failures / checks", "validation_failures", "fraction"),
        ("Model tokens", "model_tokens", "number"),
        ("Model and tool cost / task", "total_execution_cost_usd", "usd"),
    ]
    for label, key, fmt in labels:
        left, right = summaries["normal.direct_api"].get(key), summaries["normal.orchestrator"].get(key)
        def show(value):
            if value is None: return "N/A"
            if fmt == "percent": return f"{value:.1%}"
            if fmt == "percent_or_na": return f"{value:.1%}" if value is not None else "N/A (no observed events)"
            if fmt == "usd": return f"${value:.6f}"
            if fmt == "seconds": return f"{value:.6f} s"
            if fmt == "fraction": return f"{value} / {summaries['normal.orchestrator']['validation_checks']}" if key == "validation_failures" else str(value)
            return f"{value:.3f}" if isinstance(value, float) else str(value)
        rows.append(f"| {label} | {show(left)} | {show(right)} |")
    parallel = data["parallel_experiment"]
    def nfmt(v): return "N/A" if v is None else f"{v:.6f}"
    stress_table_rows = []
    for case in STRESS_CASES:
        direct = stress_summaries[f"{case['id']}.direct_api"]
        orch = stress_summaries[f"{case['id']}.orchestrator"]
        stress_table_rows.append(
            f"| {case['id']} | {direct['task_completion_rate']:.0%} / {direct['correct_completion_rate']:.0%} | "
            f"{orch['task_completion_rate']:.0%} / {orch['correct_completion_rate']:.0%} | "
            f"{orch['recovered_failures']} recovered; propagation {orch['failure_propagation_rate']:.0%} |"
        )
    return "\n".join([
        "# MCPorch Benchmark Report", "", f"Generated from {data['methodology']['normal_task_runs_per_system']} normal paired tasks per system plus separate failure cases.", "",
        "## Results", "", "| Metric | Direct API baseline | MCPorch orchestrator |", "|---|---:|---:|", *rows, "",
        "## Injected transient-failure recovery", "",
        f"- Direct API: {summaries['one_transient_failure.direct_api']['recovered_failures']} / {summaries['one_transient_failure.direct_api']['recovery_attempts']} recovered ({(summaries['one_transient_failure.direct_api']['recovery_success_rate'] or 0):.1%}).",
        f"- Orchestrator: {summaries['one_transient_failure.orchestrator']['recovered_failures']} / {summaries['one_transient_failure.orchestrator']['recovery_attempts']} recovered ({(summaries['one_transient_failure.orchestrator']['recovery_success_rate'] or 0):.1%}).", "",
        "## Hard orchestration cases", "", "These cases target pipeline problems rather than simple single-tool success. The direct baseline executes the same authored steps in order without validation or retry; MCPorch executes the same dependency structure with its DAG scheduler, output validation, and bounded retry.", "",
        "| Case | Direct API: completed / expected behavior | MCPorch: completed / expected behavior | MCPorch recovery or propagation |", "|---|---:|---:|---:|", *stress_table_rows, "",
        "## Parallel scheduling experiment", "",
        f"Two independent local operations: serial mean {nfmt(parallel['serial_mean_seconds'])} s; concurrency=4 mean {nfmt(parallel['concurrent_mean_seconds'])} s; speedup {nfmt(parallel['mean_speedup'])}x; time saved {nfmt(parallel['mean_time_saved_seconds'])} s.", "",
        "## Interpretation and limits", "", f"This is a small deterministic benchmark with a simulated {SIMULATED_API_LATENCY_SECONDS:.3f}s API round trip per tool invocation. The direct API baseline is intentionally strong: it is given the correct tool and arguments, while MCPorch must plan and validate execution. This does not call a real remote provider and is not a provider SLA or billing measurement. Gemini is not called, so its planning tokens/cost are zero for this benchmark; local tool prices are set to $0. Validation failure rate is N/A if the run produced no invalid outputs.", "",
        "### Methodology", "", "- Completion means the workflow returned successfully. Correctness is checked separately against expected output fields, substrings, or a numeric tolerance.",
        "- Failure recovery is measured both in the ordinary suite's separate transient-failure scenario and in the stress suite's explicit timeout/malformed-response cases.",
        "- End-to-end time is wall-clock time from just before tool/workflow invocation through completion. For API deployment studies, repeat against the actual remote API and record provider latency and billing.",
        "- Raw paired observations are in `benchmark-results.csv`; machine-readable details are in `benchmark-results.json`.", ""])
