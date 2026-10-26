"""DAG validation, scheduling, output validation, and bounded recovery."""
import asyncio, json, time
from datetime import datetime, timezone
from typing import Any, Callable
import networkx as nx
from ..models import Status, Workflow
from .tools import ToolRegistry


def to_graph(workflow: Workflow) -> nx.DiGraph:
    g = nx.DiGraph()
    ids = {n.task_id for n in workflow.nodes}
    for n in workflow.nodes:
        if n.selected_tool not in (None,) and n.selected_tool not in {t["name"] for t in ToolRegistry().list()}:
            raise ValueError(f"unknown tool reference: {n.selected_tool}")
        if len(set(n.dependencies)) != len(n.dependencies): raise ValueError(f"duplicate dependency for {n.task_id}")
        g.add_node(n.task_id, description=n.description, selected_tool=n.selected_tool or "", status=n.status.value,
                   latency=float(n.observed_latency), cost=float(n.estimated_cost), retry_count=int(n.retry_count),
                   validation_status=n.validation_status, error_message=n.error_message or "")
    for n in workflow.nodes:
        missing = set(n.dependencies) - ids
        if missing: raise ValueError(f"missing dependencies: {sorted(missing)}")
        for dep in n.dependencies: g.add_edge(dep, n.task_id)
    if not nx.is_directed_acyclic_graph(g): raise ValueError("workflow graph contains a cycle")
    return g


def validate_output(tool: str, output: Any) -> tuple[bool, str]:
    if not isinstance(output, dict) or not output: return False, "output must be a non-empty object"
    required = {
        "local.search": {"query", "results"}, "local.analysis": {"operation", "value"}, "local.document": {"title", "report", "sources"},
        "local.files": {"documents"} if "documents" in output else {"filename", "content"},
        "local.text": {"word_count", "character_count", "top_terms"},
        "local.dataset": {"count", "valid_count", "missing_count", "units_consistent"},
        "local.units": {"input_value", "from_unit", "value", "to_unit"},
    }
    if not required.get(tool, set()).issubset(output): return False, "required output fields missing"
    if tool == "local.analysis" and (type(output["value"]) not in (int, float) or not __import__("math").isfinite(output["value"])):
        return False, "analysis result must be finite numeric value"
    if tool == "local.search" and not isinstance(output["results"], list): return False, "results must be a list"
    if tool == "local.files":
        if "documents" in output and not isinstance(output["documents"], list): return False, "documents must be a list"
        if "content" in output and not isinstance(output["content"], str): return False, "document content must be text"
    if tool == "local.text" and (type(output["word_count"]) is not int or not isinstance(output["top_terms"], list)): return False, "text analysis fields have invalid types"
    if tool == "local.dataset" and (type(output["count"]) is not int or type(output["missing_count"]) is not int or type(output["units_consistent"]) is not bool): return False, "dataset summary fields have invalid types"
    if tool == "local.units" and (type(output["value"]) not in (int, float) or not __import__("math").isfinite(output["value"])): return False, "converted value must be finite"
    return True, "valid"


async def execute(workflow: Workflow, registry: ToolRegistry, checkpoint: Callable[[Workflow], None] | None = None) -> Any:
    graph = to_graph(workflow)
    lookup = {n.task_id: n for n in workflow.nodes}
    workflow.status = Status.running
    semaphore = asyncio.Semaphore(workflow.max_concurrency)
    # Rehydrate successful steps from the checkpoint. Resume runs only pending or
    # failed work, while successors still wait until their prerequisites complete.
    outputs: dict[str, Any] = {n.task_id: n.output_data for n in workflow.nodes if n.status == Status.completed}
    async def run_node(node):
        async with semaphore:
            tool = node.selected_tool
            if not tool: raise ValueError("no tool selected")
            node.status = Status.running
            node.input_data.setdefault("dependencies", {d: outputs[d] for d in node.dependencies})
            if tool == "local.document" and node.dependencies:
                verified = [outputs[d] for d in node.dependencies if d in outputs]
                facts, sources = [], []
                for value in verified:
                    if isinstance(value, dict) and isinstance(value.get("results"), list):
                        for item in value["results"]:
                            if isinstance(item, dict):
                                facts.append(item.get("text", json.dumps(item, sort_keys=True)))
                                if item.get("title"): sources.append(item["title"])
                    else:
                        facts.append(json.dumps(value, sort_keys=True, default=str))
                if facts:
                    node.input_data["findings"] = list(dict.fromkeys(node.input_data.get("findings", []) + facts))
                if sources:
                    node.input_data["sources"] = list(dict.fromkeys(node.input_data.get("sources", []) + sources))
            started = time.perf_counter()
            for attempt in range(workflow.max_retries + 1):
                try:
                    result = await registry.invoke(tool, node.input_data)
                    ok, reason = validate_output(tool, result)
                    node.validation_status = "valid" if ok else "invalid"
                    if not ok: raise ValueError(reason)
                    node.observed_latency = time.perf_counter() - started
                    node.output_data, node.status = result, Status.completed
                    outputs[node.task_id] = result
                    workflow.events.append({"time":datetime.now(timezone.utc).isoformat(),"task_id":node.task_id,"type":"completed","tool":tool})
                    if checkpoint:
                        checkpoint(workflow)
                    return
                except Exception as e:
                    node.error_message = str(e)
                    node.retry_count = attempt + 1 if attempt < workflow.max_retries else node.retry_count
                    node.recovery_history.append(f"{type(e).__name__}: {e}")
                    if attempt < workflow.max_retries:
                        node.status = Status.recovering
                        workflow.events.append({"time":datetime.now(timezone.utc).isoformat(),"task_id":node.task_id,"type":"retry","attempt":attempt+1,"error":str(e)})
                        await asyncio.sleep(.05 * (2 ** attempt))
                    else:
                        node.status = Status.failed
                        node.observed_latency = time.perf_counter() - started
                        raise
    pending = {key for key in graph.nodes if lookup[key].status != Status.completed}
    while pending:
        ready = [key for key in pending if all(lookup[d].status == Status.completed for d in lookup[key].dependencies)]
        if not ready:
            for key in pending:
                if any(lookup[d].status == Status.failed for d in lookup[key].dependencies):
                    lookup[key].status = Status.failed
                    lookup[key].error_message = "prerequisite failed"
            break
        results = await asyncio.gather(*(run_node(lookup[k]) for k in ready), return_exceptions=True)
        pending.difference_update(ready)
        if any(isinstance(r, BaseException) for r in results):
            for r in results:
                if isinstance(r, BaseException):
                    workflow.events.append({"time":datetime.now(timezone.utc).isoformat(),"type":"failed","error":str(r)})
            # Independent ready nodes may finish; dependent work is not released if a prerequisite failed.
    workflow.status = Status.completed if all(n.status == Status.completed for n in workflow.nodes) else Status.failed
    if checkpoint:
        checkpoint(workflow)
    return workflow.nodes[-1].output_data if workflow.nodes else None


