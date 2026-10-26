"""Transparent deterministic decomposition and weighted tool selection."""
import re
from typing import Any
from .tools import ToolRegistry
from ..models import TaskSpec


def extract_document_name(text: str) -> str | None:
    """Extract a supported document name without swallowing instruction words."""
    match = re.search(r"\b(?:file|document)\s+['\"]?(.+?\.(?:txt|md|csv|json))(?=['\"]|\b|$)", text, flags=re.I)
    if not match:
        match = re.search(r"([\w-][\w.-]*\.(?:txt|md|csv|json))\b", text, flags=re.I)
    return match.group(1).strip().strip("'\"") if match else None


def select_tool(description: str, registry: ToolRegistry) -> str:
    text = description.lower()
    keywords = {
        "local.search": {"search", "find", "research", "look up", "sources", "information"},
        "local.analysis": {"calculate", "sum", "average", "mean", "divide", "analyze", "number", "data"},
        "local.document": {"report", "document", "summarize", "write", "draft"},
        "local.files": {"list files", "list documents", "read file", "open file", "local file", ".txt", ".md", ".csv", ".json"},
        "local.text": {"word count", "count words", "analyze text", "text analysis", "frequent words", "frequent terms", "keywords in this text"},
        "local.dataset": {"dataset", "data set", "dataset values", "summarize dataset", "analyze dataset", "missing values", "missing data", "unit consistency", "numeric dataset"},
        "local.units": {"convert", "conversion", "kilometers", "kilometres", "miles", "celsius", "fahrenheit", "pounds", "kilograms"},
    }
    # Relevance dominates; reliability/cost/latency priors break ties transparently.
    def matched(keyword: str) -> bool:
        if re.fullmatch(r"[a-z0-9]+", keyword):
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text
    scores = {name: sum(1 for k in keys if matched(k)) + .1 * t.reliability - .001 * t.latency - t.cost
              for name, t in registry.tools.items() for keys in [keywords[name]]}
    return max(scores, key=scores.get)


def plan(task: str, registry: ToolRegistry) -> list[TaskSpec]:
    text = task.strip()
    segments = [s.strip(" ,;.") for s in re.split(r"\bthen\b|\band then\b|\bafter that\b", text, flags=re.I) if s.strip(" ,;.")]
    # A search followed by a report is represented explicitly so dependency edges control execution.
    research_intent = bool(re.search(r"\b(research|find|search)\b", text, flags=re.I))
    report_intent = bool(re.search(r"\b(report|summari[sz]e|write|draft)\b|\bcreate (?:a )?document\b", text, flags=re.I))
    if len(segments) == 1 and research_intent and report_intent:
        segments = [f"Search for information about {text}", f"Write a report using the verified search results for {text}"]
    # Sequential wording alone does not imply a data dependency. Independent
    # actions can run together; dependency language means later steps consume
    # earlier results and must wait.
    sequential = bool(re.search(r"\b(after that|based on (?:the )?(?:result|results|findings)|using (?:the )?(?:result|results|findings)|from (?:the )?(?:result|results|findings))\b", text, flags=re.I))
    # Conventional research-then-report tasks need the search output as report input.
    sequential = sequential or (len(segments) > 1 and any(k in text.lower() for k in ("research", "search", "find")) and any(k in text.lower() for k in ("report", "summarize", "document")))
    nodes: list[TaskSpec] = []
    for i, segment in enumerate(segments):
        tool = select_tool(segment, registry)
        args: dict[str, Any] = {}
        if tool == "local.search": args = {"query": segment}
        elif tool == "local.analysis":
            nums = [float(x) for x in re.findall(r"(?<!\w)-?\d+(?:\.\d+)?", segment)]
            op = "mean" if any(k in segment.lower() for k in ("average", "mean")) else "divide" if "divide" in segment.lower() else "sum"
            args = {"operation": op, "values": nums or [0]}
        elif tool == "local.files":
            filename = extract_document_name(segment)
            args = {"action": "read" if any(k in segment.lower() for k in ("read", "open", "view")) and filename else "list"}
            if filename and args["action"] == "read": args["filename"] = filename
        elif tool == "local.text":
            value = re.split(r"\b(?:text is|text:|this text:)\s*", segment, maxsplit=1, flags=re.I)
            args = {"text": value[-1] if len(value) > 1 else segment}
        elif tool == "local.dataset":
            raw_values = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?|(?<!\w)(?:null|none|missing|n/?a)(?!\w)", segment, flags=re.I)
            values = [None if value.lower() in {"null", "none", "missing", "na", "n/a"} else float(value) for value in raw_values]
            units = re.findall(r"\b(?:mm|cm|km|m|mi|ft|g|kg|lb|s|min|h|celsius|fahrenheit)\b", segment, flags=re.I)
            args = {"values": values or [0], "units": units}
        elif tool == "local.units":
            match = re.search(r"(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*([a-z°]+)\s+to\s+([a-z°]+)", segment, flags=re.I)
            args = {"value": float(match.group(1)) if match else 0, "from_unit": match.group(2) if match else "m", "to_unit": match.group(3) if match else "m"}
        else: args = {"title": segment[:80], "findings": [segment], "sources": []}
        dependencies = [nodes[-1].task_id] if i and sequential else []
        nodes.append(TaskSpec(description=segment, selected_tool=tool, dependencies=dependencies, input_data=args, estimated_latency=registry.tools[tool].latency))
    return nodes
