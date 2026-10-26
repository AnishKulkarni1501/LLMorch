"""Deterministic allowlisted tool registry (local adapter for the MCP tool contract)."""
import asyncio
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    reliability: float = .98
    latency: float = .02
    cost: float = 0.0


def _search(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    docs = [
        {"title": "Graph scheduling", "text": "DAG schedulers run independent tasks concurrently and wait for prerequisites."},
        {"title": "MCP tools", "text": "Model Context Protocol servers expose discoverable tools with schemas."},
        {"title": "Failure recovery", "text": "Bounded retries, validation, and alternate tools improve workflow resilience."},
    ]
    terms = set(re.findall(r"\w+", query.lower()))
    ranked = sorted(docs, key=lambda d: len(terms & set(re.findall(r"\w+", (d["title"] + " " + d["text"]).lower()))), reverse=True)
    return {"query": query, "results": ranked[:2]}


def _calculate(args: dict[str, Any]) -> dict[str, Any]:
    op = args.get("operation", "sum")
    values = args.get("values")
    if not isinstance(values, list) or not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("values must be a non-empty list of finite numbers")
    if op == "sum": result = sum(values)
    elif op == "mean": result = sum(values) / len(values)
    elif op == "min": result = min(values)
    elif op == "max": result = max(values)
    elif op == "divide":
        if len(values) != 2: raise ValueError("divide expects exactly two values")
        if values[1] == 0: raise ValueError("division by zero")
        result = values[0] / values[1]
    else: raise ValueError(f"unsupported operation: {op}")
    return {"operation": op, "value": result}


def _report(args: dict[str, Any]) -> dict[str, Any]:
    title = str(args.get("title", "Research summary")).strip()
    sources = args.get("sources", [])
    findings = args.get("findings", [])
    if not isinstance(sources, list) or not isinstance(findings, list):
        raise ValueError("sources and findings must be lists")
    body = "\n".join(f"- {item}" for item in findings) or "No verified findings provided."
    return {"title": title, "report": f"# {title}\n\n{body}", "sources": sources}


def _documents(args: dict[str, Any]) -> dict[str, Any]:
    """List or read small text documents from one dedicated user-data folder."""
    root = Path(__file__).resolve().parents[2] / "data" / "user_documents"
    root.mkdir(parents=True, exist_ok=True)
    action = args.get("action", "list")
    allowed = {".txt", ".md", ".csv", ".json"}
    if action == "list":
        files = []
        for path in root.iterdir():
            if path.suffix.lower() in allowed and path.is_file() and path.resolve().parent == root.resolve():
                files.append({"filename": path.name, "size_bytes": path.stat().st_size})
        return {"documents": sorted(files, key=lambda item: item["filename"].lower())}
    if action != "read":
        raise ValueError("action must be 'list' or 'read'")
    filename = args.get("filename", "")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("filename must be a simple file name in the user documents folder")
    path = root / filename
    if path.suffix.lower() not in allowed or not path.is_file() or path.resolve().parent != root.resolve():
        raise ValueError("document not found or file type is not supported")
    if path.stat().st_size > 100_000:
        raise ValueError("document exceeds the 100 KB read limit")
    return {"filename": filename, "content": path.read_text(encoding="utf-8"), "size_bytes": path.stat().st_size}


def _analyze_text(args: dict[str, Any]) -> dict[str, Any]:
    text = args.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    words = re.findall(r"[\w'-]+", text.lower())
    stopwords = {"the", "and", "for", "are", "but", "not", "with", "this", "that", "from", "you", "your", "was", "were", "have", "has", "into", "about", "then"}
    counts: dict[str, int] = {}
    for word in words:
        if len(word) > 2 and word not in stopwords:
            counts[word] = counts.get(word, 0) + 1
    top_terms = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    return {"word_count": len(words), "character_count": len(text), "estimated_reading_minutes": round(len(words) / 200, 2), "top_terms": [{"term": word, "count": count} for word, count in top_terms]}


def _summarize_dataset(args: dict[str, Any]) -> dict[str, Any]:
    values = args.get("values")
    units = args.get("units", [])
    if not isinstance(values, list) or not values:
        raise ValueError("values must be a non-empty list of numbers or nulls")
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in values):
        raise ValueError("values must contain only finite numbers or nulls")
    if not isinstance(units, list) or any(not isinstance(unit, str) for unit in units):
        raise ValueError("units must be a list of unit labels")
    numeric = [float(value) for value in values if value is not None]
    unique_units = sorted(set(units))
    result: dict[str, Any] = {"count": len(values), "valid_count": len(numeric), "missing_count": len(values) - len(numeric), "units": unique_units, "units_consistent": len(unique_units) <= 1}
    if numeric:
        result.update({"sum": sum(numeric), "mean": sum(numeric) / len(numeric), "minimum": min(numeric), "maximum": max(numeric)})
    else:
        result["statistics_error"] = "no numeric values available"
    return result


def _convert_units(args: dict[str, Any]) -> dict[str, Any]:
    value = args.get("value")
    source = str(args.get("from_unit", "")).lower().replace("°", "").strip()
    target = str(args.get("to_unit", "")).lower().replace("°", "").strip()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("value must be a finite number")
    aliases = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km", "centimeter": "cm", "centimeters": "cm", "mile": "mi", "miles": "mi", "foot": "ft", "feet": "ft", "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "pound": "lb", "pounds": "lb", "celsius": "c", "fahrenheit": "f", "kelvin": "k", "seconds": "s", "second": "s", "minutes": "min", "minute": "min", "hours": "h", "hour": "h"}
    source, target = aliases.get(source, source), aliases.get(target, target)
    factors = {"m": 1.0, "cm": .01, "km": 1000.0, "mi": 1609.344, "ft": .3048, "g": .001, "kg": 1.0, "lb": .45359237, "s": 1.0, "min": 60.0, "h": 3600.0}
    if source in factors and target in factors and ((source in {"m", "cm", "km", "mi", "ft"}) == (target in {"m", "cm", "km", "mi", "ft"})) and ((source in {"g", "kg", "lb"}) == (target in {"g", "kg", "lb"})) and ((source in {"s", "min", "h"}) == (target in {"s", "min", "h"})):
        converted = value * factors[source] / factors[target]
    elif source in {"c", "f", "k"} and target in {"c", "f", "k"}:
        celsius = value if source == "c" else (value - 32) * 5 / 9 if source == "f" else value - 273.15
        converted = celsius if target == "c" else celsius * 9 / 5 + 32 if target == "f" else celsius + 273.15
    else:
        raise ValueError(f"unsupported or incompatible unit conversion: {source} to {target}")
    return {"input_value": value, "from_unit": source, "value": converted, "to_unit": target}


class ToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "local.search": Tool("local.search", "Search local research documents", {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}, _search),
            "local.analysis": Tool("local.analysis", "Perform allowlisted numeric analysis", {"type":"object","properties":{"operation":{"type":"string"},"values":{"type":"array","items":{"type":"number"}}},"required":["values"]}, _calculate),
            "local.document": Tool("local.document", "Generate a report from verified inputs", {"type":"object","properties":{"title":{"type":"string"},"findings":{"type":"array"},"sources":{"type":"array"}}}, _report),
            "local.files": Tool("local.files", "List or read small text files from backend/data/user_documents", {"type":"object","properties":{"action":{"type":"string","enum":["list","read"]},"filename":{"type":"string"}}}, _documents),
            "local.text": Tool("local.text", "Count words and find frequent terms in text", {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}, _analyze_text),
            "local.dataset": Tool("local.dataset", "Summarize numeric values, missing entries, and unit consistency", {"type":"object","properties":{"values":{"type":"array","items":{"type":["number","null"]}},"units":{"type":"array","items":{"type":"string"}}},"required":["values"]}, _summarize_dataset),
            "local.units": Tool("local.units", "Convert common length, mass, time, and temperature units", {"type":"object","properties":{"value":{"type":"number"},"from_unit":{"type":"string"},"to_unit":{"type":"string"}},"required":["value","from_unit","to_unit"]}, _convert_units),
        }
        self.fail_once: set[str] = set()
        self.calls = 0
        self.mcp_manager = None
        self.external_tools: list[dict[str, Any]] = []

    def configure_mcp(self, manager, tools: list[dict[str, Any]]) -> None:
        """Use discovered official MCP tools when matching local capabilities exist."""
        self.mcp_manager, self.external_tools = manager, tools

    def list(self) -> list[dict[str, Any]]:
        local=[{"name": t.name, "description": t.description, "inputSchema": t.input_schema, "reliability": t.reliability, "estimated_latency": t.latency, "estimated_cost": t.cost, "transport":"local"} for t in self.tools.values()]
        remote=[{**t,"transport":"mcp"} for t in self.external_tools]
        return local+remote

    async def invoke(self, name: str, args: dict[str, Any], timeout: float = 10) -> Any:
        if self.mcp_manager and name in self.tools:
            suffix={"local.search":"search_documents","local.analysis":"analyze","local.document":"create_report","local.text":"analyze_text","local.dataset":"summarize_dataset","local.units":"convert_units"}.get(name)
            if name == "local.files":
                suffix = "read_document" if args.get("action") == "read" else "list_documents"
            match=next((t["name"] for t in self.external_tools if t["name"].endswith("."+suffix)),None)
            if match:
                self.calls += 1
                remote_args = {} if suffix == "list_documents" else args
                remote_args = {key: value for key, value in remote_args.items() if key != "dependencies"}
                response=await asyncio.wait_for(self.mcp_manager.call_tool(match,remote_args),timeout=timeout)
                if getattr(response,"structuredContent",None) is not None: return response.structuredContent
                for block in getattr(response,"content",[]):
                    if getattr(block,"text",None):
                        import json
                        try: return json.loads(block.text)
                        except json.JSONDecodeError: pass
                raise ValueError(f"MCP tool {match} returned no structured output")
        if name not in self.tools: raise ValueError(f"unknown tool: {name}")
        self.calls += 1
        if name in self.fail_once:
            self.fail_once.remove(name)
            raise TimeoutError(f"injected timeout from {name}")
        tool = self.tools[name]
        start = time.perf_counter()
        result = await asyncio.wait_for(asyncio.to_thread(tool.handler, args), timeout=timeout)
        return result

