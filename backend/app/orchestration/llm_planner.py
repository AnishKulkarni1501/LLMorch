"""Optional Gemini planner with schema-constrained output and usage accounting."""
import asyncio
import json
import os
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from ..models import TaskSpec
from .tools import ToolRegistry
from .planner import extract_document_name


class ProposedStep(BaseModel):
    description: str = Field(min_length=1)
    tool: Literal["local.search", "local.analysis", "local.document", "local.files", "local.text", "local.dataset", "local.units"]
    dependencies: list[int] = Field(description="Zero-based indexes of prerequisite steps")


class ProposedPlan(BaseModel):
    steps: list[ProposedStep] = Field(min_length=1, max_length=12)


PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["steps"],
    "properties": {"steps": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["description", "tool", "dependencies"],
        "properties": {
            "description": {"type": "string"},
            "tool": {"type": "string", "enum": ["local.search", "local.analysis", "local.document", "local.files", "local.text", "local.dataset", "local.units"]},
            "dependencies": {"type": "array", "items": {"type": "integer"}},
        },
    }}}
}


def _tool_input(tool: str, description: str) -> dict[str, Any]:
    """Build safe allowlisted tool arguments from a proposed step."""
    import re
    if tool == "local.search":
        return {"query": description}
    if tool == "local.analysis":
        values = [float(x) for x in re.findall(r"(?<!\w)-?\d+(?:\.\d+)?", description)]
        low = description.lower()
        operation = "mean" if any(k in low for k in ("average", "mean")) else "divide" if "divide" in low else "sum"
        return {"operation": operation, "values": values or [0]}
    if tool == "local.files":
        filename = extract_document_name(description)
        action = "read" if filename and any(k in description.lower() for k in ("read", "open", "view")) else "list"
        return {"action": action, **({"filename": filename} if filename and action == "read" else {})}
    if tool == "local.text":
        import re
        parts = re.split(r"\b(?:text is|text:|this text:)\s*", description, maxsplit=1, flags=re.I)
        return {"text": parts[-1] if len(parts) > 1 else description}
    if tool == "local.dataset":
        import re
        raw_values = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?|(?<!\w)(?:null|none|missing|n/?a)(?!\w)", description, flags=re.I)
        values = [None if value.lower() in {"null", "none", "missing", "na", "n/a"} else float(value) for value in raw_values]
        units = re.findall(r"\b(?:mm|cm|km|m|mi|ft|g|kg|lb|s|min|h|celsius|fahrenheit)\b", description, flags=re.I)
        return {"values": values or [0], "units": units}
    if tool == "local.units":
        import re
        match = re.search(r"(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*([a-z°]+)\s+to\s+([a-z°]+)", description, flags=re.I)
        return {"value": float(match.group(1)) if match else 0, "from_unit": match.group(2) if match else "m", "to_unit": match.group(3) if match else "m"}
    return {"title": description[:80], "findings": [description], "sources": []}


class GeminiPlanner:
    def __init__(self, registry: ToolRegistry, client=None, model: str | None = None) -> None:
        self.registry = registry
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    async def create_plan(self, task: str) -> tuple[list[TaskSpec], dict[str, Any]]:
        tool_inventory = [
            {"name": item["name"], "description": item["description"], "input_schema": item["inputSchema"]}
            for item in self.registry.list() if item["name"].startswith("local.")
        ]
        prompt=("Decompose the user's task into a small executable DAG. Choose only listed tools. "
                "Dependencies are zero-based indexes and may refer only to earlier steps. "
                "Do not invent tool names or perform tool execution.\n\n"+
                json.dumps({"task": task, "tools": tool_inventory}, ensure_ascii=False))
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProposedPlan,
                max_output_tokens=1200,
            ),
        )
        raw = getattr(response, "text", "") or ""
        proposed = ProposedPlan.model_validate_json(raw)
        nodes: list[TaskSpec] = []
        for index, step in enumerate(proposed.steps):
            if step.tool not in self.registry.tools:
                raise ValueError(f"LLM proposed an unauthorized tool: {step.tool}")
            if any(dep < 0 or dep >= index for dep in step.dependencies):
                raise ValueError(f"LLM proposed a missing or forward dependency at step {index}")
            nodes.append(TaskSpec(
                description=step.description,
                selected_tool=step.tool,
                dependencies=[nodes[dep].task_id for dep in step.dependencies],
                input_data=_tool_input(step.tool, step.description),
                estimated_latency=self.registry.tools[step.tool].latency,
            ))
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        reasoning = int(getattr(usage, "thoughts_token_count", 0) or 0)
        usage_data = {
            "provider": "gemini", "model": self.model,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": int(getattr(usage, "total_token_count", input_tokens + output_tokens) or input_tokens + output_tokens),
            "cached_input_tokens": cached, "reasoning_tokens": reasoning,
            "estimated_cost_usd": configured_cost(input_tokens, output_tokens, cached),
            "cost_basis": "configured_rate" if configured_rates() else "unpriced",
        }
        return nodes, usage_data


def configured_rates() -> dict[str, float]:
    """Rates are USD per million tokens and deliberately have no built-in defaults."""
    names = {"input": "GEMINI_INPUT_USD_PER_1M", "output": "GEMINI_OUTPUT_USD_PER_1M", "cached_input": "GEMINI_CACHED_INPUT_USD_PER_1M"}
    rates = {}
    for key, env_name in names.items():
        value = os.getenv(env_name)
        if value is None:
            return {}
        try:
            number = float(value)
        except ValueError:
            return {}
        if number < 0:
            return {}
        rates[key] = number
    return rates


def configured_cost(input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float | None:
    rates = configured_rates()
    if not rates:
        return None
    cached_tokens = min(input_tokens, cached_tokens)
    uncached = input_tokens - cached_tokens
    return (uncached * rates["input"] + cached_tokens * rates["cached_input"] + output_tokens * rates["output"]) / 1_000_000
