import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.orchestration.llm_planner import GeminiPlanner, configured_cost
from app.orchestration.tools import ToolRegistry

class FakeModels:
    def __init__(self, response): self.response=response; self.request=None
    def generate_content(self, **kwargs): self.request=kwargs; return self.response

class FakeClient:
    def __init__(self, response): self.models=FakeModels(response)

def test_llm_plan_uses_model_choices_and_reports_api_usage():
    response=SimpleNamespace(text=json.dumps({"steps":[
        {"description":"Search MCP references","tool":"local.search","dependencies":[]},
        {"description":"Write report","tool":"local.document","dependencies":[0]},
    ]}),usage_metadata=SimpleNamespace(prompt_token_count=120,candidates_token_count=35,total_token_count=155,
        cached_content_token_count=20,thoughts_token_count=4))
    client=FakeClient(response)
    nodes,usage=asyncio.run(GeminiPlanner(ToolRegistry(),client=client,model="test-model").create_plan("Research MCP and report"))
    assert nodes[1].dependencies==[nodes[0].task_id]
    assert usage["total_tokens"]==155 and usage["cached_input_tokens"]==20 and usage["reasoning_tokens"]==4
    assert client.models.request["config"].response_mime_type=="application/json"
    assert client.models.request["config"].response_schema is not None

def test_llm_rejects_forward_dependencies():
    response=SimpleNamespace(text=json.dumps({"steps":[
        {"description":"Search","tool":"local.search","dependencies":[1]},
    ]}),usage_metadata=None)
    with pytest.raises(ValueError,match="forward dependency"):
        asyncio.run(GeminiPlanner(ToolRegistry(),client=FakeClient(response)).create_plan("Search"))

def test_cost_is_only_calculated_from_configured_rates(monkeypatch):
    for key in ("GEMINI_INPUT_USD_PER_1M","GEMINI_OUTPUT_USD_PER_1M","GEMINI_CACHED_INPUT_USD_PER_1M"):
        monkeypatch.delenv(key,raising=False)
    assert configured_cost(1000,500,200) is None
    monkeypatch.setenv("GEMINI_INPUT_USD_PER_1M","2")
    monkeypatch.setenv("GEMINI_OUTPUT_USD_PER_1M","8")
    monkeypatch.setenv("GEMINI_CACHED_INPUT_USD_PER_1M","0.5")
    assert configured_cost(1000,500,200)==pytest.approx(.0057)
