import sys
import asyncio
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from app.models import Workflow,TaskSpec,Status
from app.orchestration.engine import to_graph,execute,validate_output
from app.orchestration.tools import ToolRegistry
from app.orchestration.planner import plan
from app import workflow_store

def test_cycle_rejected():
    a=TaskSpec(task_id="a",description="a",selected_tool="local.search",dependencies=["b"])
    b=TaskSpec(task_id="b",description="b",selected_tool="local.search",dependencies=["a"])
    with pytest.raises(ValueError,match="cycle"):to_graph(Workflow(task="x",nodes=[a,b]))

def test_missing_dependency_rejected():
    with pytest.raises(ValueError,match="missing dependencies"):
        to_graph(Workflow(task="x",nodes=[TaskSpec(task_id="a",description="a",selected_tool="local.search",dependencies=["gone"])]))

def test_output_validation():
    assert validate_output("local.analysis",{"operation":"sum","value":3})[0]
    assert not validate_output("local.analysis",{"operation":"divide","value":float("inf")})[0]

def test_execution_respects_dependency_topology():
    reg=ToolRegistry(); wf=Workflow(task="Research MCP and write a report",nodes=plan("Research MCP then write a report",reg))
    asyncio.run(execute(wf,reg))
    assert wf.status==Status.completed
    graph=to_graph(wf)
    assert graph.number_of_nodes()==len(wf.nodes) and graph.number_of_edges()==1
    assert all(node.status==Status.completed for node in wf.nodes)

def test_bounded_retry_recovers_timeout():
    reg=ToolRegistry(); reg.fail_once.add("local.search")
    wf=Workflow(task="Search MCP",nodes=plan("Search MCP",reg),max_retries=1)
    asyncio.run(execute(wf,reg))
    assert wf.status==Status.completed and wf.nodes[0].retry_count==1 and reg.calls==2

def test_resume_reuses_completed_checkpoint_and_retries_failed_dependency():
    reg=ToolRegistry(); reg.fail_once.add("local.search")
    nodes=[
        TaskSpec(task_id="independent",description="sum",selected_tool="local.analysis",input_data={"operation":"sum","values":[4,5]}),
        TaskSpec(task_id="search",description="search",selected_tool="local.search",input_data={"query":"MCP"}),
        TaskSpec(task_id="report",description="report",selected_tool="local.document",dependencies=["search"],input_data={"title":"Results","findings":[],"sources":[]}),
    ]
    wf=Workflow(task="Checkpoint test",nodes=nodes,max_concurrency=3,max_retries=0)
    checkpoints=[]
    asyncio.run(execute(wf,reg,checkpoint=lambda current:checkpoints.append(current.model_dump_json())))
    assert wf.status==Status.failed
    assert wf.nodes[0].status==Status.completed and wf.nodes[2].status==Status.failed
    calls_after_failure=reg.calls
    for node in wf.nodes:
        if node.status==Status.failed: node.status=Status.pending; node.error_message=None
    wf.status=Status.pending
    asyncio.run(execute(wf,reg))
    assert wf.status==Status.completed
    assert reg.calls-calls_after_failure==2 # search and report only; completed sum reused
    assert wf.nodes[0].output_data["value"]==9
    assert checkpoints

def test_workflow_store_round_trip(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow_store,"store_dir",lambda:tmp_path)
    wf=Workflow(task="Persist outputs",nodes=[TaskSpec(task_id="saved",description="saved",selected_tool="local.analysis",status=Status.completed,output_data={"value":9})])
    workflow_store.save_workflow(wf)
    restored=workflow_store.load_workflows()[wf.workflow_id]
    assert restored.nodes[0].status==Status.completed
    assert restored.nodes[0].output_data=={"value":9}
