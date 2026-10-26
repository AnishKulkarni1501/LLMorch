"""FastAPI application for adaptive MCP-style workflow orchestration."""
import json
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import Workflow, WorkflowCreate, RunResponse, Status
from .orchestration.tools import ToolRegistry
from .orchestration.planner import plan
from .orchestration.engine import execute, to_graph
from .benchmark.runner import TASKS, run_benchmarks, save_results
from .mcp.client_manager import MCPClientManager
from .orchestration.llm_planner import GeminiPlanner
from .workflow_store import load_workflows, save_workflow

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

@asynccontextmanager
async def lifespan(app):
    workflows.update(load_workflows())
    manager=MCPClientManager()
    if os.getenv("MCP_SERVERS"):
        try:
            await manager.connect_configured()
            registry.configure_mcp(manager,await manager.list_tools())
        except Exception:
            await manager.close()
            raise
    yield
    await manager.close()

app=FastAPI(title="Adaptive MCP Orchestration Engine",version="0.1.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
registry=ToolRegistry(); workflows:dict[str,Workflow]={}; benchmark_results:dict={"runs":[],"strategies":{}}

@app.get("/health")
async def health(): return {"status":"ok","planner":"gemini" if os.getenv("GEMINI_API_KEY") else "deterministic","tool_transport":"mcp" if registry.external_tools else "local adapters"}
@app.get("/tools")
async def tools(): return registry.list()
@app.post("/workflows",response_model=Workflow)
async def create_workflow(request:WorkflowCreate):
    try:
        nodes=plan(request.task,registry)
        planner_mode="deterministic"
        llm_usage={}
        if os.getenv("GEMINI_API_KEY") and os.getenv("LLM_PROVIDER","gemini").lower()=="gemini":
            try:
                nodes,llm_usage=await GeminiPlanner(registry).create_plan(request.task)
                planner_mode="gemini"
            except Exception as exc:
                planner_mode="deterministic_fallback"
                llm_usage={"provider":"gemini","model":os.getenv("GEMINI_MODEL","gemini-2.5-flash"),"input_tokens":0,"output_tokens":0,"total_tokens":0,"estimated_cost_usd":None,"cost_basis":"unavailable","error":str(exc)}
        wf=Workflow(task=request.task,nodes=nodes,max_concurrency=request.max_concurrency,max_retries=request.max_retries,planner_mode=planner_mode,llm_usage=llm_usage)
        if llm_usage:
            wf.events.append({"type":"planner_usage",**llm_usage})
        to_graph(wf); workflows[wf.workflow_id]=wf; save_workflow(wf); return wf
    except ValueError as exc: raise HTTPException(422,str(exc))
@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id:str):
    if workflow_id not in workflows: raise HTTPException(404,"workflow not found")
    return workflows[workflow_id]
@app.get("/workflows/{workflow_id}/graph")
async def get_graph(workflow_id:str):
    if workflow_id not in workflows: raise HTTPException(404,"workflow not found")
    wf=workflows[workflow_id]; g=to_graph(wf)
    return {"nodes":[{"id":k,**dict(v)} for k,v in g.nodes(data=True)],"edges":[{"source":a,"target":b} for a,b in g.edges]}
@app.post("/workflows/{workflow_id}/run",response_model=RunResponse)
async def run_workflow(workflow_id:str):
    if workflow_id not in workflows: raise HTTPException(404,"workflow not found")
    wf=workflows[workflow_id]
    result=await execute(wf,registry,checkpoint=save_workflow)
    return RunResponse(workflow_id=workflow_id,status=wf.status,result=result,events=wf.events)
@app.post("/workflows/{workflow_id}/resume",response_model=RunResponse)
async def resume_workflow(workflow_id:str):
    if workflow_id not in workflows: raise HTTPException(404,"workflow not found")
    wf=workflows[workflow_id]
    if wf.status == Status.running: raise HTTPException(409,"workflow is already running")
    failed=[n for n in wf.nodes if n.status == Status.failed]
    if not failed: raise HTTPException(409,"workflow has no failed steps to resume")
    for node in failed:
        node.status=Status.pending
        node.error_message=None
        node.validation_status="not_checked"
    wf.status=Status.pending
    wf.events.append({"type":"resume_requested","failed_steps_requeued":[node.task_id for node in failed]})
    save_workflow(wf)
    result=await execute(wf,registry,checkpoint=save_workflow)
    return RunResponse(workflow_id=workflow_id,status=wf.status,result=result,events=wf.events)
@app.get("/workflows/{workflow_id}/metrics")
async def metrics(workflow_id:str):
    if workflow_id not in workflows: raise HTTPException(404,"workflow not found")
    wf=workflows[workflow_id]
    return {"completion":wf.status.value=="completed","tool_calls":sum(1+n.retry_count for n in wf.nodes),"retries":sum(n.retry_count for n in wf.nodes),"latency_seconds":sum(n.observed_latency for n in wf.nodes),"planner_mode":wf.planner_mode,"llm_usage":wf.llm_usage,"model_tokens":wf.llm_usage.get("total_tokens",0),"estimated_model_cost_usd":wf.llm_usage.get("estimated_cost_usd")}
@app.get("/benchmarks")
async def benchmarks(): return {"tasks":len(TASKS),"stress_cases":4,"systems":["direct_api","orchestrator"],"scenarios":["normal","transient_failure","fanout_fanin","invalid_output","blocked_dependency"]}
@app.post("/benchmarks/run")
async def benchmark_run(trials:int=1):
    global benchmark_results
    benchmark_results=await run_benchmarks(max(1,min(trials,20))); save_results(benchmark_results)
    return benchmark_results
@app.get("/benchmarks/results")
async def benchmark_get(): return benchmark_results
