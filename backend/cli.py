"""Run MCPorch workflows from a terminal without starting the dashboard or API."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.models import Status, Workflow
from app.mcp.client_manager import MCPClientManager
from app.orchestration.engine import execute, to_graph
from app.orchestration.llm_planner import GeminiPlanner
from app.orchestration.planner import plan as deterministic_plan
from app.orchestration.tools import ToolRegistry


load_dotenv(Path(__file__).resolve().parent / ".env")


async def run_task(task: str, *, use_llm: bool = True, max_concurrency: int = 4,
                   max_retries: int = 1, as_json: bool = False) -> int:
    registry = ToolRegistry()
    manager = MCPClientManager()
    try:
        if os.getenv("MCP_SERVERS"):
            await manager.connect_configured()
            registry.configure_mcp(manager, await manager.list_tools())

        nodes = deterministic_plan(task, registry)
        planner_mode = "deterministic"
        usage: dict[str, Any] = {}
        if use_llm and os.getenv("GEMINI_API_KEY") and os.getenv("LLM_PROVIDER", "gemini").lower() == "gemini":
            try:
                nodes, usage = await GeminiPlanner(registry).create_plan(task)
                planner_mode = "gemini"
            except Exception as exc:
                planner_mode = "deterministic_fallback"
                if not as_json:
                    print(f"Gemini planning failed ({type(exc).__name__}); using the local fallback planner.\n")

        workflow = Workflow(task=task, nodes=nodes, max_concurrency=max_concurrency,
                            max_retries=max_retries, planner_mode=planner_mode, llm_usage=usage)
        to_graph(workflow)  # Reject bad plans before execution.

        if not as_json:
            print(f"\nPlanner: {planner_mode}")
            print("Plan:")
            step_number = {node.task_id: index + 1 for index, node in enumerate(workflow.nodes)}
            for index, node in enumerate(workflow.nodes, 1):
                waits_for = f" (after step {', '.join(str(step_number[d]) for d in node.dependencies)})" if node.dependencies else ""
                print(f"  {index}. {node.description}  [{node.selected_tool}]{waits_for}")
            print("\nRunning workflow...\n")

        result = await execute(workflow, registry)
        payload = {
            "planner": planner_mode,
            "status": workflow.status.value,
            "steps": [{"description": n.description, "tool": n.selected_tool,
                       "status": n.status.value, "retries": n.retry_count,
                       "validation": n.validation_status, "result": n.output_data,
                       "error": n.error_message} for n in workflow.nodes],
            "result": result,
            "llm_usage": usage,
        }
        if as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        else:
            print("Steps:")
            for index, node in enumerate(workflow.nodes, 1):
                print(f"  {index}. {node.status.value.upper()} — {node.selected_tool} ({node.description})")
                if node.error_message:
                    print(f"     Error: {node.error_message}")
            print("\nResult:")
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            if usage:
                cost = usage.get("estimated_cost_usd")
                cost_text = f"${cost:.6f} estimated" if cost is not None else "cost unpriced"
                print(f"\nGemini usage: {usage.get('input_tokens', 0)} input + "
                      f"{usage.get('output_tokens', 0)} output = {usage.get('total_tokens', 0)} total tokens; {cost_text}.")
            elif planner_mode == "deterministic":
                print("\nGemini was not called (no key configured or --no-llm).")
        return 0 if workflow.status == Status.completed else 1
    finally:
        await manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an MCPorch task directly in your terminal.")
    parser.add_argument("task", nargs="*", help="Task text. Omit it to enter interactive mode.")
    parser.add_argument("--no-llm", action="store_true", help="Use the deterministic planner even when Gemini is configured.")
    parser.add_argument("--json", action="store_true", help="Print the run as JSON for scripts and piping.")
    parser.add_argument("--max-concurrency", type=int, default=4, choices=range(1, 33), metavar="1..32")
    parser.add_argument("--max-retries", type=int, default=1, choices=range(0, 6), metavar="0..5")
    parser.add_argument("--list-tools", action="store_true", help="List local tools and exit.")
    return parser.parse_args()


async def list_tools() -> None:
    registry = ToolRegistry()
    manager = MCPClientManager()
    try:
        if os.getenv("MCP_SERVERS"):
            await manager.connect_configured()
            registry.configure_mcp(manager, await manager.list_tools())
        print(json.dumps(registry.list(), indent=2, ensure_ascii=False))
    finally:
        await manager.close()


def main() -> None:
    args = parse_args()
    if args.list_tools:
        asyncio.run(list_tools())
        return
    if args.task:
        raise SystemExit(asyncio.run(run_task(" ".join(args.task), use_llm=not args.no_llm,
                                             max_concurrency=args.max_concurrency,
                                             max_retries=args.max_retries, as_json=args.json)))
    if args.json:
        raise SystemExit("--json needs a task argument")
    print("MCPorch terminal mode. Type a task, or 'exit' to quit.")
    while True:
        try:
            task = input("\nTask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if task.lower() in {"exit", "quit"}:
            break
        if not task:
            continue
        asyncio.run(run_task(task, use_llm=not args.no_llm,
                             max_concurrency=args.max_concurrency,
                             max_retries=args.max_retries))


if __name__ == "__main__":
    main()
