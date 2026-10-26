# MCPorch: Adaptive MCP Orchestration Prototype

Local research prototype for dependency-aware orchestration, LLM-proposed plans, deterministic tool authorization, output validation, bounded retry, and strategy comparisons. With `GEMINI_API_KEY` configured, the backend uses the Gemini API with JSON structured output to propose DAG steps and tool assignments. Without it, the deterministic planner remains available. Built-in execution uses local allowlisted adapters so the app can still run without a paid tool service. Official MCP SDK stdio client and three runnable MCP servers are also included; configure `MCP_SERVERS` for external servers.

## Run locally

Requires Python 3.11+ and Node 18+.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. API docs are at http://127.0.0.1:8000/docs.

## Use it from PowerShell

You do not need to start the API or open the dashboard for terminal use. From the `backend` folder, activate the virtual environment once and run:

```powershell
cd "C:\Users\Anish Kulkarni\OneDrive\Documents\Projects\MCPorch\backend"
.\.venv\Scripts\Activate.ps1
python cli.py "Research graph scheduling and write a report"
```

The CLI prints the plan, step statuses, result, and Gemini token counts. For an interactive prompt, run `python cli.py` and type tasks until you enter `exit`. Use `python cli.py --no-llm "Calculate the sum of 4 and 5"` to skip Gemini, `python cli.py --list-tools` to see tools, or add `--json` after a task for JSON output. To use the file tool, copy small `.txt`, `.md`, `.csv`, or `.json` documents into `backend/data/user_documents/`, then ask to list the documents or read a specific filename. The tool can access only files directly inside that folder.

To enable model planning, copy `backend/.env.example` to `backend/.env`, set `GEMINI_API_KEY`, and restart the backend. The default model is `gemini-2.5-flash`; set `GEMINI_MODEL` to another model available to your account.

## Architecture and algorithm

`llm_planner.py` asks Gemini for a schema-constrained plan; the backend then independently checks every tool and dependency before creating the graph. It builds tool arguments in deterministic code and never lets the model execute arbitrary code. `planner.py` is the no-key fallback and creates explicit task nodes from simple natural-language conjunctions. `engine.py` validates references and DAG acyclicity, launches each ready frontier concurrently under a semaphore, validates typed tool outputs before releasing successors, and retries failures with bounded exponential backoff. The tool registry includes local research search, safe numeric calculations, report creation, contained user-document reading, text statistics, dataset checks, and unit conversion. Document reads are restricted to small `.txt`, `.md`, `.csv`, or `.json` files directly inside `backend/data/user_documents/`. No shell or general-purpose code execution tool exists.

Every workflow returns Gemini-reported input, output and total tokens, plus cached-input and thinking-token breakdowns when provided by the API. To estimate USD cost, set `GEMINI_INPUT_USD_PER_1M`, `GEMINI_OUTPUT_USD_PER_1M`, and `GEMINI_CACHED_INPUT_USD_PER_1M` using the rates that apply to the selected model and account. No prices are built in; without all three rates, the cost is reported as unpriced. A failed API request may not return token usage, so failed-call usage can be unavailable.

`app/mcp/client_manager.py` uses official MCP Python SDK stdio sessions to initialize servers, discover schemas, and invoke tools. Sample servers can be started with `python -m servers.local_servers --server search|analysis|document|files|text|dataset|units` (from `backend`). MCP server connections are opt-in via JSON in `MCP_SERVERS`; when configured, matching discovered MCP tools back their corresponding built-in capabilities. The API and terminal CLI still expose local tools as fallback. Model token counts are provider-reported, and dollar rates are user-configured; local tool execution estimates are zero. MCP SDK is pinned below 2.0 because the sample servers use its stable v1 `FastMCP` API.

## API

- `GET /health`, `GET /tools`
- `POST /workflows` with `{ "task": "...", "max_concurrency": 4, "max_retries": 1 }`
- `GET /workflows/{id}`, `GET /workflows/{id}/graph`, `GET /workflows/{id}/metrics`
- `POST /workflows/{id}/run`
- `POST /workflows/{id}/resume` retries failed steps after a transient problem is fixed, reusing completed step outputs.
- `GET /benchmarks`, `POST /benchmarks/run?trials=1`, `GET /benchmarks/results`

Workflow responses include `planner_mode` and `llm_usage`; metrics repeat token totals and estimated model cost. LLM usage is included in the workflow's `planner_usage` event.

Workflow checkpoints are atomically saved as local JSON under `backend/data/workflows/` and restored after backend restart. Completed step outputs are reused when a failed workflow is resumed. The checkpoint includes workflow inputs and tool outputs, so keep this local folder private; it is ignored by Git. This is single-process local persistence, not a shared database or encrypted production store. Benchmark files are written to `backend/data/`.

## Benchmarking

The paired benchmark includes normal tasks and pipeline stress cases: parallel fan-out/fan-in, a transient timeout, a malformed tool response, and a failed prerequisite that must block downstream work. The direct baseline runs the same authored tool steps serially without validation or retries; MCPorch uses dependency-aware scheduling, output validation, and bounded retry. Run with `POST /benchmarks/run?trials=3`. JSON and CSV raw runs plus a Markdown report are written to `backend/data/`. A fixed simulated API round trip is used so parallel scheduling can be measured; it is not a live provider timing or billing result. The benchmark deliberately does not call Gemini, so its token cost is zero and does not measure planning quality. The stress suite is synthetic and should be interpreted as a focused engineering check, not general proof of superiority.

## Limitations

The deterministic fallback only understands lightweight task phrasing and simple sequential conjunctions; it does not robustly infer arbitrary multi-step dependencies. Dataset and text tools provide lightweight descriptive checks, not a replacement for a full data-science or NLP package. Local document access is opt-in by placing files in the dedicated folder. The dashboard shows the final run state rather than streaming updates. In-memory state is single-process. These are explicit prototype limits.
