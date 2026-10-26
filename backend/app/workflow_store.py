"""Small atomic local checkpoints for workflow state and resumable execution."""
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from .models import Workflow

_LOCK = RLock()


def store_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "workflows"


def save_workflow(workflow: Workflow) -> None:
    """Atomically persist a workflow snapshot, including completed tool outputs."""
    folder = store_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{workflow.workflow_id}.json"
    temporary = folder / f".{workflow.workflow_id}.{os.getpid()}.tmp"
    payload = workflow.model_dump_json(indent=2)
    with _LOCK:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)


def load_workflows() -> dict[str, Workflow]:
    """Load valid persisted workflows; ignore partial/corrupt files safely."""
    folder = store_dir()
    workflows: dict[str, Workflow] = {}
    if not folder.exists():
        return workflows
    for path in folder.glob("*.json"):
        try:
            workflow = Workflow.model_validate_json(path.read_text(encoding="utf-8"))
            workflows[workflow.workflow_id] = workflow
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return workflows
