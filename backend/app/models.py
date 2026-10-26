"""Validated workflow and execution models."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


class Status(str, Enum):
    pending = "pending"
    running = "running"
    recovering = "recovering"
    completed = "completed"
    failed = "failed"


class TaskSpec(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(min_length=1)
    selected_tool: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    input_data: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    status: Status = Status.pending
    output_data: Any = None
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    observed_latency: float = 0.0
    retry_count: int = 0
    validation_status: str = "not_checked"
    error_message: str | None = None
    recovery_history: list[str] = Field(default_factory=list)


class WorkflowCreate(BaseModel):
    task: str = Field(min_length=1)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    max_retries: int = Field(default=1, ge=0, le=5)


class Workflow(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    task: str
    nodes: list[TaskSpec]
    created_at: datetime = Field(default_factory=now)
    status: Status = Status.pending
    max_concurrency: int = 4
    max_retries: int = 1
    events: list[dict[str, Any]] = Field(default_factory=list)
    planner_mode: str = "deterministic"
    llm_usage: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    workflow_id: str
    status: Status
    result: Any = None
    events: list[dict[str, Any]]
