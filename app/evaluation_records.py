"""Durable, versioned evaluation records and a provider-call budget."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
import threading

from app.dataset_governance import ROOT, sha256, read_jsonl
from app.tracing import redact


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(redact(record), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def provenance(root=ROOT):
    root = Path(root)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    paths = [*sorted((root / "app").glob("*.py")), *sorted((root / "data/knowledge").glob("*.md")),
             *sorted((root / "data/evaluation").glob("*.json*"))]
    return {"git_commit": commit, "files": {p.relative_to(root).as_posix(): sha256(p) for p in paths}}


class RunStore:
    """Single-writer run directory. Existing rows are never overwritten."""
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "run.json"
        config = redact(config)
        if path.exists():
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            if self.manifest["config_hash"] != fingerprint(config):
                raise ValueError("运行配置或版本发生变化，请使用新的输出目录")
        else:
            self.manifest = {"run_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                             "config": config, "config_hash": fingerprint(config)}
            path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.path = self.directory / "records.jsonl"

    def rows(self):
        return read_jsonl(self.path) if self.path.exists() else []

    def append(self, row):
        append_jsonl(self.path, {**row, "run_id": self.manifest["run_id"]})

    def report(self, name, value):
        (self.directory / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class CallBudgetExceeded(RuntimeError):
    pass


class CallBudget:
    _locks = {}
    _locks_guard = threading.Lock()

    def __init__(self, path, maximum):
        if maximum <= 0:
            raise ValueError("max_calls必须大于0")
        self.path, self.maximum = Path(path), maximum
        with self._locks_guard:
            self._lock = self._locks.setdefault(str(self.path.resolve()), threading.Lock())
        self.used = len(read_jsonl(self.path)) if self.path.exists() else 0

    def consume(self):
        with self._lock:
            self._consume_locked()

    def _consume_locked(self):
        # A synchronous test session may create several model wrappers sharing a ledger.
        if self.path.exists():
            self.used = max(self.used, len(read_jsonl(self.path)))
        if self.used >= self.maximum:
            raise CallBudgetExceeded("模型调用预算已耗尽")
        self.used += 1
        # Persist before the provider call; failed/uncertain calls still consume budget.
        append_jsonl(self.path, {"call": self.used, "time": datetime.now(timezone.utc).isoformat()})


class BudgetedModel:
    def __init__(self, model, budget):
        self.model, self.budget = model, budget

    def invoke(self, *args, **kwargs):
        self.budget.consume()
        return self.model.invoke(*args, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return BudgetedModel(self.model.bind_tools(tools, **kwargs), self.budget)


def attach_budget(model, budget):
    # Put the budget inside RetryingChatModel so every provider attempt is charged.
    from app.resilient_model import RetryingChatModel
    if isinstance(model, RetryingChatModel):
        model.model = BudgetedModel(model.model, budget)
        return model
    return BudgetedModel(model, budget)
