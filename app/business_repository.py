"""SQLite business state, isolated from the vector store and model state."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from app.business_errors import BusinessError

ROOT = Path(__file__).resolve().parents[1]
DEMO_EMPLOYEES = {"E001": 5, "E002": 0, "E003": 12, "E004": 3, "E005": 8}
SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
 employee_id TEXT PRIMARY KEY, leave_balance INTEGER NOT NULL CHECK(leave_balance >= 0)
);
CREATE TABLE IF NOT EXISTS tickets (
 ticket_id TEXT PRIMARY KEY, requester_id TEXT NOT NULL REFERENCES employees(employee_id),
 title TEXT NOT NULL CHECK(length(title) BETWEEN 5 AND 100),
 description TEXT NOT NULL CHECK(length(description) BETWEEN 1 AND 2000),
 priority TEXT NOT NULL CHECK(priority IN ('P1','P2','P3')),
 status TEXT NOT NULL CHECK(status IN ('open','in_progress','closed')),
 created_at TEXT NOT NULL, request_key TEXT NOT NULL, request_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency (
 requester_id TEXT NOT NULL REFERENCES employees(employee_id), request_key TEXT NOT NULL,
 request_hash TEXT NOT NULL, ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id), created_epoch REAL NOT NULL,
 PRIMARY KEY(requester_id, request_key)
);
"""


class SQLiteBusinessRepository:
    def __init__(self, path: str | Path = "var/business.sqlite3", *, before_commit=None):
        value = Path(path)
        self.path = (value if value.is_absolute() else ROOT / value).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.before_commit = before_commit
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            yield connection
        finally:
            connection.close()

    def seed_demo(self):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany("INSERT INTO employees(employee_id,leave_balance) VALUES (?,?) ON CONFLICT(employee_id) DO NOTHING", DEMO_EMPLOYEES.items())
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_leave_balance(self, employee_id):
        with self.connection() as connection:
            row = connection.execute("SELECT leave_balance FROM employees WHERE employee_id=?", (employee_id,)).fetchone()
        if row is None:
            raise BusinessError("EMPLOYEE_NOT_FOUND", "员工不存在")
        return row["leave_balance"]

    def ticket_owner(self, ticket_id):
        with self.connection() as connection:
            row = connection.execute("SELECT requester_id FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return row["requester_id"] if row else None

    def get_own_ticket(self, ticket_id, employee_id):
        with self.connection() as connection:
            row = connection.execute("SELECT ticket_id,requester_id,title,description,priority,status,created_at FROM tickets WHERE ticket_id=? AND requester_id=?", (ticket_id, employee_id)).fetchone()
        return dict(row) if row else None

    def create_ticket(self, employee_id, payload, request_key, now):
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute("SELECT * FROM idempotency WHERE requester_id=? AND request_key=?", (employee_id, request_key)).fetchone()
                if existing and now - existing["created_epoch"] < 86400:
                    if existing["request_hash"] != digest:
                        raise BusinessError("IDEMPOTENCY_CONFLICT", "同一请求键对应的工单内容不一致")
                    row = connection.execute("SELECT ticket_id,requester_id,title,description,priority,status,created_at FROM tickets WHERE ticket_id=?", (existing["ticket_id"],)).fetchone()
                    connection.commit()
                    return {**dict(row), "idempotent_replay": True}
                if existing:
                    connection.execute("DELETE FROM idempotency WHERE requester_id=? AND request_key=?", (employee_id, request_key))
                if connection.execute("SELECT 1 FROM employees WHERE employee_id=?", (employee_id,)).fetchone() is None:
                    raise BusinessError("EMPLOYEE_NOT_FOUND", "员工不存在")
                ticket_id = "T" + uuid.uuid4().hex[:20]
                created_at = datetime.fromtimestamp(now, timezone.utc).isoformat()
                connection.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?)", (ticket_id, employee_id, payload["title"], payload["description"], payload["priority"], "open", created_at, request_key, digest))
                connection.execute("INSERT INTO idempotency VALUES (?,?,?,?,?)", (employee_id, request_key, digest, ticket_id, now))
                if self.before_commit:
                    self.before_commit(connection)
                connection.commit()
                return {"ticket_id": ticket_id, "requester_id": employee_id, **payload, "status": "open", "created_at": created_at, "idempotent_replay": False}
            except Exception:
                connection.rollback()
                raise

    def recover_own_request(self, employee_id, request_key):
        """Host-only recovery, including after the idempotency cache expires.

        This is never exposed as a model tool. The conversation host derives the
        key and owner; callers cannot use an arbitrary employee or ticket ID.
        """
        with self.connection() as connection:
            row = connection.execute(
                "SELECT ticket_id,requester_id,title,description,priority,status,created_at FROM tickets "
                "WHERE requester_id=? AND request_key=? ORDER BY created_at LIMIT 1",
                (employee_id, request_key)).fetchone()
        return dict(row) if row else None
