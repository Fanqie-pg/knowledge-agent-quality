"""Durable owner-bound sessions and fenced, idempotent conversation turns."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid

from app.business_errors import BusinessError
from app.identity import require_identity
from app.tool_permissions import identity_binding


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
 session_id TEXT PRIMARY KEY, principal_id TEXT NOT NULL, employee_id TEXT NOT NULL,
 binding TEXT NOT NULL, updated REAL NOT NULL, epoch_start INTEGER NOT NULL DEFAULT 0,
 facts TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS turns (
 session_id TEXT NOT NULL REFERENCES sessions(session_id), turn_id TEXT NOT NULL,
 sequence INTEGER NOT NULL, query_hash TEXT NOT NULL, binding TEXT NOT NULL,
 query TEXT NOT NULL, status TEXT NOT NULL, request_key TEXT NOT NULL,
 worker_token TEXT NOT NULL, lease_until REAL NOT NULL, result TEXT, checkpoint TEXT,
 PRIMARY KEY(session_id,turn_id), UNIQUE(session_id,sequence)
);
CREATE TABLE IF NOT EXISTS turn_attempts (
 session_id TEXT NOT NULL, turn_id TEXT NOT NULL, worker_token TEXT NOT NULL,
 status TEXT NOT NULL, result TEXT, checkpoint TEXT, PRIMARY KEY(session_id,turn_id,worker_token)
);
"""


class ConversationStore:
    def __init__(self, path="var/conversations.sqlite3", *, ttl_seconds=1800, lease_seconds=300,
                 max_turns=40, max_sessions=200, history_turns=8, history_chars=12000, clock=time.time):
        if min(ttl_seconds, lease_seconds, max_turns, max_sessions, history_turns, history_chars) <= 0:
            raise ValueError("会话限制必须大于0")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl, self.lease, self.max_turns, self.max_sessions = ttl_seconds, lease_seconds, max_turns, max_sessions
        self.history_turns, self.history_chars, self.clock = history_turns, history_chars, clock
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            if conn.in_transaction:
                conn.rollback()
            conn.close()

    @staticmethod
    def valid_id(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError("会话和轮次标识必须为1到128字符")

    def _owned(self, conn, session_id, context):
        require_identity(context, "session.read")
        row = conn.execute("SELECT * FROM sessions WHERE session_id=? AND principal_id=? AND employee_id=?",
                           (session_id, context.principal_id, context.employee_id)).fetchone()
        if row is None:
            raise BusinessError("SESSION_NOT_ACCESSIBLE", "会话不存在或无权访问")
        if self.clock() - row["updated"] > self.ttl:
            raise BusinessError("SESSION_EXPIRED", "会话已过期，请开始新会话")
        return row

    def create(self, context, session_id=None):
        require_identity(context, "session.create")
        session_id = session_id or uuid.uuid4().hex
        self.valid_id(session_id)
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row:
                self._owned(conn, session_id, context)
            else:
                if conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= self.max_sessions:
                    raise BusinessError("SESSION_CAPACITY", "会话存储已达到容量限制")
                conn.execute("INSERT INTO sessions(session_id,principal_id,employee_id,binding,updated) VALUES(?,?,?,?,?)",
                             (session_id, context.principal_id, context.employee_id, identity_binding(context), self.clock()))
            conn.commit()
        return session_id

    def claim(self, session_id, turn_id, query, query_hash, request_key, context):
        self.valid_id(session_id)
        self.valid_id(turn_id)
        binding, now = identity_binding(context), self.clock()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            session = self._owned(conn, session_id, context)
            old = conn.execute("SELECT * FROM turns WHERE session_id=? AND turn_id=?", (session_id, turn_id)).fetchone()
            if old and old["query_hash"] != query_hash:
                raise BusinessError("TURN_CONTENT_CONFLICT", "同一轮次标识不能对应不同内容")
            if old and old["binding"] != binding:
                raise BusinessError("TURN_PERMISSION_CHANGED", "权限已变化，不能复用旧轮次结果，请发起新轮次")
            if old and old["status"] in {"completed", "limited", "partial"}:
                conn.commit()
                return {"cached": json.loads(old["result"])}
            active = conn.execute("SELECT * FROM turns WHERE session_id=? AND status IN ('running','uncertain') ORDER BY sequence DESC LIMIT 1", (session_id,)).fetchone()
            if active and active["turn_id"] != turn_id:
                raise BusinessError("SESSION_BUSY", "本会话有未完成轮次，请先重试原轮次")
            if old and old["status"] == "running" and old["lease_until"] > now:
                raise BusinessError("SESSION_BUSY", "本轮正在处理中，请稍后重试相同轮次")
            latest = conn.execute("SELECT COALESCE(MAX(sequence),0) FROM turns WHERE session_id=?", (session_id,)).fetchone()[0]
            if old and old["sequence"] < latest:
                raise BusinessError("TURN_SUPERSEDED", "旧轮次已被后续轮次替代，请发起新轮次")
            if session["binding"] != binding:
                conn.execute("UPDATE sessions SET binding=?,epoch_start=?,facts='{}' WHERE session_id=?", (binding, latest, session_id))
                session = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not old and latest >= self.max_turns:
                raise BusinessError("SESSION_TURN_LIMIT", "本会话已达到轮次数量限制，请开始新会话")
            token = uuid.uuid4().hex
            sequence = old["sequence"] if old else latest + 1
            if old:
                conn.execute("INSERT OR IGNORE INTO turn_attempts VALUES (?,?,?,?,?,?)",
                             (session_id, turn_id, old["worker_token"], old["status"], old["result"], old["checkpoint"]))
                conn.execute("UPDATE turns SET status='running',worker_token=?,lease_until=? WHERE session_id=? AND turn_id=?",
                             (token, now + self.lease, session_id, turn_id))
            else:
                conn.execute("INSERT INTO turns(session_id,turn_id,sequence,query_hash,binding,query,status,request_key,worker_token,lease_until) VALUES(?,?,?,?,?,?,'running',?,?,?)",
                             (session_id, turn_id, sequence, query_hash, binding, query, request_key, token, now + self.lease))
            rows = conn.execute("SELECT query,result FROM turns WHERE session_id=? AND sequence>? AND sequence<? AND status IN ('completed','limited','partial') ORDER BY sequence DESC LIMIT ?",
                                (session_id, session["epoch_start"], sequence, self.history_turns)).fetchall()
            history, chars = [], 0
            for row in rows:
                result = json.loads(row["result"])
                item = {"user": row["query"], "assistant": result["answer"][:4000]}
                size = len(item["user"]) + len(item["assistant"])
                if chars + size > self.history_chars:
                    break
                history.append(item)
                chars += size
            conn.execute("UPDATE sessions SET updated=? WHERE session_id=?", (now, session_id))
            conn.commit()
            return {"token": token, "sequence": sequence, "history": history[::-1], "facts": json.loads(session["facts"]),
                    "retry": old is not None, "checkpoint": json.loads(old["checkpoint"]) if old and old["checkpoint"] else None}

    def checkpoint(self, session_id, turn_id, token, facts, checkpoint):
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute("UPDATE turns SET checkpoint=? WHERE session_id=? AND turn_id=? AND worker_token=? AND status='running'",
                                   (json.dumps(checkpoint, ensure_ascii=False), session_id, turn_id, token)).rowcount
            if not updated:
                raise BusinessError("TURN_LEASE_LOST", "本轮执行凭据已失效，停止后续操作")
            conn.execute("UPDATE sessions SET facts=?,updated=? WHERE session_id=?", (json.dumps(facts, ensure_ascii=False), self.clock(), session_id))
            conn.commit()

    def finish(self, session_id, turn_id, token, result, status="completed"):
        with self.connection() as conn:
            updated = conn.execute("UPDATE turns SET status=?,result=?,lease_until=0 WHERE session_id=? AND turn_id=? AND worker_token=? AND status='running'",
                                   (status, json.dumps(result, ensure_ascii=False), session_id, turn_id, token)).rowcount
            if not updated:
                raise BusinessError("TURN_LEASE_LOST", "本轮执行凭据已失效")

    def inspect(self, session_id, context):
        with self.connection() as conn:
            row = self._owned(conn, session_id, context)
            return {"facts": json.loads(row["facts"]) if row["binding"] == identity_binding(context) else {},
                    "binding": identity_binding(context)}
