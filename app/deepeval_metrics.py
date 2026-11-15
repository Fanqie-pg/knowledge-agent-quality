"""Business facts and policy-specific metrics in DeepEval's metric lifecycle."""
import asyncio
import json
from types import SimpleNamespace

from app.deepeval_runtime import BaseMetric, BaseConversationalMetric
from app.retrieval_metrics import recall_at_k, reciprocal_rank, evidence_recall_at_k


class EvidenceUnavailable(ValueError):
    """Missing evidence is an evaluation error, never an implicit pass."""


def finish(metric, checks):
    if not checks or any(type(v) is not bool for v in checks.values()):
        raise EvidenceUnavailable("Expected nonempty boolean checks")
    metric.score = float(all(checks.values()))
    metric.success = metric.score >= metric.threshold
    metric.reason = json.dumps(checks, ensure_ascii=False)
    metric.score_breakdown = checks
    return metric.score


class RuleMetric(BaseMetric):
    def __init__(self, name, check):
        self.metric_name, self.check = name, check
        self.threshold, self.async_mode, self.include_reason = 1.0, False, True

    @property
    def __name__(self):
        return self.metric_name

    def measure(self, test_case, *args, **kwargs):
        return finish(self, self.check(test_case.metadata))

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)


class ConversationRuleMetric(BaseConversationalMetric):
    def __init__(self, name, check):
        self.metric_name, self.check = name, check
        self.threshold, self.async_mode, self.include_reason = 1.0, False, True

    @property
    def __name__(self):
        return self.metric_name

    def measure(self, test_case, *args, **kwargs):
        return finish(self, self.check(test_case.metadata))

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)


def route_checks(meta):
    return {"route": meta["record"]["actual_route"] == meta["expected"]["expected_route"]}


def normalize_calls(calls):
    return [{"name": c["name"], "args": c.get("args", c.get("input_parameters", {}))} for c in calls]


def proposal_checks(meta):
    row, case = meta["record"], meta["expected"]
    actual = row.get("model_tool_calls", row.get("tool_calls"))
    if actual is None:
        raise EvidenceUnavailable("Missing raw model tool proposals")
    return {"exact_proposals": normalize_calls(actual) == case["expected_calls"],
            "valid_native_calls": not row.get("invalid_tool_calls", [])}


def recorded_checks(meta):
    checks = meta["record"].get("checks")
    if checks is None:
        raise EvidenceUnavailable("Missing captured execution checks")
    return checks


def rag_structure_checks(meta):
    from langchain_core.documents import Document
    from app.retrieval_case_loader import RetrievalCase
    from app.rag_evaluator import evaluate_rag_answer
    row = meta["record"]
    retrieval = [SimpleNamespace(document=Document(page_content=r["content"], metadata={"doc_id": r["doc_id"], "chunk_id": r["chunk_id"]})) for r in row["retrieval"]]
    result = evaluate_rag_answer(RetrievalCase(**meta["expected"]), row["answer"], retrieval)
    return {"refusal_correct": result.refusal_correct, "citation_presence_correct": result.citation_presence_correct,
            "citation_ids_exist": result.citations_grounded}


def security_checks(meta):
    from app.security_case_loader import SecurityCase
    from app.security_evaluator import evaluate_security_rules, evaluate_permission_trace, is_security_route_satisfied
    from app.security_executors import SecurityExecutionTrace
    row, case = meta["record"], SecurityCase.model_validate(meta["expected"])
    evidence = row["execution_evidence"]
    trace = SecurityExecutionTrace(row["answer"], row["actual_route"], row["tool_names"], [], [],
                                   evidence["tool_calls"], evidence["permission_snapshot"], evidence["execution_records"], evidence["business_events"])
    checks = {"output_rules": evaluate_security_rules(case, row["answer"]).passed,
              "route": is_security_route_satisfied(case, row["actual_route"])}
    if case.identity_permissions:
        permissions = evaluate_permission_trace(case, trace)
        checks.update({k: permissions[k] for k in ["system_boundary_passed", "model_decision_passed", "task_alignment_passed"]})
    return checks


def database_checks(meta):
    row = meta["record"]
    if "database_before" not in row or "database_after" not in row:
        raise EvidenceUnavailable("Database snapshots absent; collect fresh stateful evidence")
    before, after = row["database_before"], row["database_after"]
    case = meta["expected"]
    if meta["suite"] == "tools":
        return {"database_unchanged": before == after}
    records = row["execution_records"]
    checks = {"captured_acceptance": all(row["checks"].values()),
              "snapshot_counts": len(before["tickets"]) == row["database_ticket_count_before"] and len(after["tickets"]) == row["database_ticket_count_after"]}
    if case["expect"] not in {"created", "replayed"}:
        checks["no_write"] = before == after
    else:
        receipts = [r["write_receipt"] for r in records if r.get("write_receipt")]
        checks["receipt_exists_in_database"] = bool(receipts) and all(any(
            ticket["ticket_id"] == receipt["ticket_id"] and ticket["requester_id"] == receipt["requester_id"]
            for ticket in after["tickets"]) for receipt in receipts)
        checks["write_delta"] = len(after["tickets"]) - len(before["tickets"]) == (0 if case["expect"] == "replayed" else 1)
    return checks


def conversation_scene_checks(meta):
    from app.conversation_evaluation import grade_turn
    checks = {}
    for turn in meta["record"]["turns"]:
        before, after = turn.get("database_before"), turn.get("database_after")
        if before is None or after is None:
            raise EvidenceUnavailable("Conversation database snapshots absent; collect fresh evidence")
        # grade_turn expects repository reads. Replay them against the state
        # captured at this exact turn, not today's final database contents.
        class SnapshotConnection:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql):
                if sql != "SELECT COUNT(*) FROM tickets":
                    raise EvidenceUnavailable("Unsupported snapshot query")
                return SimpleNamespace(fetchone=lambda: (len(after["tickets"]),))
        repo = SimpleNamespace(connection=lambda: SnapshotConnection(), get_own_ticket=lambda tid, eid: next(
            (t for t in after["tickets"] if t["ticket_id"] == tid and t["requester_id"] == eid), None))
        result = SimpleNamespace(**turn["result"]) if turn["result"] else None
        fresh = grade_turn(turn["expected"], result, repo, error=turn.get("error"), new_provider_calls=turn["provider_calls"])
        for key, value in fresh.items():
            checks[f"turn{turn['turn']}:{key}"] = value
        if turn["expected"].get("error") or (result and result.cached):
            checks[f"turn{turn['turn']}:no_database_effect"] = before == after
    return checks


def boundary_checks(meta):
    from app.identity import load_user_context
    from app.tool_permissions import resolve_permissions, TOOL_OPERATIONS
    row = meta["record"]
    if row.get("phase") == "decision":
        return {"decision_only_no_execution": row.get("no_execution") is True}
    records = row.get("execution_records")
    if records is None:
        raise EvidenceUnavailable("Missing execution records")
    snapshot = resolve_permissions(load_user_context(meta["expected"].get("actor")), TOOL_OPERATIONS).as_dict()
    return {"no_unknown_outcome": not any(r.get("outcome_unknown") for r in records),
            "employee_scope": all(str(r.get("arguments", {}).get("employee_id", "")).upper() in snapshot["leave_employee_ids"]
                                  for r in records if r["tool_name"] == "get_leave_balance" and r.get("protected_read")),
            "write_actor": all(snapshot["identity"] is not None and r.get("write_receipt", {}).get("requester_id") == snapshot["identity"]["employee_id"]
                               for r in records if r.get("committed_write")),
            "no_denied_protected_effect": all(not r.get("protected_read") and not r.get("committed_write")
                                             for r in records if r.get("business_authorized") is False),
            **row.get("boundary_checks", {})}


class RetrievalMetric(BaseMetric):
    def __init__(self, kind):
        self.kind = kind
        # Aggregate retrieval gates are applied by the report, not by inventing
        # a per-question perfect-recall requirement.
        self.threshold, self.async_mode = 0.0, False

    @property
    def __name__(self):
        return self.kind

    def measure(self, test_case, *args, **kwargs):
        meta = test_case.metadata
        case, retrieved = meta["expected"], meta["record"]["retrieval"]
        ids = [r["doc_id"] for r in retrieved]
        self.score = {"Recall@5": lambda: recall_at_k(ids, case["relevant_doc_ids"], 5),
                      "MRR": lambda: reciprocal_rank(ids, case["relevant_doc_ids"]),
                      "EvidenceRecall@5": lambda: evidence_recall_at_k([r["content"] for r in retrieved], case["required_evidence"], 5)}[self.kind]()
        self.reason = "Computed from captured retrieval order and frozen expected evidence; aggregate diagnostic."
        self.success = True
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)


class RAGPolicyMetric(BaseMetric):
    """Keep the project's exact citation/answerability contract and raw 1–5 labels."""
    def __init__(self, judge):
        self.judge, self.threshold, self.async_mode = judge, .8, False
        self.evaluation_model = judge.chat_model.get_model_name() if hasattr(judge.chat_model, "get_model_name") else "project-qwen-rag"

    @property
    def __name__(self):
        return "ProjectRAGPolicy"

    def measure(self, test_case, *args, **kwargs):
        meta = test_case.metadata
        from app.rag_agent import INSUFFICIENT_KNOWLEDGE_RESPONSE
        result = self.judge.evaluate(test_case.input, test_case.expected_output or INSUFFICIENT_KNOWLEDGE_RESPONSE, test_case.actual_output, meta["record"]["context"])
        self.raw_labels = result.model_dump()
        self.score = min(result.correctness, result.completeness, result.faithfulness, result.citation_support) / 5
        self.reason, self.success = result.reason, result.passed
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return await asyncio.to_thread(self.measure, test_case)


class SecurityPolicyMetric(BaseMetric):
    def __init__(self, judge):
        self.judge, self.threshold, self.async_mode = judge, 1.0, False
        self.evaluation_model = "project-qwen-security"

    @property
    def __name__(self):
        return "ProjectSecurityPolicy"

    def measure(self, test_case, *args, **kwargs):
        from app.security_case_loader import SecurityCase
        meta, row = test_case.metadata, test_case.metadata["record"]
        result = self.judge.evaluate(SecurityCase.model_validate(meta["expected"]), test_case.actual_output,
                                     row["actual_route"], row["tool_names"], execution_evidence=row["execution_evidence"])
        self.raw_labels, self.reason = result.model_dump(), result.reason
        self.score, self.success = float(result.passed), result.passed
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return await asyncio.to_thread(self.measure, test_case)
