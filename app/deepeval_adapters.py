"""Versioned source records -> real DeepEval test cases, without re-generation."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from app.deepeval_runtime import LLMTestCase, ToolCall, Turn, ConversationalTestCase
from app.dataset_governance import ROOT, read_jsonl, sha256
from app.evaluation_records import fingerprint

PROFILE = ROOT / "data/evaluation/deepeval_v1/sources.json"


@dataclass
class EvaluationItem:
    key: str
    suite: str
    test_case: object
    source: dict


def tool_calls(calls):
    return [ToolCall(name=c["name"], input_parameters=c.get("args", {}), output=c.get("output")) for c in calls]


def execution_calls(row):
    """Never turn blocked/proposed-only calls into actual execution."""
    return [ToolCall(name=r["tool_name"], input_parameters=r.get("arguments", {}), output=r.get("output"))
            for r in row.get("execution_records", []) if r.get("invoked")]


def context_chunks(context):
    chunks = re.split(r"(?=\[来源：[^\]]+\])", context)
    return [c.strip() for c in chunks if c.strip()]


def security_evidence(row):
    return row.get("execution_evidence") or {k: row.get(k, {} if k == "permission_snapshot" else [])
                                            for k in ("tool_calls", "permission_snapshot", "execution_records", "business_events")}


def single_item(suite, key, expected, record, source, *, query=None, answer=None):
    record = dict(record)
    if suite in {"security", "calibration-security"}:
        record["execution_evidence"] = security_evidence(record)
    question = query if query is not None else expected.get("question", expected.get("query", ""))
    output = answer if answer is not None else record.get("answer", record.get("decision_content", record.get("actual_route", "")))
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    # Pure model-decision interactions may have no prose. Preserve the actual
    # structured output, not a fabricated final business answer.
    if not output and "model_tool_calls" in record:
        output = json.dumps({"tool_calls": record["model_tool_calls"]}, ensure_ascii=False)
    retrieved = [r["content"] for r in record.get("retrieval", [])] or context_chunks(record.get("context", ""))
    test_case = LLMTestCase(name=key, input=question, actual_output=output,
                            expected_output=expected.get("reference_answer"),
                            retrieval_context=retrieved or None,
                            tools_called=execution_calls(record),
                            expected_tools=tool_calls(expected.get("expected_calls", [])),
                            metadata={"expected": expected, "record": record, "source": source, "suite": suite})
    return EvaluationItem(key, suite, test_case, source)


def source_rows(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    digest = sha256(path)
    for row in read_jsonl(path):
        yield row, {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": digest,
                    "row_hash": fingerprint(row), "run_id": row.get("run_id"), "new_agent_generation": False}


def _case_index(name):
    return {r["case_id"]: r for r in read_jsonl(ROOT / "data/evaluation" / name)}


def _same_case(row, expected):
    if row.get("case") != expected:
        raise ValueError("Source case differs from active data: " + expected["case_id"])


def load_items(profile=PROFILE, suites=None):
    profile = json.loads(Path(profile).read_text(encoding="utf-8"))
    wants = lambda name: suites is None or name in suites
    items = []
    rag = _case_index("retrieval_cases.jsonl")
    for variant in (("direct", "rewrite") if wants("rag") else ()):
        seen = set()
        for row, source in source_rows(profile["rag_" + variant]):
            if row.get("suite") != "rag" or row.get("variant") != variant or "case" not in row or row.get("case_id") not in rag:
                continue
            case = rag[row["case_id"]]
            _same_case(row, case)
            if row["case_id"] in seen:
                raise ValueError("Duplicate RAG source")
            seen.add(row["case_id"])
            items.append(single_item("rag", f"rag/{variant}/{row['case_id']}", case, row, source))
        if seen != set(rag):
            raise ValueError("Incomplete RAG source: " + variant)
    routes = _case_index("routing_cases.jsonl")
    seen = set()
    for row, source in (source_rows(profile["routing"]) if wants("routing") else ()):
        if row.get("suite") == "routing" and row.get("case_id") in routes:
            case = routes[row["case_id"]]
            _same_case(row, case)
            if row["case_id"] in seen:
                raise ValueError("Duplicate routing source")
            seen.add(row["case_id"])
            items.append(single_item("routing", "routing/" + row["case_id"], case, row, source))
    if wants("routing") and seen != set(routes):
        raise ValueError("Incomplete routing source")
    tools = _case_index("tool_control_v2/cases.jsonl")
    for row, source in (source_rows(profile["tools"]) if wants("tools") else ()):
        _same_case(row, tools[row["case_id"]])
        suffix = f"{row['phase']}/{row['transport']}/{row['trial']}/{row['case_id']}"
        items.append(single_item("tools", "tools/" + suffix, tools[row["case_id"]], row, source))
    business = _case_index("business_v1/agent_cases.jsonl")
    business_routes = _case_index("business_v1/routing_cases.jsonl")
    for row, source in (source_rows(profile["business"]) if wants("business") or wants("business-routing") else ()):
        suite = "business-routing" if row["transport"] == "routing" else "business"
        if not wants(suite): continue
        case = (business_routes if suite == "business-routing" else business)[row["case_id"]]
        items.append(single_item(suite, f"{suite}/{row['transport']}/{row['case_id']}", case, row, source, query=row["query"]))
    security = _case_index("security_cases.jsonl")
    for row, source in (source_rows(profile["security"]) if wants("security") else ()):
        _same_case(row, security[row["case_id"]])
        items.append(single_item("security", "security/" + row["case_id"], security[row["case_id"]], row, source))
    if wants("conversation") or wants("conversation-scene"):
        items.extend(i for i in conversation_items(profile["conversation"]) if wants(i.suite))
    from app.calibration import load_references, validate_references
    calibration_requested = wants("calibration-rag") or wants("calibration-security")
    references = load_references() if calibration_requested else []
    if calibration_requested:
        validate_references(read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl"), references)
    reference_index = {reference["sample_id"]: reference for reference in references}
    for row, source in (source_rows("data/evaluation/calibration/samples.jsonl") if calibration_requested else ()):
        suite = "calibration-" + row["suite"]
        if not wants(suite): continue
        if row["sample_id"] not in reference_index:
            raise ValueError("校准样本缺少参考标签：" + row["sample_id"])
        reference = reference_index[row["sample_id"]]
        source.update(label_kind=reference["origin"], labels=reference["labels"],
                      split=row["split"], group_id=row["group_id"], synthetic_calibration=True)
        case = row if row["suite"] == "rag" else row["security_case"]
        items.append(single_item(suite, suite + "/" + row["sample_id"], case, row, source))
    for row, source in (source_rows("data/evaluation/deepeval_v1/native_calibration.jsonl") if wants("calibration-native") else ()):
        key, suite = "calibration-native/" + row["sample_id"], "calibration-native"
        source.update(label_kind="synthetic_contract", labels={"passed": row["expected_passed"]},
                      split="calibration", group_id=row["group_id"], synthetic_calibration=True)
        if "turns" in row:
            case = ConversationalTestCase(name=key, turns=[Turn(**t) for t in row["turns"]],
                                           metadata={"record": row, "expected": row, "source": source})
            items.append(EvaluationItem(key, suite, case, source))
        else:
            items.append(single_item(suite, key, row, row, source))
    keys = [x.key for x in items]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluation item")
    coverage = {suite: sum(i.suite == suite for i in items) for suite in {i.suite for i in items}}
    required = {"rag": 120, "routing": 36, "tools": 120, "business": 26, "business-routing": 12,
                "security": 30, "conversation-scene": 32, "calibration-rag": 40, "calibration-security": 20, "calibration-native": 12}
    if any(coverage.get(suite) != count for suite, count in required.items() if wants(suite)):
        raise ValueError("Incomplete framework coverage: " + json.dumps(coverage))
    return items


def conversation_items(paths):
    cases = {c["case_id"]: c for c in json.loads((ROOT / "data/evaluation/conversation_v1/cases.json").read_text(encoding="utf-8"))["cases"]}
    indexed = {}
    for path in paths:
        for row, source in source_rows(path):
            indexed[(row["case_id"], row["transport"])] = (row, source)
    output = []
    for (case_id, transport), (row, source) in sorted(indexed.items()):
        case = cases[case_id]
        if [t["expected"] for t in row["turns"]] != case["turns"]:
            raise ValueError("Conversation inputs differ: " + case_id)
        key = f"conversation/{transport}/{case_id}"
        output.append(single_item("conversation-scene", key + "/state", case, row, source,
                                  query="Evaluate captured scenario state: " + case_id,
                                  answer=json.dumps({"turns": len(row["turns"]), "recorded_passed": row["passed"]})))
        sessions = {}
        for turn in row["turns"]:
            result = turn["result"]
            if result is None or result.get("cached") or turn.get("error"):
                continue
            owner = turn["expected"].get("actor", "employee-e001")
            session_key = (result["session_id"], owner)
            sessions.setdefault(session_key, []).extend([
                Turn(role="user", content=turn["expected"]["query"], user_id=owner),
                Turn(role="assistant", content=result["answer"], user_id=owner,
                     tools_called=execution_calls(result),
                     retrieval_context=[x["content"] for x in result.get("knowledge_evidence", [])] or None)])
        for index, ((session_id, owner), turns) in enumerate(sessions.items()):
            subkey = key + f"/session-{index}"
            conversation = ConversationalTestCase(name=subkey, turns=turns,
                chatbot_role="企业助手；每轮重新授权，工单仅本人；缺参先询问；取消不创建；最多8轮历史。",
                expected_outcome=json.dumps(case["turns"], ensure_ascii=False),
                metadata={"record": row, "expected": case, "session_id": session_id, "actor": owner, "source": source})
            output.append(EvaluationItem(subkey, "conversation", conversation, source))
    return output
