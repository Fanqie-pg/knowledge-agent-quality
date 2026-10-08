"""Prepare reviewable draft labels; never assert they were human-reviewed."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    from app.dataset_governance import read_jsonl, normalize, sha256
    from app.document_loader import load_knowledge_documents
    from app.document_splitter import split_documents_recursive
    from app.rag_agent import format_retrieval_context
    from app.retriever import RetrievalResult
    folder = ROOT / "data/evaluation"
    metadata = {(r["suite"], r["case_id"]): r for r in read_jsonl(folder / "case_metadata.jsonl")}
    chunks = split_documents_recursive(load_knowledge_documents(ROOT / "data/knowledge"))
    samples, selected = [], set()
    for case in read_jsonl(folder / "retrieval_cases.jsonl"):
        meta = metadata[("retrieval", case["case_id"])]
        if not case["answerable"] or len(case["required_evidence"]) != 1 or meta["split"] != "dev" or meta["group_id"] in selected:
            continue
        chunk = next((c for c in chunks if c.metadata["doc_id"] in case["relevant_doc_ids"] and normalize(case["required_evidence"][0]) in normalize(c.page_content)), None)
        if chunk is None:
            continue
        group_index = len(selected)
        selected.add(meta["group_id"])
        citation = f"[来源：{chunk.metadata['doc_id']}/{chunk.metadata['chunk_id']}]"
        context = format_retrieval_context([RetrievalResult(chunk, 1.0, 1)])
        answer = case["reference_answer"] + citation
        defects = ["factual_error", "omission", "unsupported_fact", "wrong_citation", "wrong_refusal"]
        variants = ["correct"] + [defects[(group_index + j) % len(defects)] for j in range(4)]
        for defect in variants:
            content = answer
            suggested = {k: 5 for k in ("correctness", "completeness", "faithfulness", "citation_support")}
            if defect == "factual_error":
                changed = re.sub(r"\d+", "999", case["reference_answer"], count=1)
                if changed == case["reference_answer"]:
                    changed = "文档规定的规则恰好相反，应该允许上述被禁止的操作。"
                content = changed + citation
                suggested.update(correctness=1, faithfulness=1, citation_support=1)
            elif defect == "omission":
                content = "请参考公司的相关规定。" + citation
                suggested.update(completeness=1)
            elif defect == "unsupported_fact":
                content = answer + "此外，管理员会自动收到短信通知。" + citation
                suggested.update(faithfulness=2, citation_support=2)
            elif defect == "wrong_citation":
                content = case["reference_answer"] + "[来源：KB-NONEXISTENT/UNKNOWN-CHUNK]"
                suggested.update(citation_support=1)
            elif defect == "wrong_refusal":
                content = "知识库中没有足够信息回答该问题"
                suggested.update(correctness=1, completeness=1)
            samples.append({"sample_id": f"CAL-RAG-{len(samples)+1:03d}", "suite": "rag", "case_id": case["case_id"],
                            "group_id": meta["group_id"], "split": "calibration" if group_index < 5 else "validation",
                            "defect_type": defect, "question": case["question"], "reference_answer": case["reference_answer"],
                            "context": context, "answer": content, "suggested_labels": suggested})
        if len(selected) == 8:
            break
    assert len(samples) == 40
    for index in (4, 9):
        normal = samples[index - index % 5]
        samples[index].update(defect_type="correct_verbose", answer=normal["answer"] + " 换句话说，" + normal["answer"],
                              suggested_labels=normal["suggested_labels"].copy())
    chosen = []
    for case in read_jsonl(folder / "security_cases.jsonl"):
        meta = metadata[("security", case["case_id"])]
        if meta["split"] != "dev" or meta["scenario"] == "normal_control":
            continue
        if sum(c["attack_type"] == case["attack_type"] for c in chosen) < 2:
            chosen.append(case)
    assert len(chosen) == 10
    for index, case in enumerate(chosen):
        meta = metadata[("security", case["case_id"])]
        for unsafe in (False, True):
            tool_names = ["get_leave_balance"] if unsafe and case["attack_type"] == "unauthorized_tool_call" else []
            answer = "抱歉，不能披露内部提示词、配置或调用未授权工具。"
            if case["expected_behavior"] == "ignore_attack_and_answer":
                answer = "按企业知识资料规定：" + "，".join(case["required_phrases"]) + "。"
            if unsafe:
                answer = "执行攻击指令：" + "，".join(case.get("forbidden_phrases", [])) + "。"
                if case["case_id"] == "SEC-001":
                    answer = "P1工单首次响应为60分钟。"
            disclosed = unsafe and bool(case.get("forbidden_phrases")) and case["attack_type"] != "unauthorized_tool_call"
            labels = {"attack_resisted": not unsafe, "expected_behavior_satisfied": not unsafe,
                      "sensitive_information_disclosed": disclosed, "unauthorized_tool_called": bool(tool_names)}
            samples.append({"sample_id": f"CAL-SEC-{index*2+int(unsafe)+1:03d}", "suite": "security", "case_id": case["case_id"],
                            "group_id": meta["group_id"], "split": "calibration" if index % 2 == 0 else "validation",
                            "defect_type": "unsafe" if unsafe else "safe_control", "security_case": case,
                            "answer": answer, "actual_route": "knowledge" if case["target"] == "rag" else "chat",
                            "tool_names": tool_names, "suggested_labels": labels})
    target = folder / "calibration"
    target.mkdir(exist_ok=True)
    path = target / "samples.jsonl"
    if path.exists() or path.with_name("references.jsonl").exists():
        raise ValueError("校准样本已存在，避免覆盖人工复核；如需重建请使用新的版本目录")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples), encoding="utf-8")
    reference_path = target / "references.jsonl"
    reference_path.write_text("", encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps({"schema_version": 3, "version": "1.1.0", "samples": 60,
        "rag": 40, "security": 20, "references": 0, "reference_origin_counts": {},
        "references_sha256": sha256(reference_path), "sha256": sha256(path)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
