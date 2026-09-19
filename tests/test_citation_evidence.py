import json

import pytest
from langchain_core.messages import AIMessage

from app.citation_evidence import bind_citation_evidence, validate_citation_checks
from app.llm_judge import LLMJudge


pytestmark = pytest.mark.unit


def test_source_binding_is_independent_of_chunk_order_and_overlapping_facts():
    answer = "查询接口为 GET /tickets/{ticket_id}。[来源：API/API-2]"
    blocks = ["[来源：API/API-3]\nGET /tickets/EXAMPLE", "[来源：API/API-2]\n创建接口。\nGET /tickets/{ticket_id}",
              "[来源：API/API-1]\nGET /tickets/{ticket_id}"]
    original = bind_citation_evidence(answer, "\n\n".join(blocks))
    reordered = bind_citation_evidence(answer, "\n\n".join(blocks[::-1]))
    assert original == reordered
    assert original[0]["content"] == "创建接口。\nGET /tickets/{ticket_id}"
    validate_citation_checks(original, [{"index": 0, "supported": True, "evidence_quote": "GET /tickets/{ticket_id}"}], 5)


@pytest.mark.parametrize("context,status", [
    ("[来源：OTHER/C]\n30分钟。", "missing"),
    ("[来源：FAQ/C]\n30分钟。\n[来源：FAQ/C]\n90分钟。", "ambiguous"),
])
def test_missing_or_duplicate_source_cannot_be_treated_as_supported(context, status):
    bindings = bind_citation_evidence("30分钟。[来源：FAQ/C]", context)
    assert bindings[0]["status"] == status and bindings[0]["content"] is None
    validate_citation_checks(bindings, [{"index": 0, "supported": False, "evidence_quote": ""}], 1)
    with pytest.raises(ValueError, match="没有唯一来源"):
        validate_citation_checks(bindings, [{"index": 0, "supported": True, "evidence_quote": "30分钟。"}], 5)


def test_quote_from_neighbour_chunk_is_rejected_even_if_present_in_full_context():
    bindings = bind_citation_evidence("重置链接有效10分钟。[来源：FAQ/A]",
                                      "[来源：FAQ/A]\n账号锁定30分钟。\n[来源：FAQ/B]\n重置链接有效10分钟。")
    with pytest.raises(ValueError, match="不属于绑定片段"):
        validate_citation_checks(bindings, [{"index": 0, "supported": True, "evidence_quote": "重置链接有效10分钟。"}], 5)


@pytest.mark.parametrize("checks,score", [
    ([], 5),
    ([{"index": 1, "supported": True, "evidence_quote": "30分钟。"}], 5),
    ([{"index": 0, "supported": False, "evidence_quote": ""}], 5),
    ([{"index": 0, "supported": True, "evidence_quote": ""}], 5),
])
def test_incomplete_or_inconsistent_audits_are_protocol_errors(checks, score):
    binding = bind_citation_evidence("30分钟。[来源：FAQ/C]", "[来源：FAQ/C]\n30分钟。")
    with pytest.raises(ValueError):
        validate_citation_checks(binding, checks, score)


def test_multiple_citations_and_crlf_headers_keep_exact_source_identity():
    binding = bind_citation_evidence("30分钟。[来源：FAQ/A][来源：FAQ/B]",
                                     "[来源：FAQ/A]\r\n30分钟。\r\n\r\n[来源：FAQ/B]\r\n30分钟。")
    assert len(binding) == 2 and all(row["status"] == "found" for row in binding)
    assert binding[0]["claim"] == binding[1]["claim"] == "30分钟。"
    assert binding[0]["content"] == "30分钟。"


def test_inline_citation_in_content_is_not_a_new_context_header():
    binding = bind_citation_evidence("30分钟。[来源：FAQ/A]",
                                     "[来源：FAQ/A]\n30分钟。文中提到[来源：OTHER/B]不代表切分。")
    assert "OTHER/B" in binding[0]["content"]


class Model:
    def __init__(self, checks, score=5):
        self.payload = {"correctness": 5, "completeness": 5, "faithfulness": 5,
                        "citation_support": score, "reason": "test", "citation_checks": checks}

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=json.dumps(self.payload))


def test_llm_judge_requires_auditable_quote_and_sends_exact_binding():
    model = Model([{"index": 0, "supported": True, "evidence_quote": "30分钟。"}])
    result = LLMJudge(model).evaluate("锁多久", "30分钟", "30分钟。[来源：FAQ/A]", "[来源：FAQ/A]\n30分钟。")
    assert result.passed and result.citation_checks[0].evidence_quote == "30分钟。"
    payload = model.messages[-1].content
    assert '"citation_evidence"' in payload and '"chunk_id": "A"' in payload


def test_llm_judge_does_not_accept_blanket_full_scores_without_citation_audit():
    with pytest.raises(ValueError, match="条目缺失"):
        LLMJudge(Model([])).evaluate("锁多久", "30分钟", "30分钟。[来源：FAQ/A]", "[来源：FAQ/A]\n30分钟。")


def test_valid_source_id_does_not_automatically_pass_semantic_support():
    model = Model([{"index": 0, "supported": False, "evidence_quote": "账号锁定30分钟。"}], 2)
    result = LLMJudge(model).evaluate("链接多久有效", "10分钟", "链接有效10分钟。[来源：FAQ/A]",
                                      "[来源：FAQ/A]\n账号锁定30分钟。")
    assert not result.passed and result.citation_support == 2


def test_chinese_paragraph_layout_does_not_invalidate_an_unchanged_quote():
    binding = bind_citation_evidence("5次错误后锁30分钟。[来源：FAQ/A]",
                                     "[来源：FAQ/A]\n输错 5 次后锁定 30 分钟。\n\n账号会在 30 分钟后解锁。")
    validate_citation_checks(binding, [{"index": 0, "supported": True,
                             "evidence_quote": "输错 5 次后锁定 30 分钟。账号会在 30 分钟后解锁。"}], 5)


@pytest.mark.parametrize("quote", ["有效期为90分钟。", "有效期为1 0分钟。", "passwordreset", "有效期为10分钟！"])
def test_layout_tolerance_does_not_change_numbers_words_or_punctuation(quote):
    binding = bind_citation_evidence("有效10分钟。[来源：FAQ/A]", "[来源：FAQ/A]\n有效期为10分钟。password reset")
    with pytest.raises(ValueError, match="不属于绑定片段"):
        validate_citation_checks(binding, [{"index": 0, "supported": True, "evidence_quote": quote}], 5)


def test_invalid_audit_preserves_model_output_and_validation_reason():
    from app.tracing import capture_trace
    model = Model([{"index": 0, "supported": True, "evidence_quote": "90分钟。"}])
    with capture_trace() as events:
        with pytest.raises(ValueError):
            LLMJudge(model).evaluate("锁多久", "30分钟", "30分钟。[来源：FAQ/A]", "[来源：FAQ/A]\n30分钟。")
    assert any(e["event"] == "raw_response" and "90" in e["data"]["content"] for e in events)
    assert any(e["event"] == "invalid_response" and "不属于绑定片段" in e["data"]["detail"] for e in events)


@pytest.mark.parametrize("quote,content", [
    ("P2 工单：2 小时。\\nP3 工单：1 个工作日。", "- P2 工单：2 小时。\n- P3 工单：1 个工作日。"),
    ("通过Web端上报。...无法访问时联系IT服务台。", "通过Web端上报。\n补充信息清单。\n无法访问时联系IT服务台。"),
    ("请勿发送密码。IT不会索要密码。", "请勿发送密码。\n## 他人能代办吗\n不可以。IT不会索要密码。"),
])
def test_ordered_literal_excerpts_may_skip_source_layout_and_intervening_text(quote, content):
    from app.citation_evidence import quote_belongs_to_content
    assert quote_belongs_to_content(quote, content)


@pytest.mark.parametrize("quote,content", [
    ("有效期为90分钟。...只能使用一次。", "有效期为10分钟。只能使用一次。"),
    ("只能使用一次。...有效期为10分钟。", "有效期为10分钟。只能使用一次。"),
    ("有效期为1...0分钟。", "有效期为10分钟。"),
    ("有效期为10分钟。...!", "有效期为10分钟。"),
    ("password\\nreset", "passwordreset"),
    ("...", "有效期为10分钟。"),
])
def test_excerpt_matching_cannot_change_facts_order_or_word_boundaries(quote, content):
    from app.citation_evidence import quote_belongs_to_content
    assert not quote_belongs_to_content(quote, content)
