"""Bind answer citations to exact input chunks before semantic judging."""
import re

from app.citation_validator import CITATION_PATTERN


CITATION_AUDIT_VERSION = "1.2-exact-source-ordered-excerpts"
SOURCE_HEADER = re.compile(r"^\[来源：([^/\]\r\n]+)/([^\]\r\n]+)\][ \t]*\r?$", re.MULTILINE)


def normalize_quote_layout(text: str) -> str:
    """Ignore layout whitespace around CJK, preserving ASCII word/number boundaries."""
    normalized = re.sub(r"\s+", " ", text).strip()
    # A Chinese paragraph break need not appear in a short excerpt. Keep e.g.
    # '1 0' distinct from '10' and 'password reset' distinct from 'passwordreset'.
    cjk = r"[\u3400-\u9fff\u3001-\u303f，。！？；：、（）《》“”‘’]"
    return re.sub(rf"(?<={cjk}) | (?={cjk})", "", normalized)


def quote_belongs_to_content(quote: str, content: str) -> bool:
    if quote in content or normalize_quote_layout(quote) in normalize_quote_layout(content):
        return True
    # Model excerpts may join source sentences/rows or use literal JSON escapes.
    # Check every literal span, in order, inside this one bound chunk only.
    if re.search(r"\d\s*(?:\.{3,}|…+|\\n|\r?\n)\s*\d", quote):
        return False
    parts = [normalize_quote_layout(part) for part in re.split(r"\.{3,}|…+|\\n|\r?\n|(?<=[。！？])", quote) if part.strip()]
    parts = [part for part in parts if part not in {"```", "~~~"}]
    if not parts or any(not re.search(r"\w", part) for part in parts):
        return False
    normalized = normalize_quote_layout(content)
    offset = 0
    for part in parts:
        pattern = (r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", part) else "") + re.escape(part)
        pattern += r"(?![A-Za-z0-9_])" if re.search(r"[A-Za-z0-9_]$", part) else ""
        match = re.search(pattern, normalized[offset:])
        if match is None:
            return False
        offset += match.end()
    return True


def bind_citation_evidence(answer: str, context: str) -> list[dict]:
    """Do not infer chunk order or silently resolve duplicate source IDs."""
    headers = list(SOURCE_HEADER.finditer(context))
    sources = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(context)
        key = (header.group(1), header.group(2))
        sources.setdefault(key, []).append(context[header.end():end].strip())

    bindings = []
    previous_end, previous_claim = 0, ""
    for index, citation in enumerate(CITATION_PATTERN.finditer(answer)):
        preceding = answer[previous_end:citation.start()].strip()
        # A paragraph/sentence is a useful focus; full-answer coverage remains the Judge's job.
        parts = [part.strip() for part in re.split(r"\n\s*\n|(?<=[。！？])", preceding) if part.strip()]
        claim = parts[-1] if parts else previous_claim
        chunks = sources.get((citation.group(1), citation.group(2)), [])
        bindings.append({"index": index, "doc_id": citation.group(1), "chunk_id": citation.group(2),
                         "claim": claim, "status": "found" if len(chunks) == 1 else "missing" if not chunks else "ambiguous",
                         "content": chunks[0] if len(chunks) == 1 else None})
        previous_end, previous_claim = citation.end(), claim
    return bindings


def validate_citation_checks(bindings: list[dict], checks: list[dict], citation_score: int):
    """Validate evidence provenance; existence alone never proves semantic support."""
    if len(checks) != len(bindings) or [check["index"] for check in checks] != list(range(len(bindings))):
        raise ValueError("Judge引用核验条目缺失、重复或顺序不合法")
    for binding, check in zip(bindings, checks):
        if check["supported"]:
            if binding["status"] != "found" or not check["evidence_quote"].strip():
                raise ValueError("Judge声称引用受支持但没有唯一来源或证据原文")
            if not quote_belongs_to_content(check["evidence_quote"], binding["content"]):
                raise ValueError("Judge引用证据原文不属于绑定片段")
        elif check["evidence_quote"] and (binding["status"] != "found" or not quote_belongs_to_content(check["evidence_quote"], binding["content"])):
            raise ValueError("Judge引用证据原文不属于绑定片段")
    if citation_score >= 4 and any(not check["supported"] for check in checks):
        raise ValueError("Judge引用评分与逐条核验结论矛盾")
