from textwrap import dedent

import pytest

from app.document_loader import load_markdown_document, load_knowledge_documents


def write_document(tmp_path, content, file_name="test.md"):
    document_path = tmp_path / file_name
    document_path.write_text(content, encoding="utf-8")
    return document_path


def test_load_markdown_document_parses_content_and_metadata(tmp_path):
    content = dedent(
        """\
        ---
        doc_id: KB-TEST-001
        title: 测试文档
        category: test
        ---
        # 测试标题

        这是正文。
        """
    )

    document_path = write_document(tmp_path, content)
    doc = load_markdown_document(document_path)
    assert doc.id == "KB-TEST-001"
    assert doc.metadata["title"] == "测试文档"
    assert doc.metadata["category"] == "test"
    assert doc.metadata["source"] == "test.md"
    assert doc.page_content.startswith("# 测试标题")
    assert "doc_id:" not in doc.page_content


def test_load_markdown_document_rejects_missing_front_matter(tmp_path):
    content = """# 普通Markdown"""
    document_path = write_document(tmp_path, content)
    with pytest.raises(ValueError, match="文档缺少YAML Front Matter"):
        load_markdown_document(document_path)


def test_load_markdown_document_rejects_unclosed_front_matter(tmp_path):
    content = dedent(
        """\
        ---
        doc_id: KB-TEST-001
        title: 测试文档
        category: test
        """
    )
    document_path = write_document(tmp_path, content)
    with pytest.raises(ValueError, match="YAML Front Matter格式错误"):
        load_markdown_document(document_path)


def test_load_markdown_document_rejects_missing_required_field(tmp_path):
    content = dedent(
        """\
        ---
        doc_id: KB-TEST-001
        category: test
        ---
        # 测试标题

        这是正文。
        """
    )
    document_path = write_document(tmp_path, content)
    with pytest.raises(ValueError, match="缺少必要字段：title"):
        load_markdown_document(document_path)


def test_load_knowledge_documents_loads_markdown_files(tmp_path):
    content_a = dedent(
        """\
        ---
        doc_id: KB-A
        title: 文档A
        category: test
        ---
        # 文档A
        """
    )
    content_b = dedent(
        """\
        ---
        doc_id: KB-B
        title: 文档B
        category: test
        ---
        # 文档B
        """
    )
    write_document(tmp_path, content_b, "b.md")
    write_document(tmp_path, content_a, "a.md")
    documents = load_knowledge_documents(tmp_path)

    assert len(documents) == 2
    assert [document.id for document in documents] == ['KB-A', 'KB-B']


def test_load_knowledge_documents_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="知识库目录中没有Markdown文档"):
        load_knowledge_documents(tmp_path)


def test_load_knowledge_documents_rejects_duplicate_doc_id(tmp_path):
    content_a = dedent(
        """\
        ---
        doc_id: KB-DUPLICATE
        title: 文档A
        category: test
        ---
        # 文档A
        """
    )
    content_b = dedent(
        """\
        ---
        doc_id: KB-DUPLICATE
        title: 文档B
        category: test
        ---
        # 文档B
        """
    )
    write_document(tmp_path, content_b, "b.md")
    write_document(tmp_path, content_a, "a.md")
    with pytest.raises(ValueError, match="发现重复doc_id：KB-DUPLICATE"):
        load_knowledge_documents(tmp_path)


def test_load_knowledge_documents_rejects_missing_directory(tmp_path):
    missing_directory = tmp_path / "missing"

    with pytest.raises(ValueError, match="知识库目录不存在"):
        load_knowledge_documents(missing_directory)