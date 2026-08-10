from langchain_core.documents import Document
import pytest

from app.document_splitter import split_document, split_documents, split_document_recursive, split_documents_recursive


def test_split_document_creates_overlapping_chunks():
    document = Document(
        id="KB-TEST-001",
        page_content= "abcdefghij",
        metadata={
            "doc_id": "KB-TEST-001",
            "title": "测试文档",
            "category": "test",
        }
    )

    chunks = split_document(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.page_content for chunk in chunks] == ["abcd", "defg", "ghij"]


def test_split_document_adds_chunk_metadata():
    document = Document(
        id="KB-TEST-001",
        page_content="abcdefghij",
        metadata={
            "doc_id": "KB-TEST-001",
            "title": "测试文档",
            "category": "test",
            "source": "test.md",
        }
    )

    chunks = split_document(document, chunk_size=4, chunk_overlap=1)
    assert chunks[0].id == "KB-TEST-001-CHUNK-000"
    assert chunks[1].id == "KB-TEST-001-CHUNK-001"
    assert chunks[2].id == "KB-TEST-001-CHUNK-002"

    assert "chunk_id" not in document.metadata
    assert "chunk_index" not in document.metadata


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "expected_message"),
    [
        (0, 0, "chunk_size必须大于0"),
        (-1, 0, "chunk_size必须大于0"),
        (4, -1, "chunk_overlap不能小于0"),
        (4, 4, "chunk_overlap必须小于chunk_size"),
        (4, 5, "chunk_overlap必须小于chunk_size"),
    ],
)
def test_split_document_rejects_invalid_parameters(chunk_size, chunk_overlap, expected_message):
    document = Document(
        id="KB_TEST_001",
        page_content="abcdefghij",
        metadata={"doc_id": "KB-TEST-001"},
    )

    with pytest.raises(ValueError, match=expected_message):
        split_document(document, chunk_size, chunk_overlap)


@pytest.mark.parametrize("empty_content", ["", " ", "\t", "\n"])
def test_split_document_rejects_empty_content(empty_content):
    document = Document(
        id="KB-TEST-001",
        page_content=empty_content,
        metadata={"doc_id": "KB-TEST-001"},
    )

    with pytest.raises(ValueError, match="文档内容不能为空"):
        split_document(document)


def test_split_document_rejects_missing_doc_id():
    document = Document(
        page_content="有效正文",
        metadata={"title": "测试文档"},
    )

    with pytest.raises(ValueError, match="文档缺少doc_id"):
        split_document(document)


def test_split_documents_returns_flat_chunk_list():
    documents = [
        Document(
            id="KB-A",
            page_content="abcdef",
            metadata={"doc_id": "KB-A"},
        ),
        Document(
            id="KB-B",
            page_content="xyz",
            metadata={"doc_id": "KB-B"},
        )
    ]

    chunks = split_documents(documents, chunk_size=4, chunk_overlap=1)

    assert [chunk.id for chunk in chunks] == [
        "KB-A-CHUNK-000",
        "KB-A-CHUNK-001",
        "KB-B-CHUNK-000",
    ]
    assert [chunk.page_content for chunk in chunks] == [
        "abcd",
        "def",
        "xyz",
    ]
    assert [chunk.metadata["doc_id"] for chunk in chunks] == [
        "KB-A",
        "KB-A",
        "KB-B",
    ]


def test_split_document_recursive_prefers_paragraph_boundaries():
    document = Document(
        id="KB-TEST-001",
        page_content=(
            "# 标题\n\n"
            "第一段介绍产品用途。第一段包含补充信息。\n\n"
            "## 登录规则\n\n"
            "连续失败五次后，账号锁定三十分钟。"
        ),
        metadata={"doc_id": "KB-TEST-001"},
    )

    chunks = split_document_recursive(document, chunk_size=30, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].page_content.startswith("# 标题")
    assert chunks[1].page_content.startswith("## 登录规则")
    assert all(len(chunk.page_content) <= 30 for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [
        "KB-TEST-001-CHUNK-000",
        "KB-TEST-001-CHUNK-001",
    ]


def test_split_documents_recursive_returns_flat_chunk_list():
    documents = [
        Document(
            id="KB-A",
            page_content="# 文档A\n\n这是文档A的正文。",
            metadata={"doc_id": "KB-A"},
        ),
        Document(
            id="KB-B",
            page_content="# 文档B\n\n这是文档B的正文。",
            metadata={"doc_id": "KB-B"},
        ),
    ]

    chunks = split_documents_recursive(
        documents,
        chunk_size=30,
        chunk_overlap=0,
    )

    assert len(chunks) == 2
    assert [chunk.id for chunk in chunks] == [
        "KB-A-CHUNK-000",
        "KB-B-CHUNK-000",
    ]
    assert[chunk.metadata["doc_id"] for chunk in chunks] == [
        "KB-A",
        "KB-B",
    ]