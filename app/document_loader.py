from pathlib import Path

import yaml
from langchain_core.documents import Document


def load_markdown_document(path: str | Path) -> Document:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError("文档缺少YAML Front Matter")

    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("YAML Front Matter格式错误")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError("YAML Front Matter必须是对象")

    required_fields = ["doc_id", "title", "category"]
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"缺少必要字段：{field}")

    metadata["source"] = file_path.name
    return Document(
        id=metadata["doc_id"],
        page_content=parts[2].strip(),
        metadata=metadata,
    )


def load_knowledge_documents(directory: str | Path) -> list[Document]:
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError("知识库目录不存在")

    documents = []
    doc_ids = set()
    for file_path in sorted(directory_path.glob("*.md")):
        document = load_markdown_document(file_path)
        if document.id in doc_ids:
            raise ValueError(f"发现重复doc_id：{document.id}")
        doc_ids.add(document.id)
        documents.append(document)
    if not documents:
        raise ValueError("知识库目录中没有Markdown文档")
    return documents