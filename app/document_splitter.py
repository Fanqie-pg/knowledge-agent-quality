from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document(document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap不能小于0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap必须小于chunk_size")
    # strip()是为了识别" "、\t、\n的场景
    if not document.page_content.strip():
        raise ValueError("文档内容不能为空")
    if not document.metadata.get("doc_id"):
        raise ValueError("文档缺少doc_id")
    step = chunk_size - chunk_overlap
    text = document.page_content
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        chunk_metadata = document.metadata.copy()
        chunk_id = f"{document.metadata['doc_id']}-CHUNK-{chunk_index:03d}"
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["chunk_index"] = chunk_index
        end = start + chunk_size
        chunk_text = text[start: end]
        chunk = Document(
            id=chunk_id,
            page_content=chunk_text,
            metadata=chunk_metadata
        )
        chunks.append(chunk)
        if end >= len(text):
            break
        start += step
        chunk_index += 1

    return chunks


def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    chunks = []

    for document in documents:
        document_chunks = split_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks.extend(document_chunks)

    return chunks


def split_document_recursive(document:Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap不能小于0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap必须小于chunk_size")
    if not document.page_content.strip():
        raise ValueError("文档内容不能为空")
    if not document.metadata.get("doc_id"):
        raise ValueError("文档缺少doc_id")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = []
    chunk_texts = splitter.split_text(document.page_content)
    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk_metadata = document.metadata.copy()
        chunk_id = f"{document.metadata['doc_id']}-CHUNK-{chunk_index:03d}"
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["chunk_index"] = chunk_index

        chunk = Document(
            id=chunk_id,
            page_content=chunk_text,
            metadata=chunk_metadata,
        )
        chunks.append(chunk)
    return chunks


def split_documents_recursive(documents: list[Document], chunk_size: int = 500,chunk_overlap: int = 50) -> list[Document]:
    chunks = []

    for document in documents:
        document_chunk = split_document_recursive(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(document_chunk)
    return chunks