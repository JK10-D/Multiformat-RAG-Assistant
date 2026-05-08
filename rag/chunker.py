# -*- coding: utf-8 -*-


from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(text, source_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    raw_chunks = splitter.split_text(text)
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            "text": chunk,
            "source": source_name,
            "chunk_id": i
        })
    return chunks