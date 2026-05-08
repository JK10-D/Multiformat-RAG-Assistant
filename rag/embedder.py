# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

def index_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [f"{c['source']}_{c['chunk_id']}" for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"{len(chunks)} chunks indexés.")

def search(query, top_k=4):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks_found = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks_found.append({
            "text": doc,
            "source": meta["source"],
            "score": round(1 - dist, 3)
        })
    return chunks_found

