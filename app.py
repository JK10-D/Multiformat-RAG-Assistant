# -*- coding: utf-8 -*-

import sys
import streamlit as st
sys.path.insert(0, r"C:\Users\user\Documents\IA\rag_assistant")

from rag.loader import load_document
from rag.chunker import create_chunks
from rag.embedder import index_chunks, search
from rag.generator import generate

st.set_page_config(page_title="Assistant RAG", layout="wide")
st.title("Assistant RAG multiformat")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Ajouter des documents")
    url_input = st.text_input("Coller une URL")
    if st.button("Indexer") and url_input:
        with st.spinner("Indexation en cours..."):
            texte = load_document(url_input)
            chunks = create_chunks(texte, url_input)
            index_chunks(chunks)
        st.success(f"Document indexé — {len(chunks)} chunks")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources utilisées"):
                for s in msg["sources"]:
                    st.caption(f"Score: {s['score']} | {s['text'][:200]}")

if question := st.chat_input("Pose ta question..."):
    resultats = search(question)
    reponse = generate(question, resultats, st.session_state.history)
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({
        "role": "assistant",
        "content": reponse,
        "sources": resultats
    })
    st.rerun()