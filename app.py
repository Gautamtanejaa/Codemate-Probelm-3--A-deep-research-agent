"""Streamlit app for KnowledgeSeeker Agent.

Features:
- Upload multiple PDFs/TXTs for ingestion
- Build/load FAISS index
- Ask queries and follow-ups with multi-step reasoning
- View answer, reasoning steps, and sources
- Export latest result to Markdown or PDF

Run:
  streamlit run app.py
"""
from __future__ import annotations

import os
import io
import json
from typing import List, Dict, Any

import streamlit as st

from ingest import Ingestor, DEFAULT_DATA_DIR
from embeddings import Embedder
from query import KnowledgeSeeker
from export import export_markdown, export_pdf

st.set_page_config(page_title="KnowledgeSeeker", layout="wide")

# Ensure data dir exists
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

if "engine" not in st.session_state:
    st.session_state.engine = "t5"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
if "ks" not in st.session_state:
    st.session_state.ks = KnowledgeSeeker(data_dir=DEFAULT_DATA_DIR, engine=st.session_state.engine, llm_model=st.session_state.llm_model)
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temp folder under sample_docs and return path."""
    docs_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_path = os.path.join(docs_dir, uploaded_file.name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


st.title("🔎 KnowledgeSeeker (Local)")

with st.sidebar:
    st.header("Documents")
    st.subheader("Engine")
    sel_engine = st.selectbox("Reasoning engine", options=["t5", "llm"], index=0 if st.session_state.engine=="t5" else 1)
    sel_model = st.text_input("LLM model (for engine=llm)", value=st.session_state.llm_model)
    if st.button("Apply Engine"):
        changed = (sel_engine != st.session_state.engine) or (sel_model != st.session_state.llm_model)
        st.session_state.engine = sel_engine
        st.session_state.llm_model = sel_model
        if changed:
            st.session_state.ks = KnowledgeSeeker(data_dir=DEFAULT_DATA_DIR, engine=st.session_state.engine, llm_model=st.session_state.llm_model)
            st.success(f"Engine set to {st.session_state.engine} ({st.session_state.llm_model if st.session_state.engine=='llm' else 'Flan-T5-small'})")

    uploads = st.file_uploader("Upload PDFs or TXTs", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Ingest Uploaded Files", type="primary"):
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            paths: List[str] = []
            for uf in uploads:
                try:
                    p = save_uploaded_file(uf)
                    paths.append(p)
                except Exception as e:
                    st.error(f"Failed to save {uf.name}: {e}")
            if paths:
                ing = Ingestor(data_dir=DEFAULT_DATA_DIR)
                stats = ing.process_files(paths)
                st.success(f"Ingested {stats['documents']} documents into {stats['chunks']} chunks.")

    st.divider()
    st.header("Index")
    if st.button("Build/Refresh Index"):
        try:
            emb = Embedder(data_dir=DEFAULT_DATA_DIR)
            stats = emb.build_index()
            st.success(f"Index built with {stats['chunks_indexed']} chunks (dim={stats['dim']}).")
        except Exception as e:
            st.error(f"Index build failed: {e}")

    if st.button("Clear Conversation"):
        st.session_state.ks.clear_session()
        st.success("Conversation cleared.")

st.subheader("Ask a Question")
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input("Enter your query", value="Summarize the key findings")
with col2:
    top_k = st.number_input("Top K", value=5, min_value=1, max_value=20)

ask_col, follow_col = st.columns(2)
with ask_col:
    if st.button("Ask", use_container_width=True):
        try:
            res = st.session_state.ks.ask(user_query, top_k=int(top_k))
            st.session_state.latest = res
        except Exception as e:
            st.error(f"Ask failed: {e}")
with follow_col:
    if st.button("Follow-up: Explain in simpler terms", use_container_width=True):
        try:
            res = st.session_state.ks.follow_up("Explain in simpler terms", top_k=int(top_k))
            st.session_state.latest = res
        except Exception as e:
            st.error(f"Follow-up failed: {e}")

st.divider()

latest: Dict[str, Any] | None = st.session_state.get("latest") or st.session_state.ks.latest()
if latest:
    st.subheader("Answer")
    st.write(latest.get("answer", ""))

    st.subheader("Reasoning Steps")
    for i, s in enumerate(latest.get("steps", []), 1):
        st.markdown(f"{i}. {s}")

    st.subheader("Sources")
    hits: List[Dict[str, Any]] = latest.get("hits", [])
    if hits:
        for h in hits:
            meta = h.get("doc_meta", {})
            st.caption(f"- {meta.get('filename','')} (chunk {h.get('chunk_index',-1)}) score={h.get('score',0):.4f}")
    else:
        st.caption("No sources available")

    st.divider()
    st.subheader("Export")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Export Markdown"):
            path = export_markdown({**latest, "title": "KnowledgeSeeker Report"})
            st.success(f"Exported Markdown to: {path}")
    with c2:
        if st.button("Export PDF"):
            path = export_pdf({**latest, "title": "KnowledgeSeeker Report"})
            st.success(f"Exported PDF to: {path}")
else:
    st.info("Build the index and ask a question to see results here.")
