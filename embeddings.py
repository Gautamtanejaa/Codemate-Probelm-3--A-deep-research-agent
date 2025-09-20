"""Embeddings and FAISS index management for KnowledgeSeeker.

- Loads SentenceTransformer locally (all-MiniLM-L6-v2)
- Builds and persists a FAISS index for chunk retrieval
- Provides `search_index(query, top_k=5)`

Artifacts (under ./.data by default):
- chunks.jsonl: produced by ingest.py
- index.faiss: FAISS index file
- index_map.json: maps FAISS ids -> chunk metadata
- documents.jsonl: document metadata (for enrichment)
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")


class Embedder:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_name = model_name
        # Load SentenceTransformer locally (will download once if not cached)
        self.model = SentenceTransformer(model_name)
        self.index = None  # type: ignore
        self.id_map: List[Dict[str, Any]] = []
        # Resolve file paths relative to data_dir
        self.chunks_file = os.path.join(self.data_dir, "chunks.jsonl")
        self.docs_file = os.path.join(self.data_dir, "documents.jsonl")
        self.index_file = os.path.join(self.data_dir, "index.faiss")
        self.index_map_file = os.path.join(self.data_dir, "index_map.json")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}. Run ingest.py first.")
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunks.append(json.loads(line))
        return chunks

    def _load_docs_meta(self) -> Dict[str, Dict[str, Any]]:
        meta: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.docs_file):
            with open(self.docs_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    meta[rec["doc_id"]] = rec
        return meta

    def build_index(self) -> Dict[str, int]:
        """Build FAISS index from chunks.jsonl and persist to disk."""
        chunks = self._load_chunks()
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, self.index_file)
        # Save id map for reverse lookup
        id_map = [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "chunk_index": c["chunk_index"]} for c in chunks]
        with open(self.index_map_file, "w", encoding="utf-8") as f:
            json.dump(id_map, f)
        return {"chunks_indexed": len(chunks), "dim": dim}

    def _ensure_loaded(self) -> None:
        if self.index is None:
            if not os.path.exists(self.index_file) or not os.path.exists(self.index_map_file):
                raise FileNotFoundError("Missing index files. Build the index first using build_index().")
            self.index = faiss.read_index(self.index_file)
            with open(self.index_map_file, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)

    def search_index(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index using the query string.

        Returns list of hits with: score, text, doc metadata, and chunk refs.
        """
        self._ensure_loaded()
        assert self.index is not None
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q_emb, top_k)
        docs_meta = self._load_docs_meta()
        # Load chunks text lazily using another pass (we need chunk text by id)
        text_by_chunk: Dict[str, str] = {}
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text_by_chunk[rec["chunk_id"]] = rec["text"]
        hits: List[Dict[str, Any]] = []
        for pos, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            meta = self.id_map[idx]
            chunk_id = meta["chunk_id"]
            doc_id = meta["doc_id"]
            hit = {
                "rank": pos + 1,
                "score": float(D[0][pos]),
                "chunk_id": chunk_id,
                "chunk_index": meta.get("chunk_index", -1),
                "doc_id": doc_id,
                "text": text_by_chunk.get(chunk_id, ""),
                "doc_meta": self._safe_doc_meta(docs_meta.get(doc_id, {})),
            }
            hits.append(hit)
        return hits

    @staticmethod
    def _safe_doc_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "filename": meta.get("filename", ""),
            "path": meta.get("path", ""),
            "n_pages": meta.get("n_pages", 0),
            "n_chunks": meta.get("n_chunks", 0),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the FAISS index")
    parser.add_argument("action", choices=["build", "search"]) 
    parser.add_argument("--query", default="test query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    emb = Embedder()
    if args.action == "build":
        stats = emb.build_index()
        print("Index built:", stats)
    else:
        hits = emb.search_index(args.query, top_k=args.top_k)
        for h in hits:
            print(f"#{h['rank']} score={h['score']:.4f} file={h['doc_meta']['filename']} chunk={h['chunk_index']}")
