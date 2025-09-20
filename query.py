"""Query handling, multi-step reasoning, and answer synthesis for KnowledgeSeeker.

Provides a `KnowledgeSeeker` class that coordinates retrieval via FAISS,
summarization, and conversation state management so follow-up questions
reuse context. Works fully locally.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

from embeddings import Embedder, DEFAULT_DATA_DIR
from summarize import Summarizer
from reasoner import ReasonerLLM


def _ensure_data_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class KnowledgeSeeker:
    """Coordinator for retrieval-augmented answering with conversation memory."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, engine: str = "t5", llm_model: str | None = None) -> None:
        self.data_dir = data_dir
        _ensure_data_dir(self.data_dir)
        self.embedder = Embedder(data_dir=self.data_dir)
        self.summarizer = Summarizer()
        self.engine = engine  # 't5' or 'llm'
        self.reasoner = ReasonerLLM(model_name=llm_model) if engine == "llm" else None
        self.session_file = os.path.join(self.data_dir, "session.json")
        self.session: Dict[str, Any] = self._load_session()

    # ---- Session management
    def _load_session(self) -> Dict[str, Any]:
        if os.path.exists(self.session_file):
            with open(self.session_file, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return {"history": []}
        return {"history": []}

    def _save_session(self) -> None:
        _ensure_data_dir(self.data_dir)
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.session, f, ensure_ascii=False, indent=2)

    def clear_session(self) -> None:
        self.session = {"history": []}
        self._save_session()

    # ---- Retrieval and reasoning
    def _decompose_query(self, query: str) -> List[str]:
        """Very light heuristic decomposition: split by punctuation and conjunctions."""
        parts = []
        for seg in query.replace("?", ".").split("."):
            seg = seg.strip()
            if not seg:
                continue
            # Further split on 'and', 'then'
            for sub in seg.replace(",", ";").split(";"):
                sub = sub.strip()
                if sub:
                    parts.append(sub)
        return parts[:4] or [query]

    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a new query with retrieval + synthesis and store in history."""
        subqueries = self._decompose_query(query)
        # Retrieve per subquery and merge hits (simple union by chunk_id keeping best score)
        hits_by_chunk: Dict[str, Dict[str, Any]] = {}
        for sq in subqueries:
            hits = self.embedder.search_index(sq, top_k=top_k)
            for h in hits:
                cid = h["chunk_id"]
                if cid not in hits_by_chunk or h["score"] > hits_by_chunk[cid]["score"]:
                    hits_by_chunk[cid] = h
        merged_hits = sorted(hits_by_chunk.values(), key=lambda x: x["score"], reverse=True)[:max(5, top_k)]

        if self.engine == "llm" and self.reasoner is not None:
            answer = self.reasoner.generate_answer(query, merged_hits)
            steps = self.reasoner.generate_steps(query, subqueries, merged_hits)
        else:
            answer = self.summarizer.summarize_chunks(query, merged_hits)
            steps = self.summarizer.reasoning_steps(query, subqueries, merged_hits)
        turn = {
            "query": query,
            "subqueries": subqueries,
            "hits": merged_hits,
            "answer": answer,
            "steps": steps,
            "engine": self.engine,
        }
        self.session.setdefault("history", []).append(turn)
        self._save_session()
        return turn

    def follow_up(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a follow-up by augmenting query with previous answers as context."""
        context = []
        for h in self.session.get("history", [])[-3:]:
            context.append(f"Prev Q: {h['query']}\nA: {h['answer']}")
        contextualized = ("\n\n".join(context) + "\n\nFollow-up: " + query).strip()
        subqueries = self._decompose_query(query)

        hits_by_chunk: Dict[str, Dict[str, Any]] = {}
        # Use original follow-up but also include prior answers as a retrieval hint
        for sq in subqueries + [query + " " + (context[-1] if context else "")]:
            hits = self.embedder.search_index(sq, top_k=top_k)
            for h in hits:
                cid = h["chunk_id"]
                if cid not in hits_by_chunk or h["score"] > hits_by_chunk[cid]["score"]:
                    hits_by_chunk[cid] = h
        merged_hits = sorted(hits_by_chunk.values(), key=lambda x: x["score"], reverse=True)[:max(5, top_k)]

        # Synthesize with context
        joined = "\n\n".join(context)
        if joined:
            conditioned_query = query + f"\n\nUse context when helpful:\n{joined}"
        else:
            conditioned_query = query
        if self.engine == "llm" and self.reasoner is not None:
            answer = self.reasoner.generate_answer(conditioned_query, merged_hits, context=joined)
            steps = self.reasoner.generate_steps(query, subqueries, merged_hits)
        else:
            answer = self.summarizer.summarize_chunks(conditioned_query, merged_hits)
            steps = self.summarizer.reasoning_steps(query, subqueries, merged_hits)
        steps.append("Incorporate relevant points from prior answers.")

        turn = {
            "query": query,
            "subqueries": subqueries,
            "hits": merged_hits,
            "answer": answer,
            "steps": steps,
            "follow_up": True,
            "engine": self.engine,
        }
        self.session.setdefault("history", []).append(turn)
        self._save_session()
        return turn

    # Convenience for demo
    def latest(self) -> Optional[Dict[str, Any]]:
        hist = self.session.get("history", [])
        return hist[-1] if hist else None
