"""Summarization and reasoning explanation utilities.

Uses a local HuggingFace seq2seq model (Flan-T5 Small) to summarize
texts and synthesize answers. Also provides a lightweight chain-of-thought
style explanation (structured reasoning steps) without exposing raw model
hidden content.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HF_MODEL = os.environ.get("KS_SUMMARY_MODEL", "google/flan-t5-small")


class Summarizer:
    """Wrapper around a local Flan-T5 model for summarization and synthesis."""

    def __init__(self, model_name: str = HF_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def _generate(self, prompt: str, max_new_tokens: int = 320) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.strip()

    def _extract_sentences(self, text: str, keywords: List[str], max_chars: int = 800) -> str:
        # Split into sentences and keep those with keyword matches
        sents = re.split(r"(?<=[.!?])\s+", text)
        key_re = re.compile("|".join(re.escape(k) for k in keywords if k), re.IGNORECASE) if keywords else None
        kept: List[str] = []
        for s in sents:
            if not s.strip():
                continue
            if key_re is None or key_re.search(s):
                kept.append(s.strip())
            if sum(len(x) for x in kept) >= max_chars:
                break
        if not kept:
            return text[:max_chars]
        return " \n".join(kept)[:max_chars]

    def summarize_chunks(self, query: str, chunks: List[Dict[str, Any]], max_new_tokens: int = 320, cite: bool = True) -> str:
        """Summarize a set of chunks in the context of the query.

        Uses an extractive pre-filter to reduce hallucination and adds citation-friendly prompt.
        """
        # Build keyword list from query
        keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
        # Concatenate trimmed chunk texts with file attributions, using extractive filter
        joined = []
        for ch in chunks:
            src = ch.get("doc_meta", {}).get("filename", "")
            raw = ch.get("text", "") or ""
            text = self._extract_sentences(raw, keywords, max_chars=600)
            joined.append(f"[From: {src}] {text}")
        source_text = "\n".join(joined)[:2200]
        constraints = (
            "Only use facts from the notes. Prefer quoting short snippets with [filename] citations. "
            "If information is missing, state that explicitly."
        )
        if cite:
            constraints += " Provide inline citations like [filename]."
        prompt = (
            "You are a careful research assistant. " + constraints + "\n\n"
            f"Query: {query}\n\nNotes:\n{source_text}\n\nAnswer:"
        )
        out = self._generate(prompt, max_new_tokens=max_new_tokens)
        if out:
            return out
        # Fallback: simple extractive bullets
        bullets = []
        for ch in chunks[:5]:
            src = ch.get("doc_meta", {}).get("filename", "")
            snippet = (ch.get("text", "") or "").strip()[:200]
            if snippet:
                bullets.append(f"- [{src}] {snippet}")
        return "\n".join(bullets) if bullets else "No sufficient information found in the retrieved passages."

    def simplify(self, text: str, max_new_tokens: int = 200) -> str:
        prompt = (
            "Rewrite the following text in simpler, clear language for a general audience. "
            "Preserve the key points.\n\nText:\n" + text + "\n\nSimpler explanation:"
        )
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def reasoning_steps(self, query: str, subqueries: List[str], hits: List[Dict[str, Any]]) -> List[str]:
        """Produce human-readable, high-level reasoning steps.

        We do not expose chain-of-thought; instead we provide structured steps used by the agent.
        """
        steps: List[str] = []
        steps.append(f"Understand the query: '{query}'.")
        if subqueries:
            steps.append("Decompose into sub-questions: " + "; ".join(subqueries) + ".")
        if hits:
            files = list({h.get('doc_meta', {}).get('filename', '') for h in hits if h.get('doc_meta')})
            steps.append("Retrieve relevant passages from: " + ", ".join(files) + ".")
        steps.append("Synthesize a concise answer grounded in retrieved passages.")
        return steps
