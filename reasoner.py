"""Local LLM Reasoner for KnowledgeSeeker.

Provides decoder-only LLM-based reasoning and answer generation using
Hugging Face transformers with AutoModelForCausalLM. Designed to run fully
locally on CPU or CUDA if available.

Default model: TinyLlama/TinyLlama-1.1B-Chat-v0.6
Alternative: microsoft/phi-2, Qwen/Qwen2.5-1.5B-Instruct, etc.

APIs:
- ReasonerLLM(model_name: str | None)
  - generate_answer(query: str, chunks: list[dict], context: str = "") -> str
  - generate_steps(query: str, subqueries: list[str], hits: list[dict]) -> list[str]
"""
from __future__ import annotations

import os
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_LLM_MODEL = os.environ.get("KS_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v0.6")


class ReasonerLLM:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_LLM_MODEL
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def _gen(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=0.95,
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the completion after the prompt
        return text[len(prompt):].strip()

    @staticmethod
    def _format_notes(chunks: List[Dict[str, Any]], limit_chars: int = 2400) -> str:
        joined: List[str] = []
        for ch in chunks:
            src = ch.get("doc_meta", {}).get("filename", "")
            text = (ch.get("text", "") or "")[:800]
            joined.append(f"[From: {src}] {text}")
        return "\n".join(joined)[:limit_chars]

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]], context: str = "", max_new_tokens: int = 300) -> str:
        notes = self._format_notes(chunks)
        sys = (
            "You are a precise research assistant. Use only the provided notes to answer. "
            "Cite facts succinctly and avoid speculation."
        )
        prompt = (
            f"<|system|>\n{sys}\n<|user|>\n"
            f"Query: {query}\n"
            + (f"Context: {context}\n" if context else "")
            + f"Notes:\n{notes}\n\n<|assistant|>\nAnswer:"
        )
        return self._gen(prompt, max_new_tokens=max_new_tokens, temperature=0.2)

    def generate_steps(self, query: str, subqueries: List[str], hits: List[Dict[str, Any]], max_new_tokens: int = 200) -> List[str]:
        files = list({h.get('doc_meta', {}).get('filename', '') for h in hits if h.get('doc_meta')})
        notes = self._format_notes(hits, limit_chars=1000)
        sys = (
            "Explain your high-level plan as numbered steps. Do not include private chain-of-thought; "
            "list only the actions taken (e.g., decompose, retrieve, synthesize)."
        )
        subq = ", ".join(subqueries) if subqueries else query
        prompt = (
            f"<|system|>\n{sys}\n<|user|>\n"
            f"Task: Produce numbered high-level steps used to answer the query.\n"
            f"Query: {query}\nSub-queries: {subq}\nSources: {', '.join(files)}\n"
            f"Notes:\n{notes}\n\n<|assistant|>\nSteps:\n1. "
        )
        steps_text = self._gen(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        # Post-process into list
        lines = [ln.strip() for ln in steps_text.splitlines() if ln.strip()]
        # Normalize numbering
        cleaned: List[str] = []
        for ln in lines:
            ln = ln.lstrip("- ")
            if ln.startswith(tuple(str(i) + "." for i in range(1, 10))):
                ln = ln.split(".", 1)[1].strip()
            cleaned.append(ln)
        return cleaned[:8] if cleaned else [
            f"Understand the query: '{query}'.",
            "Decompose the query into sub-questions.",
            "Retrieve relevant passages from sources.",
            "Synthesize a concise answer grounded in retrieved passages.",
        ]
