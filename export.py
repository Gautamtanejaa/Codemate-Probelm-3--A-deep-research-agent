"""Export utilities for KnowledgeSeeker.

Exports research results (answers, steps, citations) to Markdown and PDF.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List
from datetime import datetime

from fpdf import FPDF

DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exports")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_markdown(result: Dict[str, Any]) -> str:
    """Create a Markdown string from a result dict produced by KnowledgeSeeker."""
    title = result.get("title", "KnowledgeSeeker Report")
    query = result.get("query", "")
    answer = result.get("answer", "")
    steps: List[str] = result.get("steps", [])
    hits: List[Dict[str, Any]] = result.get("hits", [])

    md = [f"# {title}", "", f"**Query:** {query}", "", "## Answer", answer, "", "## Reasoning Steps"]
    for i, s in enumerate(steps, 1):
        md.append(f"{i}. {s}")
    md.append("")
    md.append("## Sources")
    if hits:
        for h in hits:
            meta = h.get("doc_meta", {})
            md.append(f"- {meta.get('filename','')} (chunk {h.get('chunk_index',-1)}) score={h.get('score',0):.4f}")
    else:
        md.append("- No sources available")
    md.append("")
    return "\n".join(md)


def export_markdown(result: Dict[str, Any], out_dir: str = DEFAULT_EXPORT_DIR, filename: str | None = None) -> str:
    ensure_dir(out_dir)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}.md"
    out_path = os.path.join(out_dir, filename)
    content = format_markdown(result)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def export_pdf(result: Dict[str, Any], out_dir: str = DEFAULT_EXPORT_DIR, filename: str | None = None) -> str:
    ensure_dir(out_dir)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}.pdf"
    out_path = os.path.join(out_dir, filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, result.get("title", "KnowledgeSeeker Report"))
    pdf.ln(4)

    # Query
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Query: {result.get('query','')}")
    pdf.ln(2)

    # Answer
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Answer", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, result.get("answer", ""))
    pdf.ln(2)

    # Steps
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Reasoning Steps", ln=1)
    pdf.set_font("Arial", "", 12)
    steps: List[str] = result.get("steps", [])
    if steps:
        for i, s in enumerate(steps, 1):
            pdf.multi_cell(0, 6, f"{i}. {s}")
    else:
        pdf.multi_cell(0, 6, "No steps available")
    pdf.ln(2)

    # Sources
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Sources", ln=1)
    pdf.set_font("Arial", "", 12)
    hits: List[Dict[str, Any]] = result.get("hits", [])
    if hits:
        for h in hits:
            meta = h.get("doc_meta", {})
            pdf.multi_cell(0, 6, f"- {meta.get('filename','')} (chunk {h.get('chunk_index',-1)}) score={h.get('score',0):.4f}")
    else:
        pdf.multi_cell(0, 6, "- No sources available")

    pdf.output(out_path)
    return out_path


if __name__ == "__main__":
    # Simple manual test using dummy content
    sample = {
        "title": "Sample Report",
        "query": "Summarize the key findings",
        "answer": "This is a test answer with synthesized points.",
        "steps": ["Understand the query.", "Retrieve passages.", "Synthesize answer."],
        "hits": [
            {"doc_meta": {"filename": "doc1.pdf"}, "chunk_index": 0, "score": 0.88},
            {"doc_meta": {"filename": "doc2.pdf"}, "chunk_index": 1, "score": 0.76},
        ],
    }
    print("MD:", export_markdown(sample))
    print("PDF:", export_pdf(sample))
