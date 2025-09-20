"""CLI for KnowledgeSeeker Agent.

Provides commands to:
- ingest files (PDF/TXT)
- build index
- ask a query
- ask a follow-up query
- export latest result to Markdown/PDF

Examples:
  python cli.py ingest sample_docs/doc1.txt sample_docs/doc2.txt
  python cli.py build
  python cli.py ask "Summarize the key findings"
  python cli.py followup "Explain in simpler terms"
  python cli.py export --fmt md --out report.md
  python cli.py export --fmt pdf --out report.pdf
"""
from __future__ import annotations

import os
import argparse

from ingest import Ingestor, DEFAULT_DATA_DIR
from embeddings import Embedder
from query import KnowledgeSeeker
from export import export_markdown, export_pdf


def cmd_ingest(args: argparse.Namespace) -> None:
    ing = Ingestor(data_dir=args.data_dir)
    stats = ing.process_files(args.files)
    print("Ingestion complete:", stats)


def cmd_build(args: argparse.Namespace) -> None:
    emb = Embedder(data_dir=args.data_dir)
    stats = emb.build_index()
    print("Index built:", stats)


def cmd_ask(args: argparse.Namespace) -> None:
    ks = KnowledgeSeeker(data_dir=args.data_dir, engine=args.engine, llm_model=args.llm_model)
    res = ks.ask(args.query, top_k=args.top_k)
    print("Answer:\n", res["answer"]) 
    print("\nReasoning steps:")
    for i, s in enumerate(res["steps"], 1):
        print(f" {i}. {s}")


def cmd_followup(args: argparse.Namespace) -> None:
    ks = KnowledgeSeeker(data_dir=args.data_dir, engine=args.engine, llm_model=args.llm_model)
    res = ks.follow_up(args.query, top_k=args.top_k)
    print("Answer:\n", res["answer"]) 
    print("\nReasoning steps:")
    for i, s in enumerate(res["steps"], 1):
        print(f" {i}. {s}")


def cmd_export(args: argparse.Namespace) -> None:
    ks = KnowledgeSeeker(data_dir=args.data_dir, engine=args.engine, llm_model=args.llm_model)
    latest = ks.latest()
    if not latest:
        print("No conversation found to export.")
        return
    latest = {**latest, "title": args.title}
    if args.fmt == "md":
        path = export_markdown(latest, filename=os.path.basename(args.out) if args.out else None)
    else:
        path = export_pdf(latest, filename=os.path.basename(args.out) if args.out else None)
    print("Exported:", path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KnowledgeSeeker CLI")
    p.set_defaults(func=lambda a: p.print_help())
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--engine", choices=["t5", "llm"], default="t5", help="Reasoning/summary engine")
    p.add_argument("--llm-model", default="TinyLlama/TinyLlama-1.1B-Chat-v0.6", help="HF model id for engine=llm")

    sub = p.add_subparsers(dest="command")

    p_ing = sub.add_parser("ingest", help="Ingest PDF/TXT files")
    p_ing.add_argument("files", nargs="+")
    p_ing.set_defaults(func=cmd_ingest)

    p_bld = sub.add_parser("build", help="Build FAISS index from ingested chunks")
    p_bld.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="Ask a new query")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.set_defaults(func=cmd_ask)

    p_fu = sub.add_parser("followup", help="Ask a follow-up query using context")
    p_fu.add_argument("query")
    p_fu.add_argument("--top-k", type=int, default=5)
    p_fu.set_defaults(func=cmd_followup)

    p_exp = sub.add_parser("export", help="Export latest answer and reasoning")
    p_exp.add_argument("--fmt", choices=["md", "pdf"], default="md")
    p_exp.add_argument("--out", default=None, help="Output filename (optional)")
    p_exp.add_argument("--title", default="KnowledgeSeeker Report")
    p_exp.set_defaults(func=cmd_export)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
