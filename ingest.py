"""Ingestion utilities for KnowledgeSeeker.

Parses PDF and TXT files, cleans text, and chunks into ~500-word chunks.
Persists chunk metadata to disk so it can be embedded and indexed later without re-processing.

Outputs are saved under the data directory (default: ./.data):
- documents.jsonl: one record per document with metadata
- chunks.jsonl: one record per chunk with text and references to parent document

Usage:
    from ingest import Ingestor
    ing = Ingestor()
    ing.process_files(["sample_docs/doc1.pdf", "sample_docs/doc2.txt"])  # creates chunks.jsonl
"""
from __future__ import annotations

import os
import re
import json
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable

from PyPDF2 import PdfReader

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")
DOCS_FILE = "documents.jsonl"
CHUNKS_FILE = "chunks.jsonl"


@dataclass
class DocumentRecord:
    doc_id: str
    path: str
    filename: str
    n_pages: int
    n_chunks: int


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


class Ingestor:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.docs_path = os.path.join(self.data_dir, DOCS_FILE)
        self.chunks_path = os.path.join(self.data_dir, CHUNKS_FILE)

    def _read_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts)

    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _clean_text(self, text: str) -> str:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_text(self, text: str, target_words: int = 500) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), target_words):
            chunk_words = words[i : i + target_words]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))
        return chunks

    def _append_jsonl(self, path: str, records: Iterable[Dict]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def process_files(self, file_paths: List[str]) -> Dict[str, int]:
        """Process files into chunk records and persist to disk.

        Returns stats dict.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        n_docs, n_chunks_total = 0, 0
        doc_records: List[DocumentRecord] = []
        chunk_records: List[ChunkRecord] = []

        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"[warn] File not found: {fp}")
                continue
            ext = os.path.splitext(fp)[1].lower()
            if ext == ".pdf":
                raw = self._read_pdf(fp)
                # PyPDF2>=3: page count via len(reader.pages)
                try:
                    reader = PdfReader(fp)
                    n_pages = len(reader.pages)
                except Exception:
                    n_pages = 0
            elif ext in (".txt", ".md"):
                raw = self._read_txt(fp)
                n_pages = 1
            else:
                print(f"[warn] Unsupported file type: {fp}")
                continue

            cleaned = self._clean_text(raw)
            chunks = self._chunk_text(cleaned, target_words=500)

            doc_id = str(uuid.uuid4())
            doc_records.append(
                DocumentRecord(
                    doc_id=doc_id,
                    path=os.path.abspath(fp),
                    filename=os.path.basename(fp),
                    n_pages=n_pages,
                    n_chunks=len(chunks),
                )
            )
            for idx, ch in enumerate(chunks):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=str(uuid.uuid4()), doc_id=doc_id, chunk_index=idx, text=ch
                    )
                )

            n_docs += 1
            n_chunks_total += len(chunks)

        if doc_records:
            self._append_jsonl(self.docs_path, (asdict(d) for d in doc_records))
        if chunk_records:
            self._append_jsonl(self.chunks_path, (asdict(c) for c in chunk_records))

        return {"documents": n_docs, "chunks": n_chunks_total}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF/TXT files into chunks")
    parser.add_argument("files", nargs="+", help="Paths to PDFs or TXTs")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    ing = Ingestor(data_dir=args.data_dir)
    stats = ing.process_files(args.files)
    print("Ingestion complete:", stats)
