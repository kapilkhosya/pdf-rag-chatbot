"""Core RAG logic: PDF loading, chunking, embedding, retrieval, and answering."""

import os
from dataclasses import dataclass

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided. If the answer is not in the context, say you could not find it in "
    "the documents. Cite sources like [file.pdf, p.3] after the claims they support."
)


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def load_pdf(data: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for a PDF given as bytes."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, source, chunk_size=800, overlap=150) -> list[Chunk]:
    """Split each page into overlapping character chunks, keeping page numbers."""
    chunks = []
    step = chunk_size - overlap
    for page_num, text in pages:
        text = " ".join(text.split())  # normalise whitespace
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size]
            if len(piece) > 50:  # skip tiny fragments
                chunks.append(Chunk(piece, source, page_num))
    return chunks


class RAGIndex:
    def __init__(self, embedder: SentenceTransformer | None = None):
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vectors
        self.chunks: list[Chunk] = []
        self.sources: set[str] = set()

    def add_pdf(self, name: str, data: bytes) -> int:
        """Index a PDF. Returns the number of chunks added."""
        if name in self.sources:
            return 0
        chunks = chunk_pages(load_pdf(data), name)
        if not chunks:
            return 0
        vectors = self.embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )
        self.index.add(np.asarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)
        self.sources.add(name)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def build_context(hits) -> str:
    return "\n\n".join(
        f"[{c.source}, p.{c.page}]\n{c.text}" for c, _ in hits
    )


def _call_anthropic(messages: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return resp.content[0].text


def _call_ollama(messages: list[dict]) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer(question: str, hits, history: list[dict] | None = None) -> str:
    """Ask the LLM to answer using only the retrieved chunks.

    history: earlier turns as [{"role": "user"|"assistant", "content": str}, ...]
    so that follow-up questions work.
    """
    context = build_context(hits)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    messages = (history or []) + [{"role": "user", "content": user_msg}]

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "ollama":
        return _call_ollama(messages)
    return _call_anthropic(messages)
