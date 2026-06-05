"""Milestone 5 — Grounded generation via Groq.

ask(question, k=5, where=None) -> {answer, sources, sources_used,
                                   sources_retrieved, chunks}

Grounding is enforced in the system prompt: the model must answer ONLY from the
provided context and must emit an exact refusal string when the context is
insufficient.

Source attribution has two layers:
  * sources_retrieved — every source in the top-k (what we looked at).
  * sources_used      — the [Source: ...] tags the model actually cited,
                        VERIFIED against the retrieved set (hallucinated or
                        malformed citations are dropped). This is the precise
                        "which documents the answer drew on" list.
`sources` aliases sources_used when the model cited valid sources, else falls
back to sources_retrieved — so attribution is always programmatically guaranteed
and never invented by the LLM.

CLI:  python query.py "your question here"
      python query.py            (runs the out-of-scope refusal test)
"""

from __future__ import annotations

import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
REFUSAL = "I don't have enough information in the retrieved documents to answer that."

# Matches the citation tags we ask the model to emit, e.g.
#   [Source: dqnn_20 | Section: Results]   or   [Source: dqnn_20]
_CITE_TAG_RE = re.compile(r"\[Source:\s*([^\]|]+?)\s*(?:\||\])")

SYSTEM_PROMPT = f"""\
You are a research assistant for quantum machine learning papers.
Answer the user's question using ONLY the information in the provided context chunks.
Do NOT use your general knowledge or training data.
If the context does not contain enough information to answer, respond with exactly:
"{REFUSAL}"
Be precise and cite which document (by its [Source: ...] tag) each claim comes from.
When sources disagree, surface the disagreement rather than picking one side."""

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['source']} | Section: {c['section']}]\n{c['text']}"
        for c in chunks
    )


def _verify_citations(answer: str, retrieved: list[str]) -> list[str]:
    """Extract [Source: X] tags the model emitted and keep only valid ones.

    A citation counts only if its source is actually in the retrieved set, so a
    model that cites a paper it wasn't given (hallucinated attribution) is
    filtered out. Order follows first appearance in the answer.
    """
    retrieved_set = set(retrieved)
    used: list[str] = []
    for raw in _CITE_TAG_RE.findall(answer):
        src = raw.strip()
        if src in retrieved_set and src not in used:
            used.append(src)
    return used


def ask(question: str, k: int = 5, where: dict | None = None,
        mode: str = "semantic") -> dict:
    """Answer a question with grounded generation.

    mode: "semantic" (vector search, supports `where` metadata filter) or
          "hybrid" (BM25 + semantic RRF fusion; ignores `where`).
    """
    if mode == "hybrid":
        from hybrid import hybrid_retrieve

        chunks = hybrid_retrieve(question, k=k)
    else:
        chunks = retrieve(question, k=k, where=where)

    if not chunks:
        return {
            "answer": REFUSAL,
            "sources": [],
            "sources_used": [],
            "sources_retrieved": [],
            "chunks": [],
        }

    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
    )
    answer = resp.choices[0].message.content.strip()

    # Full retrieved set — deduplicated, retrieval-order preserved.
    sources_retrieved = list(dict.fromkeys(c["source"] for c in chunks))

    # Citations the model actually made, verified against the retrieved set.
    # A refusal cites nothing, so sources_used is naturally empty there.
    sources_used = _verify_citations(answer, sources_retrieved)

    # Headline attribution: prefer the precise "used" list; fall back to the
    # full retrieved list if the model cited nothing valid (keeps the guarantee
    # that attribution is never empty for a real answer).
    sources = sources_used or sources_retrieved

    return {
        "answer": answer,
        "sources": sources,
        "sources_used": sources_used,
        "sources_retrieved": sources_retrieved,
        "chunks": chunks,
    }


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        # Out-of-scope test: not in the corpus -> must refuse.
        question = "How does surface code error correction work?"
        print("(no question given — running out-of-scope refusal test)\n")

    result = ask(question)
    print(f"Q: {question}\n")
    print("ANSWER:\n" + result["answer"] + "\n")
    print("SOURCES USED (model-cited, verified): "
          + (", ".join(result["sources_used"]) or "(none)"))
    print("SOURCES RETRIEVED (top-k): "
          + ", ".join(result["sources_retrieved"]))


if __name__ == "__main__":
    main()
