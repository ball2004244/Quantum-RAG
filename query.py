"""Milestone 5 — Grounded generation via Groq.

ask(question, k=5, where=None) -> {answer, sources, chunks}

Grounding is enforced in the system prompt: the model must answer ONLY from the
provided context and must emit an exact refusal string when the context is
insufficient. Source attribution is programmatic (pulled from chunk metadata),
not left to the LLM.

CLI:  python query.py "your question here"
      python query.py            (runs the out-of-scope refusal test)
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
REFUSAL = "I don't have enough information in the retrieved documents to answer that."

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


def ask(question: str, k: int = 5, where: dict | None = None) -> dict:
    chunks = retrieve(question, k=k, where=where)

    if not chunks:
        return {"answer": REFUSAL, "sources": [], "chunks": []}

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

    # Programmatic source attribution — deduplicated, retrieval-order preserved.
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {"answer": answer, "sources": sources, "chunks": chunks}


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
    print("SOURCES: " + ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
