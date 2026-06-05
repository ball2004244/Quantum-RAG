"""Stretch feature — Chunking-strategy comparison.

Compares three chunking strategies on the same 5 eval queries, all retrieved with
the same semantic vector search so only the *chunking* differs:

  A) section-aware  — the baseline in chunk.py (heading split, sentence-packed,
                      ≤256 tok, 50 overlap, reference/degenerate filtering).
  B) fixed-window   — naive "split every 256 tokens, 50 overlap" ignoring all
                      section/sentence boundaries (the approach the rubric warns
                      against).
  C) semantic       — split into sentences, embed each, cut a boundary where the
                      cosine distance between consecutive sentences spikes past
                      the 85th percentile (a topic shift).

Each strategy is embedded into its own throwaway ChromaDB collection. The report
shows, per query: chunk counts, top-1 distance + source, whether top-1 is the
EXPECTED paper, and the full top-5 source list. Q2 is the contradiction case, so
instead of a single expected source it reports where BOTH sides of the
disagreement (dqnn_train_22 "yes" vs dqnn_20 "no") first rank.

Run:  python chunk_compare.py
"""

from __future__ import annotations

import chromadb
import numpy as np
from tqdm import tqdm

from chunk import (
    MAX_TOKENS,
    META,
    OVERLAP,
    _SENT_SPLIT_RE,
    _clean,
    _tok,
    build_chunks,
    token_count,
)
from embed import EMBED_MODEL, get_model
from pathlib import Path

DOCS_DIR = Path("documents")


# --- Strategy B: fixed-size token window, boundary-agnostic -----------------
def fixed_window_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Split each document into fixed 256-token windows (50 overlap), ignoring
    headings and sentences. Light cleaning only (same artifact strip as baseline)."""
    step = MAX_TOKENS - OVERLAP
    chunks: list[dict] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        stem = md_path.stem
        cluster, year = META.get(stem, ("uncategorized", 0))
        text = _clean(md_path.read_text(encoding="utf-8"))
        ids = _tok.encode(text, add_special_tokens=False)
        for start in range(0, len(ids), step):
            window = ids[start : start + MAX_TOKENS]
            if not window:
                break
            piece = _tok.decode(window, skip_special_tokens=True).strip()
            if piece:
                idx = len(chunks)
                chunks.append({
                    "id": f"{stem}__fixed__{idx:03d}",
                    "text": piece,
                    "source": stem,
                    "section": "(fixed-window)",
                    "cluster": cluster,
                    "year": year,
                })
            if start + MAX_TOKENS >= len(ids):
                break
    return chunks


# --- Strategy C: semantic chunking, boundary = embedding-distance spike -----
def semantic_chunks(docs_dir: Path = DOCS_DIR,
                    percentile: float = 85.0) -> list[dict]:
    """Split on *meaning shifts*: embed each sentence, cut a chunk boundary where
    the cosine distance between consecutive sentence embeddings exceeds the
    `percentile`-th distance for that document (a topic-shift breakpoint).
    Sentences between breakpoints are grouped, still respecting the 256-tok cap.
    """
    model = get_model()
    out: list[dict] = []
    md_files = sorted(docs_dir.glob("*.md"))
    for md_path in tqdm(md_files, desc="semantic-chunk", unit="doc"):
        stem = md_path.stem
        cluster, year = META.get(stem, ("uncategorized", 0))
        text = _clean(md_path.read_text(encoding="utf-8"))
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) < 2:
            continue

        embs = model.encode(sentences, batch_size=64, show_progress_bar=False)
        embs = np.asarray(embs)
        # Cosine distance between consecutive sentences (vectors are L2-ish; use
        # normalized dot to be safe).
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        unit = embs / np.clip(norms, 1e-9, None)
        sims = np.sum(unit[:-1] * unit[1:], axis=1)
        dists = 1.0 - sims
        threshold = float(np.percentile(dists, percentile)) if len(dists) else 1.0

        # Greedy grouping: start a new chunk at a breakpoint OR when adding the
        # next sentence would exceed the token cap.
        groups: list[str] = []
        cur = sentences[0]
        for i in range(1, len(sentences)):
            breakpoint = dists[i - 1] >= threshold
            candidate = f"{cur} {sentences[i]}"
            if breakpoint or token_count(candidate) > MAX_TOKENS:
                groups.append(cur)
                cur = sentences[i]
            else:
                cur = candidate
        if cur.strip():
            groups.append(cur)

        for piece in groups:
            # Enforce the hard cap for any group that a single long sentence blew past.
            if token_count(piece) > MAX_TOKENS:
                ids = _tok.encode(piece, add_special_tokens=False)[:MAX_TOKENS]
                piece = _tok.decode(ids, skip_special_tokens=True).strip()
            if not piece:
                continue
            idx = len(out)
            out.append({
                "id": f"{stem}__semantic__{idx:03d}",
                "text": piece,
                "source": stem,
                "section": "(semantic)",
                "cluster": cluster,
                "year": year,
            })
    return out


def _index(name: str, chunks: list[dict]):
    """Embed chunks into a fresh ephemeral collection, with a progress bar."""
    model = get_model()
    client = chromadb.Client()  # ephemeral, in-memory
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    embs = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
    ).tolist()
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embs,
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )
    return col


EVAL = [
    ("Q1", "In Quantum optical reservoir computing powered by boson sampling "
           "(Sakurai et al.), what physical system acts as the reservoir, and "
           "what task is it used for?"),
    ("Q2", "Do dissipative quantum neural networks suffer from barren plateaus?"),
    ("Q3", "Across the open-system papers, what mechanism drives the quantum "
           "classifier/QNN to a steady state?"),
    ("Q4", "In Beer et al., Training deep quantum neural networks, how is a "
           "quantum neuron/layer defined and how is the network trained?"),
    ("Q5", "Which paper introduced the steady-state quantum classifier, and on "
           "what task was it demonstrated?"),
]

# The single source whose top-1 retrieval we treat as "correct" for that query.
# Q2 is the contradiction case (no single right paper) and Q3 is satisfied by any
# open-system paper, so both are handled specially below rather than via this map.
EXPECTED = {
    "Q1": "boson_reservoir_25",
    "Q4": "dqnn_20",
    "Q5": "classifier_19",
}
OPEN_SYSTEM = {"classifier_19", "classifier_22", "classifier_23", "qnn_collide_17"}


def _ranked_sources(col, model, query: str, n: int = 30):
    res = col.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n,
        include=["metadatas", "distances"],
    )
    return [m["source"] for m in res["metadatas"][0]], res["distances"][0]


def _first_rank(sources: list[str], target: str):
    return next((i for i, s in enumerate(sources) if s == target), None)


def main() -> None:
    print("Building chunks (semantic split embeds every sentence)...")
    builders = [
        ("section-aware", build_chunks),
        ("fixed-window", fixed_window_chunks),
        ("semantic", semantic_chunks),
    ]
    chunk_sets = {label: fn() for label, fn in builders}

    print("\nChunk counts:")
    base = len(chunk_sets["section-aware"])
    for label, _ in builders:
        n = len(chunk_sets[label])
        delta = "" if label == "section-aware" else f"  ({n / base - 1:+.0%} vs section-aware)"
        print(f"  {label:<14} {n:>5} chunks{delta}")
    print(f"embedding model: {EMBED_MODEL}\n")

    model = get_model()
    cols = {}
    for label, _ in builders:
        print(f"Embedding {label}...")
        cols[label] = _index(f"cmp_{label}", chunk_sets[label])

    print("\n" + "=" * 80)
    print("PER-QUERY COMPARISON (same semantic retrieval; only chunking differs)")
    print("=" * 80)
    for tag, q in EVAL:
        print(f"\n{tag}: {q}")
        if tag == "Q2":
            print("  [contradiction case — report where BOTH sides first rank]")
        elif tag in EXPECTED:
            print(f"  expected top-1 source: {EXPECTED[tag]}")
        elif tag == "Q3":
            print(f"  expected: any open-system paper")

        for label, _ in builders:
            srcs, dists = _ranked_sources(cols[label], model, q)
            top5 = srcs[:5]
            top1 = top5[0]

            if tag == "Q2":
                r_yes = _first_rank(srcs, "dqnn_train_22")
                r_no = _first_rank(srcs, "dqnn_20")
                verdict = f"dqnn_train_22@{r_yes} | dqnn_20@{r_no}"
            elif tag in EXPECTED:
                verdict = "OK" if top1 == EXPECTED[tag] else (
                    f"WRONG (correct @{_first_rank(srcs, EXPECTED[tag])})"
                )
            elif tag == "Q3":
                verdict = "OK" if top1 in OPEN_SYSTEM else "WRONG"
            else:
                verdict = ""

            print(f"    {label:<14} d1={dists[0]:.3f}  {verdict}")
            print(f"      top5: {top5}")


if __name__ == "__main__":
    main()
