"""Stretch feature — Hybrid retrieval: BM25 (lexical) + semantic, fused by RRF.

Builds a BM25 index over the SAME chunk texts used for the vector store, runs both
retrievers for a query, and fuses their rankings with Reciprocal Rank Fusion (RRF).
RRF is rank-based, so it sidesteps the cosine-distance vs BM25-score scale mismatch
— no score normalization needed.

    hybrid_retrieve(query, k=5, k_rrf=60) -> list[dict]

Each result carries: text, source, section, cluster, year, semantic_distance,
semantic_rank, bm25_rank, rrf_score.

CLI:  python hybrid.py            (compares BM25-only / semantic-only / hybrid
                                    on the 5 eval queries; flags Q2 dqnn_20)
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from chunk import build_chunks
from embed import get_collection, get_model

# How many candidates to pull from each retriever before fusing. Larger than the
# final k so a chunk ranked low by one method but high by the other can surface.
_CANDIDATES = 50

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop length-1 tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class HybridRetriever:
    """Holds the chunk list, the BM25 index, and the shared embed model/collection."""

    def __init__(self) -> None:
        self.chunks = build_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.ids = [c["id"] for c in self.chunks]
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
        self._model = get_model()
        self._collection = get_collection()

    # --- individual retrievers (return ordered lists of chunk ids) ----------
    def _bm25_rank(self, query: str, n: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in order[:n] if scores[i] > 0]

    def _semantic_rank(self, query: str, n: int) -> tuple[list[str], dict[str, float]]:
        q_emb = self._model.encode([query]).tolist()
        res = self._collection.query(
            query_embeddings=q_emb,
            n_results=n,
            include=["distances"],
        )
        ids = res["ids"][0]
        dists = {i: d for i, d in zip(ids, res["distances"][0])}
        return ids, dists

    # --- fusion --------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5, k_rrf: int = 60) -> list[dict]:
        sem_ids, sem_dist = self._semantic_rank(query, _CANDIDATES)
        bm25_ids = self._bm25_rank(query, _CANDIDATES)

        sem_rank = {cid: r for r, cid in enumerate(sem_ids)}
        bm25_rank = {cid: r for r, cid in enumerate(bm25_ids)}

        # RRF over the union of both candidate sets.
        rrf: dict[str, float] = {}
        for cid in set(sem_ids) | set(bm25_ids):
            score = 0.0
            if cid in sem_rank:
                score += 1.0 / (k_rrf + sem_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (k_rrf + bm25_rank[cid])
            rrf[cid] = score

        ranked = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:k]
        out = []
        for cid in ranked:
            c = self.by_id[cid]
            out.append(
                {
                    "text": c["text"],
                    "source": c["source"],
                    "section": c["section"],
                    "cluster": c["cluster"],
                    "year": c["year"],
                    "semantic_distance": sem_dist.get(cid),
                    "semantic_rank": sem_rank.get(cid),
                    "bm25_rank": bm25_rank.get(cid),
                    "rrf_score": rrf[cid],
                }
            )
        return out


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def hybrid_retrieve(query: str, k: int = 5, k_rrf: int = 60) -> list[dict]:
    return get_retriever().retrieve(query, k=k, k_rrf=k_rrf)


# --- comparison harness -----------------------------------------------------
_EVAL = [
    ("Q1", "In Quantum optical reservoir computing powered by boson sampling "
           "(Sakurai et al.), what physical system acts as the reservoir, and "
           "what task is it used for?"),
    ("Q2", "Do dissipative quantum neural networks suffer from barren plateaus?"),
    ("Q5", "Which paper introduced the steady-state quantum classifier, and on "
           "what task was it demonstrated?"),
]


def _sources(ids_or_dicts) -> str:
    out, seen = [], set()
    for x in ids_or_dicts:
        s = x if isinstance(x, str) else x["source"]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return ", ".join(out)


def main() -> None:
    r = get_retriever()
    for tag, q in _EVAL:
        print("\n" + "=" * 72)
        print(f"{tag}: {q}")
        print("=" * 72)

        sem_ids, _ = r._semantic_rank(q, 5)
        bm25_ids = r._bm25_rank(q, 5)
        hyb = r.retrieve(q, k=5)

        print(f"  semantic-only top-5 sources: {_sources([r.by_id[i]['source'] for i in sem_ids])}")
        print(f"  BM25-only     top-5 sources: {_sources([r.by_id[i]['source'] for i in bm25_ids])}")
        print(f"  HYBRID (RRF)  top-5 sources: {_sources(hyb)}")

        # Spotlight the Q2 contradiction: does dqnn_20 (the 'no BP' side) appear?
        for label, ids in (("semantic", [r.by_id[i] for i in sem_ids]),
                           ("bm25", [r.by_id[i] for i in bm25_ids]),
                           ("hybrid", hyb)):
            srcs = {c["source"] for c in ids}
            if "dqnn_20" in srcs or "dqnn_train_22" in srcs:
                has20 = "dqnn_20 ✓" if "dqnn_20" in srcs else "dqnn_20 ✗"
                has22 = "dqnn_train_22 ✓" if "dqnn_train_22" in srcs else "dqnn_train_22 ✗"
                print(f"    [{label}] {has20} | {has22}")


if __name__ == "__main__":
    main()
