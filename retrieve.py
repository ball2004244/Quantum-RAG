"""Milestone 4 — Retrieval over the ChromaDB collection.

retrieve(query, k=5, where=None) -> list of chunk dicts with distances.
The optional `where` filter enables the metadata-filtering stretch feature
(e.g. {"cluster": "reservoir-computing"}).

Run:  python retrieve.py        (runs the eval queries, prints distances)
"""

from __future__ import annotations

from embed import get_collection, get_model

# The 5 evaluation queries (planning.md Evaluation Plan).
EVAL_QUERIES = [
    "In Quantum optical reservoir computing powered by boson sampling (Sakurai "
    "et al.), what physical system acts as the reservoir, and what task is it "
    "used for?",
    "Do dissipative quantum neural networks suffer from barren plateaus?",
    "Across the open-system papers, what mechanism drives the quantum "
    "classifier/QNN to a steady state?",
    "In Beer et al., Training deep quantum neural networks, how is a quantum "
    "neuron/layer defined and how is the network trained?",
    "Which paper introduced the steady-state quantum classifier, and on what "
    "task was it demonstrated?",
]


def retrieve(query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    """Return top-k chunks for a query. `where` is an optional Chroma filter."""
    model = get_model()
    collection = get_collection()

    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {
            "text": docs[i],
            "source": metas[i]["source"],
            "section": metas[i]["section"],
            "cluster": metas[i]["cluster"],
            "year": metas[i]["year"],
            "distance": dists[i],
        }
        for i in range(len(docs))
    ]


def main() -> None:
    for n, q in enumerate(EVAL_QUERIES, 1):
        print(f"\n{'=' * 70}\nQ{n}: {q}\n{'=' * 70}")
        for r in retrieve(q, k=5):
            print(f"  {r['distance']:.3f}  {r['source']:20s} | {r['section'][:45]}")


if __name__ == "__main__":
    main()
