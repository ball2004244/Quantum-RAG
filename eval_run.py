"""Evaluation harness — runs the 5 planning.md questions (and an out-of-scope
query) through ask() multiple times, printing answer, programmatic sources, and
the retrieved chunks (source | section | distance). Used to fill the README
Evaluation Report from observed behaviour.

Run:  python eval_run.py [n_repeats]
"""

from __future__ import annotations

import sys

from query import ask, REFUSAL

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
    ("OOS", "How does surface code error correction work?"),
]


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    for tag, q in EVAL:
        print("\n" + "=" * 78)
        print(f"{tag}: {q}")
        print("=" * 78)

        # Retrieval is deterministic, so print chunks once.
        first = ask(q)
        print("RETRIEVED (source | section | distance):")
        for c in first["chunks"]:
            print(f"  {c['distance']:.3f}  {c['source']:20s} | {c['section'][:50]}")
        print(f"SOURCES RETRIEVED (top-k): {', '.join(first['sources_retrieved'])}")
        print(f"SOURCES USED (model-cited, verified): "
              f"{', '.join(first['sources_used']) or '(none)'}")

        # Generation can vary run-to-run; observe stability over `repeats`.
        for i in range(repeats):
            result = first if i == 0 else ask(q)
            ans = result["answer"].replace("\n", " ").strip()
            refused = ans.startswith(REFUSAL[:40])
            print(f"\n--- run {i + 1} {'[REFUSED]' if refused else ''} ---")
            print(ans)


if __name__ == "__main__":
    main()
