"""Milestone 5 — Gradio web UI wrapping query.ask().

Includes the optional metadata-filter stretch feature: a cluster dropdown that
maps to a ChromaDB `where` filter, changing which documents can be retrieved.

Run:  python app.py   ->  http://localhost:7860
"""

from __future__ import annotations

import gradio as gr

from query import ask

CLUSTERS = [
    "all",
    "open-system-classifiers",
    "qnn-trainability",
    "reservoir-computing",
    "foundational-ml",
]


def handle_query(question: str, cluster: str, mode: str):
    if not question or not question.strip():
        return "", "", ""

    # Metadata filtering only applies to semantic mode.
    where = None if (cluster == "all" or mode == "hybrid") else {"cluster": cluster}
    result = ask(question, where=where, mode=mode)

    used_text = "\n".join(f"• {s}" for s in result["sources_used"]) or "(none cited)"
    retrieved_text = "\n".join(f"• {s}" for s in result["sources_retrieved"]) or "(none)"
    return result["answer"], used_text, retrieved_text


with gr.Blocks(title="QML Unofficial Guide") as demo:
    gr.Markdown("## QML Research RAG — Unofficial Guide")
    gr.Markdown(
        "Ask a question about the quantum-machine-learning paper corpus. "
        "Answers are grounded strictly in retrieved chunks. **Cited by answer** "
        "lists the sources the model actually used (verified against retrieval); "
        "**Retrieved (top-k)** lists everything that was searched."
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. Do dissipative QNNs suffer from barren plateaus?",
            scale=4,
        )
        cluster = gr.Dropdown(
            choices=CLUSTERS,
            value="all",
            label="Filter by cluster (semantic mode only)",
            scale=1,
        )
        mode = gr.Radio(
            choices=["semantic", "hybrid"],
            value="semantic",
            label="Retrieval mode",
            scale=1,
        )

    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=10)
    with gr.Row():
        sources_used = gr.Textbox(label="Cited by answer (verified)", lines=4)
        sources_retrieved = gr.Textbox(label="Retrieved (top-k)", lines=4)

    outputs = [answer, sources_used, sources_retrieved]
    btn.click(handle_query, inputs=[inp, cluster, mode], outputs=outputs)
    inp.submit(handle_query, inputs=[inp, cluster, mode], outputs=outputs)


if __name__ == "__main__":
    demo.launch()
