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


def handle_query(question: str, cluster: str):
    if not question or not question.strip():
        return "", ""

    where = None if cluster == "all" else {"cluster": cluster}
    result = ask(question, where=where)

    sources_text = "\n".join(f"• {s}" for s in result["sources"]) or "(none)"
    return result["answer"], sources_text


with gr.Blocks(title="QML Unofficial Guide") as demo:
    gr.Markdown("## QML Research RAG — Unofficial Guide")
    gr.Markdown(
        "Ask a question about the quantum-machine-learning paper corpus. "
        "Answers are grounded strictly in retrieved chunks, with sources listed."
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
            label="Filter by cluster (optional)",
            scale=1,
        )

    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=10)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    btn.click(handle_query, inputs=[inp, cluster], outputs=[answer, sources])
    inp.submit(handle_query, inputs=[inp, cluster], outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
