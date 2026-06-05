# Project 1 Planning: The Unofficial Guide

> Draft assembled collaboratively with Claude debater and writer.

---

## Domain

**Quantum Machine Learning (QML) in open quantum systems** — a focused research corpus spanning open-system quantum neural networks and classifiers, quantum reservoir computing, QNN trainability (barren plateaus), and the foundational ML methods these build on.

This domain is hard in 2 aspects:

1. The knowledge is academic and unfriendly to majority of people.
2. The academia still have diverge opinions on such topics and thus the unification, synthesized document is barely exist.

---

## Documents

| #   | Source (file)        | Description                                                                                                                                                                                 | Location                        |
| --- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 1   | `qnn_collide_17`     | Türkpençe, Akıncı, Şeker, _Quantum neural networks driven by information reservoir_ (arXiv:1709.03276 / EPJ, 2017). QNN unit as open system; collisional model; Markovian vs non-Markovian. | documents/qnn_collide_17.md     |
| 2   | `classifier_19`      | Türkpençe et al., _A steady state quantum classifier_ (Phys. Lett. A 383, 2019). Single spin + information reservoirs; steady-state magnetization as nonlinear activation.                  | documents/classifier_19.md      |
| 3   | `classifier_22`      | Korkmaz, Topal, Aygül, Türkpençe, _Training an open quantum classifier_ (2022). Adds training to the open-system classifier.                                                                | documents/classifier_22.md      |
| 4   | `classifier_23`      | Korkmaz & Türkpençe, _Dissipative learning of a quantum classifier_ (arXiv:2307.12293, 2023). Learning dynamics of the dissipative classifier.                                              | documents/classifier_23.md      |
| 5   | `dqnn_20`            | Beer et al., _Training deep quantum neural networks_ (Nature Commun. 11:808, 2020). Quantum perceptron = unitary; CP layer maps; fidelity cost; claims **no barren plateau**.               | documents/dqnn_20.md            |
| 6   | `dqnn_train_22`      | Sharma, Cerezo, Cincio, Coles, _Trainability of Dissipative Perceptron-Based QNNs_ (PRL 128, 180505, 2022). Proves DQNNs **can** have barren plateaus.                                      | documents/dqnn_train_22.md      |
| 7   | `barren_plat_25`     | Larocca et al., _A Review of Barren Plateaus in Variational Quantum Computing_ (arXiv:2405.00781, 2025). Review article.                                                                    | documents/barren_plat_25.md     |
| 8   | `boson_reservoir_25` | Sakurai, Hayashi, Munro, Nemoto, _Quantum optical reservoir computing powered by boson sampling_ (Optica Quantum 3(3), 2025). Boson-sampler reservoir; image classification.                | documents/boson_reservoir_25.md |
| 9   | `qrc_24`             | Ivaki, Lazarides, Ala-Nissila, _Quantum reservoir computing on random regular graphs_ (PRA 112, 012622, 2025).                                                                              | documents/qrc_24.md             |
| 10  | `qrp_24`             | Kobayashi & Motome, _Quantum reservoir probing_ (arXiv:2308.00898, 2024). Inverse QRC paradigm.                                                                                             | documents/qrp_24.md             |
| 11  | `pqc`                | Benedetti, Lloyd, Sack, Fiorentini, _Parameterized quantum circuits as machine learning models_ (Quantum Sci. Technol. 4, 2019).                                                            | documents/pqc.md                |

---

## Chunking Strategy

**Chunk size:** 256 tokens, capped to fit the **256-token input window of `all-MiniLM-L6-v2`**.

**Overlap:** 50 tokens.

**Reasoning:**

- These are long-form, structured papers. A retrievable claim (e.g. a method + its quantitative result) usually spans several sentences, so chunks must be large enough to hold one complete thought.
- Research Papers have a natural section boundary, we can utilize that as a convenient split.
- Split by 2 layers: by section (Abstract / Introduction / Methods / Results / Discussion), drop the References section, then sub-split any section longer than 256 tokens.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2`

**Top-k:** 5

**Production tradeoff reflection:**

- **Context length** — MiniLM's 256-token window is the binding constraint here and forces small chunks; a longer-context model (e.g. `bge-large`, `e5-large-v2`, `text-embedding-3-large`) would let chunks hold a full subsection, likely improving retrieval on multi-sentence claims.
- **Domain accuracy** — MiniLM is trained on general text; QML jargon and LaTeX-derived terms ("barren plateau," "completely positive map," "Lindbladian") may embed poorly. A model trained on academic text would likely retrieve better on this corpus.
- **Local vs API/SaaS pipeline** — MiniLM runs locally (free, private, no rate limits) which suits an offline corpus; an API model (OpenAI/Cohere) trades cost + data-egress for higher accuracy and longer context.

---

## Evaluation Plan

| #   | Question                                                                                                                                                       | Expected answer                                                                                                                                                                                                                                                                                                                                                                                                                     | Difficulty                                   |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 1   | In _Quantum optical reservoir computing powered by boson sampling_ (Sakurai et al.), what physical system acts as the reservoir, and what task is it used for? | A **boson sampler** — N indistinguishable single photons through an M-mode (M > N²) **random linear-optical interferometer**. Used for **image classification** (MNIST / K-MNIST / Fashion-MNIST); single-photon inputs outperform coherent states.                                                                                                                                                                                 | Easy (single source)                         |
| 2   | Do dissipative quantum neural networks suffer from barren plateaus?                                                                                            | **Contested / conditional.** Beer et al. (`dqnn_20`) report **no** barren plateau in their dissipative QNN. Sharma et al. (`dqnn_train_22`) **prove DQNNs can** exhibit them: deep _global_ perceptrons with a _global_ cost are untrainable, while shallow _local_ perceptrons with a _local_ cost remain trainable. A grounded answer must surface **both** and the condition.                                                    | Hard (failure case)                      |
| 3   | Across the open-system papers, what mechanism drives the quantum classifier/QNN to a steady state?                                                             | A **collisional (repeated-interaction) model**: the system weakly couples to an **information reservoir**; non-unitary **Lindblad / CP-map** evolution drives the output qubit to a **steady state** that is a statistical mixture of reservoir pointer states (einselection).                                                                                                                                                      | Medium (cross-paper)                         |
| 4   | In Beer et al., _Training deep quantum neural networks_, how is a quantum neuron/layer defined and how is the network trained?                                 | A **quantum perceptron** = an arbitrary **unitary** on m input + n output qubits; a layer = product of perceptrons expressed as a **completely-positive layer-transition map**. Trained by maximizing **fidelity** between output and target via a **quantum analogue of backpropagation**; only two adjacent layers are needed at a time, so memory scales with **width** (enabling deep nets).                                    | Easy-Medium                                  |
| 5   | Which paper introduced the _steady-state_ quantum classifier, and on what task was it demonstrated?                                                            | `classifier_19`, _A steady state quantum classifier_ (Phys. Lett. A, 2019). A single spin weakly coupled to information reservoirs; steady-state magnetization acts as a nonlinear activation and **linearly separates quantum data instances**. It uses **no standard named dataset** and explicitly leaves training/learning to later work (`classifier_22`, `classifier_23`). Proposed implementation: superconducting circuits. | Hard (attribution among 3 look-alike papers) |

---

## Anticipated Challenges

1. **Conflicting sources → synthesis failure.** The corpus genuinely disagrees on barren plateaus in dissipative QNNs (`dqnn_20` says no; `dqnn_train_22` proves yes; `barren_plat_25` adds a third voice). Top-k retrieval will pull contradictory chunks, and the LLM is likely to present one side confidently rather than flag the disagreement or its conditions. This is the planned failure case.
2. **Attribution confusion among near-identical papers.** `classifier_19/22/23` share titles, authors, and vocabulary. Semantic search may return the wrong one, and source citations may misattribute the answer — a citation-correctness risk, not just a retrieval one.
3. **Extraction noise on math/two-column PDFs.** Even with Marker, equations, inline citation markers (`[18]`), and running headers (`arXiv:… [quant-ph]`) can leak into chunks and pollute embeddings.
4. **Embedding-window truncation.** If a chunk exceeds MiniLM's 256-token limit it is silently truncated, so the tail never gets embedded. Mitigation: enforce the ~256-token cap at chunk time and assert it in code.

---

## Architecture

### ⚙️ OFFLINE — Conversion · Chunking · Indexing

```mermaid
flowchart LR
    A["📄 Raw PDFs<br/>raw_pdfs/*.pdf"]
    B["📝 Raw Markdown<br/>documents/*.md"]
    C["📑 Sections<br/>split by heading"]
    D["🧩 Chunks<br/>~256 tok · 50 overlap"]
    E["🔢 Embeddings<br/>384-dim vectors"]
    F[("🗄️ ChromaDB<br/>Vector Store")]

    A -->|"Marker<br/>skip &gt; 50 pages"| B
    B -->|"Section Splitter<br/>drop References"| C
    C -->|"Clean &amp; Sub-split<br/>256-tok cap"| D
    D -->|"all-MiniLM-L6-v2<br/>encode"| E
    E -->|"upsert +<br/>metadata"| F

    classDef step  fill:#1a3a2a,stroke:#4caf7d,stroke-width:3px,color:#d4f5e2
    classDef store fill:#1e3a5f,stroke:#4a90d9,stroke-width:3px,color:#e8f4ff
    class A,B,C,D,E step
    class F store
```

### 💬 ONLINE — Retrieval &amp; Generation

```mermaid
flowchart LR
    G["❓ User Query"]
    H["🔢 Query Embedding<br/>384-dim vector"]
    F[("🗄️ ChromaDB<br/>Vector Store")]
    I["📋 Top-5 Chunks<br/>+ source metadata"]
    J["🤖 LLM<br/>Groq llama-3.3-70b"]
    K["✅ Response<br/>answer + citations"]

    G -->|"clean &amp;<br/>encode"| H
    H -->|"cosine<br/>similarity"| F
    F -->|"retrieve<br/>top-k = 5"| I
    I -->|"grounded<br/>prompt"| J
    J --> K

    classDef query    fill:#3a2a00,stroke:#f0a500,stroke-width:3px,color:#fff8e1
    classDef store    fill:#1e3a5f,stroke:#4a90d9,stroke-width:3px,color:#e8f4ff
    classDef retrieve fill:#2d1b4e,stroke:#9b6bd4,stroke-width:3px,color:#ede0ff
    classDef response fill:#1a2f1a,stroke:#6dbf67,stroke-width:3px,color:#e8ffe8
    class G,H query
    class F store
    class I,J retrieve
    class K response
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking.** 
- Tool: Kiro. 
- Input: this _Documents_ section (file list, that sources are scientific PDFs), the _Chunking Strategy_ section (Marker → Markdown, ~256-token section-aware chunks, drop References), and the _Architecture_ diagram. 
- Expected output: 
    (a) an offline `convert.py` that runs Marker over `raw_pdfs/`, skips >50-page PDFs, and writes `documents/*.md`; (b) a `chunk.py` implementing section-aware splitting with the 256-token cap + 50-token overlap and attaching `{source, cluster, year, section}` metadata. 
- Verify: run on the real corpus, print 5 random chunks, confirm each is self-contained, header-aligned, reference-free, and ≤256 tokens; confirm total chunk count is sane (~50–2,000).

**Milestone 4 — Embedding and retrieval.** 
- Tool: Kiro. 
- Input: the _Retrieval Approach_ section + diagram. 
- Expected output: `embed.py` (load chunks, embed with `all-MiniLM-L6-v2`, upsert to ChromaDB with metadata) and a `retrieve(query, k=5)` function returning chunks + sources + distances. 
- Verify: run Q1, Q3, Q5 from the Evaluation Plan, print distances; confirm Q1 returns `boson_reservoir_25` with low distance; watch whether Q2 pulls _both_ `dqnn_20` and `dqnn_train_22`.

**Milestone 5 — Generation and interface.** 
- Tool: Kiro. 
- Input: the grounding requirement (answer from retrieved context only; refuse when unsupported), the output format (answer + source list), and the Gradio skeleton from the instructions. 
- Expected output: `query.py` with a grounded prompt template + programmatic source attribution, and `app.py` (Gradio). 
- Verify: confirm the system prompt _enforces_ grounding (refuses an out-of-scope query, e.g. about surface-code error correction), that citations are appended from retrieval metadata (not left to the LLM), and that Q2 exposes the contradiction behavior for the failure write-up.

