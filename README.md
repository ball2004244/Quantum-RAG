# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section _after_ you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

**Quantum Machine Learning in open quantum systems** — a focused research corpus
spanning open-system quantum neural networks and classifiers, quantum reservoir
computing, QNN trainability (barren plateaus), and the foundational ML methods
these build on.

This knowledge is valuable and hard to find through official channels in two ways:

1. It is **academic and unfriendly** to most people — the answers live inside
   dense, math-heavy papers with two-column layouts and notation, not in any
   student-friendly guide.
2. The field **genuinely disagrees** with itself. There is no single official
   synthesized document, because the literature itself is unsettled (most
   sharply: _do dissipative QNNs have barren plateaus?_ — Beer says no, Sharma
   proves yes).

---

## Document Sources

All sources are peer-reviewed / arXiv papers on well-known research journals,
converted from PDF to Markdown in `documents/`.

| #   | Source                                                                                           | Type                                | URL or file path                  |
| --- | ------------------------------------------------------------------------------------------------ | ----------------------------------- | --------------------------------- |
| 1   | Türkpençe et al., _Quantum neural networks driven by information reservoir_ (2017)               | arXiv / EPJ Plus                    | `documents/qnn_collide_17.md`     |
| 2   | Türkpençe et al., _A steady state quantum classifier_ (2019)                                     | Physics Letters A 383               | `documents/classifier_19.md`      |
| 3   | Korkmaz et al., _Training an open quantum classifier_ (2022)                                     | arXiv                               | `documents/classifier_22.md`      |
| 4   | Korkmaz & Türkpençe, _Dissipative learning of a quantum classifier_ (2023)                       | arXiv:2307.12293                    | `documents/classifier_23.md`      |
| 5   | Beer et al., _Training deep quantum neural networks_ (2020) — claims **no barren plateau**       | Nature Communications 11:808        | `documents/dqnn_20.md`            |
| 6   | Sharma et al., _Trainability of Dissipative Perceptron-Based QNNs_ (2022) — proves DQNNs **can** | Physical Review Letters 128, 180505 | `documents/dqnn_train_22.md`      |
| 7   | Larocca et al., _A Review of Barren Plateaus in Variational Quantum Computing_ (2025)            | arXiv:2405.00781                    | `documents/barren_plat_25.md`     |
| 8   | Sakurai et al., _Quantum optical reservoir computing powered by boson sampling_ (2025)           | Optica Quantum 3(3)                 | `documents/boson_reservoir_25.md` |
| 9   | Ivaki et al., _Quantum reservoir computing on random regular graphs_ (2024)                      | Physical Review A 112, 012622       | `documents/qrc_24.md`             |
| 10  | Kobayashi & Motome, _Quantum reservoir probing_ (2024)                                           | arXiv:2308.00898                    | `documents/qrp_24.md`             |
| 11  | Benedetti et al., _Parameterized quantum circuits as machine learning models_ (2019)             | Quantum Sci. Technol. 4             | `documents/pqc.md`                |

---

## Chunking Strategy

**Chunk size:** ≤ **256 tokens**, capped to fit the **256-token input window of
`all-MiniLM-L6-v2`**. Anything longer is silently truncated by the model before
embedding, so the cap is set _by the model_, not arbitrarily.

**Overlap:** **50 tokens** between sub-chunks of an over-long section.

**Why these choices fit your documents:** These are long-form, structured papers
where a retrievable claim (a method plus its result) usually spans several
sentences, so chunks must be big enough to hold one complete thought — but the
256-token model window is a hard ceiling. The 50-token overlap keeps a claim that
straddles a boundary retrievable from either side. Preprocessing before chunking:
split each doc on Markdown headings into sections and drop
References/Bibliography/Acknowledgements; strip Marker artifacts (image embeds,
`<span>` anchors, `<sup>`/`<sub>` tags, markdown/citation links reduced to their
visible text, escaped brackets, inline `[12]`-style citation markers); pack whole
sentences up to the 256-token cap (re-measured with the real MiniLM tokenizer)
with a hard-split fallback; and drop two kinds of garbage that survive
conversion — OCR repetition loops (unique-word ratio < 0.18) and bibliography
chunks (dense with publication years + journal abbreviations) that slip past the
heading filter.

**Final chunk count:** 721 chunks across the 11 documents (mean 194.5 tokens; min
1, max 256).

**Sample chunks** (5 chunks, each labeled with its source document):

| #   | Source               | Section                                           | Chunk text (excerpt)                                                                                                                                                                                                                             |
| --- | -------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `boson_reservoir_25` | Introduction                                      | "Boson sampling is associated with sampling the probability distribution of identical bosons passing through a random interferometer, and its quantum advantage has been demonstrated. It has, however, proven elusive to use such a model for…" |
| 2   | `dqnn_20`            | Results                                           | "…the most practice to exploit noncommuting perceptrons acting on qubits. In fact, the most general quantum perceptron is an arbitrary unitary applied to the input and output qubits…"                                                          |
| 3   | `dqnn_train_22`      | Trainability of Dissipative Perceptron-Based QNNs | "…Each perceptron corresponds to a unitary operation on the qubits it connects; the jth qubit of the lth layer is denoted q_j^l, mapping the input state ρ_in to the output ρ_out."                                                              |
| 4   | `classifier_19`      | 3.1 Theory                                        | "…six curves are plotted, each point representing the steady-state magnetization during the presence of two environmental states \| θ₁⟩ and \| θ₂⟩…"                                                                                            |
| 5   | `qrc_24`             | III. Training, Learning, and Readout              | "…[reservoir outputs] are combined linearly to fit the desired targets y_n, and the optimal weights w are determined by this fitting process. Importantly, we only consider measurements in the computational basis…"                            |

(Generated by `python chunk.py`, which prints 5 random chunks and runs the
≤256-token sanity assertions on every run.)

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` — 384-dimensional
embeddings, runs locally with no API key and no rate limits, stored in a
persistent ChromaDB collection (`qml_papers`, cosine space), retrieving top-k = 5.

**Production tradeoff reflection:** If this were deployed for real users and cost
weren't a constraint, the factors worth weighing against MiniLM are: **context
length** — MiniLM's 256-token window is the binding constraint in this whole
project and forces small chunks; a longer-context model (`bge-large`,
`e5-large-v2`, `text-embedding-3-large`) would let a chunk hold a full subsection,
helping on the multi-sentence claims these papers make. **Domain accuracy** —
MiniLM is trained on general text, so QML jargon and LaTeX-derived terms ("barren
plateau," "completely positive map," "Lindbladian") likely embed imprecisely; a
model trained on scientific text would probably retrieve better here. **Local vs
API / latency** — MiniLM is free, private, and offline, which fits a fixed
research corpus, whereas an API model (OpenAI/Cohere) buys accuracy and context
length at the cost of per-query spend and shipping document text off-machine.

---

## Grounded Generation

**System prompt grounding instruction:**

```
You are a research assistant for quantum machine learning papers.
Answer the user's question using ONLY the information in the provided context chunks.
Do NOT use your general knowledge or training data.
If the context does not contain enough information to answer, respond with exactly:
"I don't have enough information in the retrieved documents to answer that."
Be precise and cite which document (by its [Source: ...] tag) each claim comes from.
When sources disagree, surface the disagreement rather than picking one side.
```

Grounding is enforced by mechanism, not just instruction: each retrieved chunk is
wrapped as `[Source: <stem> | Section: <section>]\n<text>` before being
concatenated into the context, generation runs at **temperature 0**, and if
retrieval returns nothing the code short-circuits to the exact refusal string
without ever calling the LLM. The prompt mandates the _exact_ refusal string,
which makes out-of-scope behaviour testable.

**How source attribution is surfaced in the response:** A **cite-then-verify**
scheme. The model cites `[Source: …]` tags inline per claim; the pipeline then
parses those tags and keeps only the ones whose source is actually in the
retrieved top-k, dropping any the model invents or mis-cites. The UI shows two
lists: **Cited by answer (verified)** — the documents the answer actually drew on
— and **Retrieved (top-k)** — everything searched. On a refusal the model cites
nothing, so the "cited" list is correctly empty. (Example: on Q1 `qrp_24` is
retrieved but unused, so it appears only under Retrieved, not Cited.)

---

## Evaluation Report

Each question was run 3 times through `ask()` (the same function the Gradio UI
calls); generation at temperature 0 was stable across runs. Distances are cosine
(lower = closer).

| #   | Question                                                                      | Expected answer                                                                                                                                | System response (summarized)                                                                                                                                                            | Retrieval quality  | Response accuracy                                                     |
| --- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | --------------------------------------------------------------------- |
| 1   | Boson-sampling reservoir: what system, what task?                             | Boson sampler / random linear-optical interferometer; image classification (MNIST/K-MNIST/Fashion-MNIST).                                      | "The reservoir is the random interferometer powering boson sampling… used to generate complex dynamics… applied to image recognition problems." Top chunk `boson_reservoir_25` @ 0.314. | Relevant           | Accurate (task correct; named datasets not surfaced)                  |
| 2   | Do dissipative QNNs suffer from barren plateaus?                              | Contested: Beer (`dqnn_20`) says no; Sharma (`dqnn_train_22`) proves yes, conditional on global vs local perceptron + cost. Must surface both. | "DQNNs are not immune… authors analytically prove DQNNs can suffer from barren plateaus." Cites only `dqnn_train_22`; never surfaces Beer's "no" side.                                  | Partially relevant | Partially accurate (one side only)                                    |
| 3   | What mechanism drives the open-system classifier/QNN to a steady state?       | Collisional / repeated-interaction model; weak coupling to an information reservoir; non-unitary Lindblad/CP-map evolution → steady state.     | "Interaction with a reservoir… repeated interactions model open-system dynamics… decision encoded in the steady state." Pulls `classifier_19/22/23` + `qnn_collide_17`.                 | Relevant           | Accurate ("Lindblad/einselection" vocabulary not explicit)            |
| 4   | How is a quantum neuron/layer defined and trained in Beer et al.?             | Perceptron = arbitrary unitary on m+n qubits; layer = CP map; fidelity cost; quantum backprop; memory scales with width.                       | "Quantum neuron = completely positive map between input/output qubit registers… trained via fidelity cost, process resembling backpropagation… qudits scale with width."                | Relevant           | Accurate (CP map, fidelity cost, backprop, width-scaling all correct) |
| 5   | Which paper introduced the steady-state quantum classifier, and on what task? | `classifier_19` (Phys. Lett. A 2019); linear separation of quantum data instances; no named dataset; superconducting implementation proposed.  | "Introduced in `classifier_19`… demonstrated on classifying input data in the steady-state limit… `classifier_23` builds on it but doesn't claim to introduce it."                      | Relevant           | Accurate (correct paper + distinguishes look-alikes `_19`/`_23`)      |

**Retrieval quality:** Relevant (Q2 Partially relevant)
**Response accuracy:** Accurate (Q2 Partially accurate)

---

## Failure Case Analysis

**Question that failed:** Q2 — "Do dissipative quantum neural networks suffer from
barren plateaus?"

**What the system returned:** A confident, one-sided answer — "DQNNs are not
immune to barren plateaus… the authors analytically prove that DQNNs can suffer
from barren plateaus" — citing only `dqnn_train_22` (Sharma et al.). It alludes to
"a previous speculation… that suggested DQNNs might not have this problem" but
never names or grounds the opposing source.

**Root cause (tied to a specific pipeline stage):** The retrieval (ranking) stage.
The corpus contains a genuine contradiction: `dqnn_20` (Beer) explicitly claims
"there is no barren plateau in the cost function landscape," while `dqnn_train_22`
(Sharma) proves they can occur. At k = 5 the retrieved chunks were four
`dqnn_train_22` + one `qrc_24`; `dqnn_20` does not appear at all. We verified
`dqnn_20.md` does contain the "no barren plateau" claim and is embedded — it
simply ranks 8th, just outside the top-5 window. The "yes" paper repeats the
query's exact terms ("barren plateau," "dissipative perceptron"), so it dominates
cosine similarity, while Beer phrases its result as the absence of a plateau and
uses the term sparingly. The LLM faithfully grounded its answer in what it was
given, so this is a retrieval/synthesis failure, not a generation or grounding
failure — the correct answer requires integrating two opposing sources, and top-k
similarity surfaced only one.

**What you would change to fix it:** (1) Source-diversified retrieval (MMR or a
per-source cap on top-k chunks) to force the second voice into the context. (2)
Raise k for known-contested topics, since `dqnn_20` appears by k≈8.

---

## Spec Reflection

**One way the spec helped you during implementation:** `planning.md` pinned the
chunk size to the embedding model's 256-token window before any code was written,
and called out the silent-truncation trap explicitly. That single decision drove
the entire chunker design — the sentence-aware packer and the hard 256-token
assertion exist because the spec framed the constraint as model-imposed rather
than a tunable knob. The spec also pre-designated Q2 as the failure case, so when
retrieval pulled one-sided chunks that was a predicted outcome to document rather
than a bug to chase.

**One way your implementation diverged from the spec, and why:** The spec's
sub-split pseudocode was a token-ID sliding window (decode 256 IDs back to text).
In practice that didn't round-trip: decoding token IDs to a string and re-encoding
shifts subword boundaries, so chunks came out over 256 tokens and the sanity
assertion failed. I replaced it with sentence-aware greedy packing that
re-measures each candidate chunk with the real tokenizer. I also dropped one paper
the spec listed (`q_graddesc`) after it hit a deterministic Apple-MPS crash in the
OCR model, landing at 11 papers (the `foundational-ml` cluster is still
represented by `pqc`).

---

## AI Usage

**Instance 1**

- _What I gave the AI:_ The Chunking Strategy and Documents sections of
  `planning.md`, plus the requirement that chunks must fit MiniLM's 256-token
  window and drop References.
- _What it produced:_ A `chunk.py` that followed the spec's token-ID
  sliding-window pseudocode and attached the `{source, cluster, year, section}`
  metadata.
- _What I changed or overrode:_ The sliding window produced chunks that exceeded
  256 tokens (decode→re-encode doesn't round-trip), so I directed a rewrite to a
  sentence-aware packer that re-measures with the real tokenizer. After inspecting
  real output I also added cleanup for Marker artifacts and two content filters
  (OCR repetition loops, bibliography chunks) the spec hadn't anticipated.

**Instance 2**

- _What I gave the AI:_ The Marker-based conversion plan from `planning.md` and the
  constraint to skip >50-page PDFs, plus the observation that CPU OCR was
  projecting ~3h.
- _What it produced:_ A working `convert.py`, then a GPU (MPS) variant that cut
  text-recognition from ~26 s/it to ~1–2 s/it.
- _What I changed or overrode:_ When the OCR model hit a deterministic MPS crash on
  one paper (`q_graddesc`, same step every run), I made the converter resumable
  with a per-PDF CPU fallback, then made the call to skip that paper entirely —
  ending with 11 papers.

---

## Stretch Features

### Metadata filtering

`retrieve` function supports ChromaDB metadata filters.

The filter changes the retrieved sources in a visible way. For example, for the question _"What drives the system to a steady state?"_:

| Filter                    | Top sources                                 |
| ------------------------- | ------------------------------------------- |
| _(none)_                  | `qnn_collide_17`, `classifier_19`, `qrc_24` |
| `reservoir-computing`     | `qrc_24` only                               |
| `open-system-classifiers` | `qnn_collide_17`, `classifier_19` only      |

### Hybrid search: BM25 + semantic

I added BM25 lexical search over the same 721 chunks and combined it with semantic ranking using **Reciprocal Rank Fusion**:

`score = Σ 1/(60 + rank)`

RRF is rank-based, so it does not require score normalization. The UI exposes this through a **Retrieval mode** radio button with `semantic` and `hybrid` options. The implementation is in `hybrid.py`.

Top 5 sources by method:

| Q                   | semantic                         | BM25                                                         | hybrid                                            |
| ------------------- | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| Q1 boson reservoir  | `boson_reservoir_25`, `qrp_24`   | `boson_reservoir_25`                                         | `boson_reservoir_25`, `qrp_24`                    |
| Q2 barren plateaus  | `dqnn_train_22`, `qrc_24`        | `dqnn_train_22`, `qrc_24`                                    | `dqnn_train_22`, `qrc_24`                         |
| Q5 which classifier | `classifier_19`, `classifier_23` | `classifier_19`, `classifier_22`, `dqnn_train_22`, `dqnn_20` | `classifier_23`, `classifier_19`, `classifier_22` |

**Finding: hybrid search does not fix the Q2 failure.** I expected BM25 to surface Beer's "no barren plateau" chunk from `dqnn_20`, but it only ranks around #10 in BM25. The query _"suffer from barren plateaus"_ matches the affirmative paper, `dqnn_train_22`, both lexically and semantically. Beer phrases her result as the _absence_ of a plateau, so both retrieval methods under-rank it. Fusing two weak rankings does not recover the correct source.

Hybrid search does help on exact-term queries, such as _"boson sampling interferometer"_.

### Chunking-strategy comparison (+1)

I compared three chunking strategies on the 5 evaluation queries, each in its own throwaway collection using `chunk_compare.py`:

- **Section-aware**: baseline, 721 chunks
- **Fixed-window**: 256-token windows, 1010 chunks, +40%
- **Semantic**: sentence-embedding breakpoints, 1564 chunks, +117%

✓ means the top-1 result is the expected paper.

| Q                 | section-aware          | fixed-window           | semantic                |
| ----------------- | ---------------------- | ---------------------- | ----------------------- |
| Q1                | `boson_reservoir_25` ✓ | `boson_reservoir_25` ✓ | `qrp_24` ✗ (correct #4) |
| Q2 _(both sides)_ | yes@1, no@**8**        | yes@2, no@23           | yes@3, no@24            |
| Q3                | open-system ✓          | open-system ✓          | open-system ✓           |
| Q4                | `dqnn_train_22` ✗ (#2) | `dqnn_train_22` ✗ (#2) | `dqnn_20` ✓             |
| Q5                | `classifier_19` ✓      | `pqc` ✗ (#3)           | `classifier_22` ✗ (#7)  |

**Finding: section-aware chunking works best overall, but the result is mixed.** Lower distance does not always mean better retrieval. The semantic strategy produces the lowest distances, but it returns the wrong paper for Q1 and Q5. In Q5, it pushes `classifier_19` down to rank 7 because it removes the headings that help distinguish the similar `classifier_19`, `classifier_22`, and `classifier_23` papers.

Semantic chunking does win on Q4. However, **no chunking strategy fixes Q2**. `dqnn_20` still stays outside the top 5, ranking 8, 23, and 24 across the three strategies. Section-aware chunking keeps it closest.

Overall, the Q2 issue is a ranking problem, not a chunking problem. It is not fixed by lexical fusion either. Section-aware chunking also has a practical advantage: it preserves section labels, which makes citations clearer.
