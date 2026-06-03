"""Milestone 3 — Section-aware chunking of documents/*.md.

Strategy (see planning.md):
  1. Tokenize with the actual MiniLM tokenizer (true token counts, not char estimate).
  2. Split each Markdown doc on headings (##, ###, ...) into sections.
  3. Drop References / Bibliography sections.
  4. For each section:
       - token_count <= MAX_TOKENS  -> emit as one chunk
       - else                       -> sliding-window sub-split (256 tok, 50 overlap)
  5. Skip empty chunks.

Every chunk carries metadata: id, text, source, section, cluster, year.

The 256-token cap is set BY the model: all-MiniLM-L6-v2 has a 256-token input
window and silently truncates longer inputs.

Importable:  from chunk import build_chunks
CLI:         python chunk.py        (builds, asserts, prints 5 random chunks)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

DOCS_DIR = Path("documents")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256
OVERLAP = 50

# --- Cluster + year lookup (from planning.md) -------------------------------
META = {
    "qnn_collide_17":     ("open-system-classifiers", 2017),
    "classifier_19":      ("open-system-classifiers", 2019),
    "classifier_22":      ("open-system-classifiers", 2022),
    "classifier_23":      ("open-system-classifiers", 2023),
    "dqnn_20":            ("qnn-trainability",        2020),
    "dqnn_train_22":      ("qnn-trainability",        2022),
    "barren_plat_25":     ("qnn-trainability",        2025),
    "boson_reservoir_25": ("reservoir-computing",     2025),
    "qrc_24":             ("reservoir-computing",     2024),
    "qrp_24":             ("reservoir-computing",     2024),
    "pqc":                ("foundational-ml",         2019),
}

_tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# Match markdown ATX headings: leading #, ##, ### ... up to 6.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.MULTILINE)
_REFERENCES_RE = re.compile(r"references?|bibliography|acknowledg", re.IGNORECASE)

# --- Markup / citation noise cleaners (Marker output artifacts) -------------
# Image embeds:  ![](_page_1_Figure_1.jpeg)  or  ![alt](path)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# HTML span anchors:  <span id="page-1-1"></span>
_SPAN_RE = re.compile(r"</?span[^>]*>")
# Superscript/subscript tags:  <sup>5</sup>, <sub>i</sub>
_SUPSUB_RE = re.compile(r"</?su[bp]>")
# Markdown links — keep the visible text, drop the (url) part. Handles the
# escaped-bracket citation form Marker emits, e.g. [\[1\]](#page-14-0) -> [1].
_MDLINK_RE = re.compile(r"\[([^\]]*)\]\((?:#|https?:|mailto:)[^)]*\)")
# Leftover escaped brackets/parens from citation text:  \[ \] \( \)
_ESC_BRACKET_RE = re.compile(r"\\([\[\]()])")
# Inline numeric citation markers like [12] or [1, 3] or [1–7] (not md links).
# Also catches residue like "[ 292, ]" or "[ 294 – ]" left when some inner
# numbers were already stripped — i.e. brackets holding only digits/punctuation.
_CITATION_RE = re.compile(r"\[\s*\d[\d\s,–\-]*\](?!\()")
# Empty or punctuation-only brackets left after the above:  [ ] , [ , ]
_EMPTY_BRACKET_RE = re.compile(r"\[\s*[,–\-]*\s*\]")
# Bare internal anchors left behind:  (#page-2-0)
_ANCHOR_RE = re.compile(r"\(#[^)]*\)")


def token_count(text: str) -> int:
    return len(_tok.encode(text, add_special_tokens=False))


def _clean(text: str) -> str:
    """Strip Marker markup artifacts and inline citation markers.

    The order matters: resolve markdown links to their visible text first,
    then remove the numeric citation markers that text may contain.
    """
    text = _IMAGE_RE.sub("", text)
    text = _MDLINK_RE.sub(r"\1", text)      # [text](url) -> text
    text = _ESC_BRACKET_RE.sub(r"\1", text)  # \[ -> [ , \] -> ]
    text = _ANCHOR_RE.sub("", text)
    text = _SPAN_RE.sub("", text)
    text = _SUPSUB_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = _EMPTY_BRACKET_RE.sub("", text)
    # Tidy spacing left by removals.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:)])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_title(title: str) -> str:
    """Strip span anchors / markdown links from a heading to get clean text."""
    title = _SPAN_RE.sub("", title)
    title = _MDLINK_RE.sub(r"\1", title)
    title = _SUPSUB_RE.sub("", title)
    title = _ANCHOR_RE.sub("", title)
    title = re.sub(r"\s{2,}", " ", title)
    return title.strip() or "Untitled"


def split_sections(md_text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, body) pairs by ATX headings.

    Content before the first heading is grouped under a synthetic 'Preamble'
    section (covers title + abstract that some papers emit before any heading).
    """
    matches = list(_HEADING_RE.finditer(md_text))
    sections: list[tuple[str, str]] = []

    if not matches:
        return [("Body", md_text)]

    # Text before the first heading.
    pre = md_text[: matches[0].start()].strip()
    if pre:
        sections.append(("Preamble", pre))

    for i, m in enumerate(matches):
        title = _clean_title(m.group(2))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        sections.append((title, body))

    return sections


def _hard_split(text: str) -> list[str]:
    """Last-resort splitter for a single unit that alone exceeds MAX_TOKENS.

    Splits on word boundaries and verifies each piece against the real
    tokenizer, so the 256-token cap holds even for pathological inputs
    (long equations, URL-like strings) with no sentence punctuation.
    """
    words = text.split()
    pieces: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if token_count(" ".join(cur)) > MAX_TOKENS:
            cur.pop()
            if cur:
                pieces.append(" ".join(cur))
            # If a single word is somehow > MAX_TOKENS, keep it alone.
            cur = [w] if token_count(w) <= MAX_TOKENS else []
            if not cur:
                pieces.append(w[:2000])
    if cur:
        pieces.append(" ".join(cur))
    return pieces


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_long_section(text: str) -> list[str]:
    """Pack an over-long section into <=MAX_TOKENS chunks on sentence boundaries.

    Greedy sentence packing keeps chunks as complete thoughts. A trailing
    OVERLAP-token tail from each emitted chunk is prepended to the next so a
    claim spanning a boundary stays retrievable. Every emitted chunk is
    re-measured with the real tokenizer, so the 256 cap is guaranteed.
    """
    sentences = _SENT_SPLIT_RE.split(text)
    chunks: list[str] = []
    cur = ""

    def overlap_tail(s: str) -> str:
        ids = _tok.encode(s, add_special_tokens=False)
        if len(ids) <= OVERLAP:
            return s
        return _tok.decode(ids[-OVERLAP:], skip_special_tokens=True)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # A single sentence longer than the window must be hard-split.
        if token_count(sent) > MAX_TOKENS:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(sent))
            continue

        candidate = f"{cur} {sent}".strip() if cur else sent
        if token_count(candidate) <= MAX_TOKENS:
            cur = candidate
        else:
            chunks.append(cur)
            cur = f"{overlap_tail(cur)} {sent}".strip()
            # Overlap may push us over; if so, drop the overlap for this chunk.
            if token_count(cur) > MAX_TOKENS:
                cur = sent

    if cur.strip():
        chunks.append(cur)
    return chunks


def _is_degenerate(text: str) -> bool:
    """Detect OCR repetition-loop garbage (e.g. 'the second of the second of...').

    Marker occasionally emits a degenerate decode where a short phrase repeats
    for the whole chunk. Such chunks carry almost no information and pollute
    retrieval, so we drop them. Heuristic: for a reasonably long chunk, a very
    low unique-word ratio means near-total repetition.
    """
    words = text.split()
    if len(words) < 30:
        return False
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    return unique_ratio < 0.18


# A bibliography chunk: many publication years and/or numbered citation items,
# and/or a high density of journal abbreviations. Real prose almost never hits
# these densities, so this catches reference lists that slipped past the heading
# filter (some papers merge the bibliography under an Appendix heading with no
# "References" heading of its own).
_PUB_YEAR_RE = re.compile(r"\((?:19|20)\d{2}\)")          # (2019)
_BARE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")          # 2019
_CITE_ITEM_RE = re.compile(r"(?m)^\s*-?\s*\[\d+\]\s")      # [12] ...
_JOURNAL_RE = re.compile(
    r"Phys\.\s*Rev\.|Nat\.\s*Commun\.|Nature|arXiv:|npj|"
    r"J\.\s*High\s*Energ|Rev\.\s*Mod\.\s*Phys\.|Quantum\s|PRX",
)


def _is_reference_chunk(text: str) -> bool:
    pub_years = len(_PUB_YEAR_RE.findall(text))
    bare_years = len(_BARE_YEAR_RE.findall(text))
    cite_items = len(_CITE_ITEM_RE.findall(text))
    journals = len(_JOURNAL_RE.findall(text))
    # Bibliography fingerprint: several years AND several journal markers, or an
    # unambiguous run of numbered citation items.
    if cite_items >= 3:
        return True
    if pub_years >= 3:
        return True
    if bare_years >= 4 and journals >= 3:
        return True
    return False


def chunk_document(md_text: str, stem: str) -> list[dict]:
    cluster, year = META.get(stem, ("uncategorized", 0))
    chunks: list[dict] = []

    for title, body in split_sections(md_text):
        if _REFERENCES_RE.search(title):
            continue
        body = _clean(body)
        if not body:
            continue

        section_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "section"

        if token_count(body) <= MAX_TOKENS:
            pieces = [body]
        else:
            pieces = _split_long_section(body)

        for piece in pieces:
            if not piece.strip() or _is_degenerate(piece) or _is_reference_chunk(piece):
                continue
            idx = len(chunks)
            chunks.append(
                {
                    "id": f"{stem}__{section_slug}__{idx:03d}",
                    "text": piece,
                    "source": stem,
                    "section": title,
                    "cluster": cluster,
                    "year": year,
                }
            )
    return chunks


def build_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No .md files in {docs_dir}/ — run convert.py first."
        )

    all_chunks: list[dict] = []
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        doc_chunks = chunk_document(text, md_path.stem)
        all_chunks.extend(doc_chunks)
        print(f"{md_path.stem:24s} -> {len(doc_chunks):4d} chunks")
    return all_chunks


def _sanity_checks(chunks: list[dict]) -> None:
    over = [c["id"] for c in chunks if token_count(c["text"]) > MAX_TOKENS]
    assert not over, f"chunks exceed {MAX_TOKENS}-token window: {over[:5]}"
    assert 50 <= len(chunks) <= 2000, f"unexpected chunk count: {len(chunks)}"
    assert all(c["text"].strip() for c in chunks), "empty chunk found"


def main() -> None:
    import random

    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    _sanity_checks(chunks)
    print("Sanity checks passed: all <=256 tokens, count in [50, 2000], no empties.")

    token_lens = [token_count(c["text"]) for c in chunks]
    print(
        f"Token length — min {min(token_lens)}, "
        f"max {max(token_lens)}, mean {sum(token_lens) / len(token_lens):.1f}"
    )

    print("\n--- 5 random chunks ---")
    random.seed(42)
    for c in random.sample(chunks, 5):
        print(f"\n[{c['id']}]  ({token_count(c['text'])} tok)")
        print(f"source={c['source']} | section={c['section']} | "
              f"cluster={c['cluster']} | year={c['year']}")
        preview = c["text"][:400]
        print(preview + ("..." if len(c["text"]) > 400 else ""))


if __name__ == "__main__":
    sys.exit(main())
