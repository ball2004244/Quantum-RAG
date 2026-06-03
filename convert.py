"""Milestone 3 — Offline PDF -> Markdown conversion.

Converts each in-scope PDF in `raw_pdfs/` to Markdown in `documents/` using
Marker (layout-aware, de-scrambles two-column journal layout). PDFs longer than
MAX_PAGES are skipped (e.g. the excluded PhD thesis).

Robustness:
  * Resumable — skips PDFs whose .md already exists (re-run after a crash).
  * MPS (Apple GPU) is used by default for speed, but the surya OCR model can
    intermittently crash on Metal with a bogus-index AcceleratorError. If a PDF
    fails on the GPU, it is automatically retried on CPU (slower but stable),
    so one flaky page never kills the whole batch.

Run:  python convert.py
      TORCH_DEVICE=cpu python convert.py     # force CPU for everything
      python convert.py --force              # re-convert even if .md exists
Marker downloads model weights (~1-2 GB) on first run and caches them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Device selection -------------------------------------------------------
# Use the Apple Silicon Metal (MPS) GPU when available; fall back to CPU.
# These env vars are read by torch/marker at import time, so set them first.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _select_device() -> str:
    if os.environ.get("TORCH_DEVICE"):
        return os.environ["TORCH_DEVICE"]
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = _select_device()
os.environ["TORCH_DEVICE"] = DEVICE

import pypdfium2  # noqa: E402  (bundled with marker-pdf)
from marker.converters.pdf import PdfConverter  # noqa: E402
from marker.models import create_model_dict  # noqa: E402
from marker.output import text_from_rendered  # noqa: E402

RAW_DIR = Path("raw_pdfs")
OUT_DIR = Path("documents")
MAX_PAGES = 50


def page_count(pdf_path: Path) -> int:
    doc = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def _build_converter(device: str) -> PdfConverter:
    models = create_model_dict(device=device)
    return PdfConverter(artifact_dict=models)


def convert_one(converter: PdfConverter, pdf_path: Path) -> str:
    rendered = converter(str(pdf_path))
    md_text, _ext, _images = text_from_rendered(rendered)
    return md_text


def main() -> None:
    force = "--force" in sys.argv
    OUT_DIR.mkdir(exist_ok=True)

    # Optional positional args: specific PDF stems to convert (e.g. "q_graddesc").
    # If none given, process every PDF in raw_pdfs/.
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]

    all_pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if wanted:
        pdfs = [p for p in all_pdfs if p.stem in wanted]
    else:
        pdfs = all_pdfs
    if not pdfs:
        print(f"No matching PDFs found in {RAW_DIR}/")
        return

    print(f"Loading Marker models on device: {DEVICE}")
    converter = _build_converter(DEVICE)
    cpu_converter = None  # built lazily only if an MPS conversion fails

    converted = skipped = failed = 0
    for pdf_path in pdfs:
        out = OUT_DIR / (pdf_path.stem + ".md")
        if out.exists() and not force:
            print(f"HAVE {pdf_path.name} -> {out.name} (skip; use --force to redo)")
            skipped += 1
            continue

        pages = page_count(pdf_path)
        if pages > MAX_PAGES:
            print(f"SKIP {pdf_path.name} ({pages} pages > {MAX_PAGES})")
            skipped += 1
            continue

        try:
            md_text = convert_one(converter, pdf_path)
        except Exception as exc:  # MPS AcceleratorError etc.
            print(f"WARN {pdf_path.name} failed on {DEVICE} ({type(exc).__name__}); "
                  f"retrying on CPU...")
            if cpu_converter is None:
                cpu_converter = converter if DEVICE == "cpu" else _build_converter("cpu")
            try:
                md_text = convert_one(cpu_converter, pdf_path)
            except Exception as exc2:
                print(f"FAIL {pdf_path.name}: {type(exc2).__name__}: {exc2}")
                failed += 1
                continue

        out.write_text(md_text, encoding="utf-8")
        print(f"OK   {pdf_path.name} ({pages} pages) -> {out.name}")
        converted += 1

    print(f"\nDone: {converted} converted, {skipped} skipped, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
