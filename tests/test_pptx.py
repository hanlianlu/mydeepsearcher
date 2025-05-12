"""
tests/test_pptx_loader.py
Quick sanity test for AdvancedPPTXLoader (EasyOCR version).

What it does
------------
1. Loads a given .pptx file.
2. Prints total slide count, total document‑chunks, and
   the first 250 characters of up to three chunks.
3. Reports whether OCR was actually used.

Usage
-----
python tests/test_pptx_loader.py /path/to/deck.pptx

If no path is supplied, a tiny sample deck (`sample.pptx`) must be located
in the same directory as this script.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path
from typing import Sequence

from deepsearcher.loader.file_loader.pptx_loader import AdvancedPPTXLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------- #
def preview_chunks(docs: Sequence, preview_chars: int = 250) -> None:
    for i, doc in enumerate(docs[:3], 1):
        preview = textwrap.shorten(
            doc.page_content.replace("\n", " "),
            width=preview_chars,
            placeholder=" …",
        )
        logger.info("Chunk %d\n  meta: %s\n  text: %s", i, doc.metadata, preview)


def main() -> None:
    # ---- arg parse ---------------------------------------- #
    if len(sys.argv) > 1:
        pptx_path = Path(sys.argv[1])
    else:
        pptx_path = Path(__file__).with_name("PPA.pptx")

    if not pptx_path.is_file():
        logger.error("PPTX file not found: %s", pptx_path)
        sys.exit(1)

    # ---- instantiate loader ------------------------------- #
    loader = AdvancedPPTXLoader(
        ocr_images=True,      # enable OCR
        languages=["en"],     # English OCR
        chunk_size=3000,      # demonstrate in‑loader chunking
        chunk_overlap=400,
    )

    # ---- load deck ---------------------------------------- #
    docs = loader.load_pptx(str(pptx_path))

    if not docs:
        logger.error("No content extracted from %s", pptx_path)
        sys.exit(1)

    # ---- diagnostics -------------------------------------- #
    slides = {d.metadata["slide"] for d in docs}
    ocr_chunks = [d for d in docs if "OCR" in d.page_content]

    logger.info("Slides detected         : %d", len(slides))
    logger.info("Document chunks created : %d", len(docs))
    logger.info("Chunks with OCR content : %d", len(ocr_chunks))
    logger.info("-" * 60)
    preview_chunks(docs)

    logger.info("PPTX loader test finished OK")


if __name__ == "__main__":
    main()
