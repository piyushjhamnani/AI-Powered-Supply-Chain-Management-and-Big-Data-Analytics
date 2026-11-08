"""Document ingestion + text extraction for PDF, TXT and CSV."""
import io
import os
from dataclasses import dataclass

import pandas as pd

from rag.config import CSV_ROWS_PER_CHUNK, SUPPORTED


class DocumentError(Exception):
    """Raised for unreadable / unsupported / empty documents (message is shown to the user)."""


@dataclass
class Segment:
    text: str
    ref: str     # human-readable location, e.g. "page 3" or "rows 16-30"


def extract(name: str, data: bytes) -> list[Segment]:
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED:
        raise DocumentError(f"'{name}': unsupported file type (use PDF, TXT or CSV).")
    if not data:
        raise DocumentError(f"'{name}': file is empty.")
    try:
        segs = {".pdf": _pdf, ".txt": _txt, ".csv": _csv}[ext](data)
    except DocumentError:
        raise
    except Exception as e:
        raise DocumentError(f"'{name}': could not be read ({type(e).__name__}: {e}).") from e
    segs = [s for s in segs if s.text.strip()]
    if not segs:
        raise DocumentError(f"'{name}': no readable text found (scanned PDFs need OCR, which is not supported).")
    return segs


def _pdf(data: bytes) -> list[Segment]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise DocumentError("PDF is password-protected.")
    return [Segment(p.extract_text() or "", f"page {i}") for i, p in enumerate(reader.pages, 1)]


def _txt(data: bytes) -> list[Segment]:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return [Segment(data.decode(enc), "text")]
        except UnicodeError:
            continue
    raise DocumentError("unknown text encoding.")


def _csv(data: bytes) -> list[Segment]:
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    if df.empty:
        return []
    cols = list(df.columns)
    out = []
    for start in range(0, len(df), CSV_ROWS_PER_CHUNK):
        part = df.iloc[start:start + CSV_ROWS_PER_CHUNK]
        # "column: value" rows read much better to an embedding model than raw comma-separated text
        lines = ["; ".join(f"{c}: {v}" for c, v in zip(cols, row) if v != "") for row in part.itertuples(index=False)]
        out.append(Segment("\n".join(lines), f"rows {start + 1}-{start + len(part)}"))
    return out
