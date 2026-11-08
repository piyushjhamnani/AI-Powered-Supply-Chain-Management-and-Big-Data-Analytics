"""Split extracted text into overlapping, sentence-aware chunks."""
import re

from rag.config import CHUNK_CHARS, CHUNK_OVERLAP
from rag.loader import Segment

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    pieces = []
    for p in _SPLIT.split(text):
        p = p.strip()
        while len(p) > size:            # a single very long sentence: hard split
            pieces.append(p[:size]); p = p[size - overlap:]
        if p:
            pieces.append(p)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail    # start overlap on a word boundary
            cur = f"{tail} {p}"
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_segments(segments: list[Segment]) -> list[tuple[str, str]]:
    """-> [(chunk_text, source_ref)]. Each segment (PDF page, CSV row group) is chunked on its own,
    so every chunk keeps an exact page / row reference for citations."""
    out = []
    for s in segments:
        out += [(c, s.ref) for c in chunk_text(s.text)]
    return out
