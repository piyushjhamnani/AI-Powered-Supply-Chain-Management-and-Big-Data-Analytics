"""Embeddings (Sentence Transformers) + FAISS vector index, persisted to disk."""
import hashlib
import json
import os

import numpy as np

from rag import config as C
from rag.chunker import chunk_segments
from rag.loader import DocumentError, extract


class EmbeddingError(Exception):
    pass


class Embedder:
    """Lazy wrapper so the model loads once, on first use (not at app start-up)."""

    def __init__(self, model_name: str = C.EMBED_MODEL):
        self.model_name, self._m = model_name, None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._m is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._m = SentenceTransformer(self.model_name)
            except Exception as e:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}' ({type(e).__name__}). The first run needs "
                    "internet access to download it (~90 MB); after that it is cached locally.") from e
        v = self._m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")


class KnowledgeBase:
    def __init__(self, embedder=None, index_dir: str = C.INDEX_DIR):
        import faiss
        self.faiss = faiss
        self.embedder = embedder or Embedder()
        self.index_dir = index_dir
        self.chunks: list[dict] = []        # [{doc, ref, text}]  (row i  <->  vector i in FAISS)
        self.docs: dict[str, str] = {}      # doc name -> content hash
        self.index = None
        self._load()

    # ---------------------------------------------------------------- persistence
    @property
    def _paths(self):
        return os.path.join(self.index_dir, "index.faiss"), os.path.join(self.index_dir, "meta.json")

    def _load(self):
        ip, mp = self._paths
        if not (os.path.exists(ip) and os.path.exists(mp)):
            return
        try:
            meta = json.load(open(mp, encoding="utf-8"))
            index = self.faiss.read_index(ip)
            if meta["model"] != self.embedder.model_name or index.ntotal != len(meta["chunks"]):
                raise ValueError("stale index")
            self.index, self.chunks, self.docs = index, meta["chunks"], meta["docs"]
        except Exception:      # corrupt / outdated index -> start clean; sync() re-creates it
            self.index, self.chunks, self.docs = None, [], {}

    def _save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        ip, mp = self._paths
        if self.index is not None:
            self.faiss.write_index(self.index, ip)
        json.dump({"model": self.embedder.model_name, "chunks": self.chunks, "docs": self.docs},
                  open(mp, "w", encoding="utf-8"))

    # ---------------------------------------------------------------- ingestion
    def add_document(self, name: str, data: bytes) -> int:
        """Extract -> chunk -> embed -> add to FAISS. Returns number of chunks added (0 if unchanged)."""
        h = hashlib.sha256(data).hexdigest()
        if self.docs.get(name) == h:
            return 0
        pairs = chunk_segments(extract(name, data))
        if not pairs:
            raise DocumentError(f"'{name}': nothing to index.")
        vecs = self.embedder.encode([t for t, _ in pairs])
        if name in self.docs:
            self._drop(name)
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(vecs.shape[1])    # inner product on unit vectors = cosine
        self.index.add(vecs)
        self.chunks += [dict(doc=name, ref=r, text=t) for t, r in pairs]
        self.docs[name] = h
        self._save()
        return len(pairs)

    def _drop(self, name: str):
        keep = [i for i, c in enumerate(self.chunks) if c["doc"] != name]
        vecs = np.vstack([self.index.reconstruct(i) for i in keep]) if keep else None
        self.index = self.faiss.IndexFlatIP(self.index.d)
        if vecs is not None:
            self.index.add(vecs.astype("float32"))
        self.chunks = [self.chunks[i] for i in keep]
        self.docs.pop(name, None)

    def remove_document(self, name: str):
        if name in self.docs:
            self._drop(name)
            self._save()

    def sync(self, files: dict[str, str]) -> dict:
        """files: {display name: path}. Indexes new/changed files, drops vanished ones.
        Unchanged files are skipped, so a restart does NOT re-embed anything."""
        report = {"added": [], "skipped": [], "errors": {}}
        for name, path in files.items():
            try:
                with open(path, "rb") as f:
                    n = self.add_document(name, f.read())
                (report["added"] if n else report["skipped"]).append(name)
            except (DocumentError, OSError) as e:
                report["errors"][name] = str(e)
        for gone in [d for d in self.docs if d not in files]:
            self.remove_document(gone)
        return report

    # ---------------------------------------------------------------- retrieval
    def search(self, query: str, k: int = C.TOP_K, min_score: float = C.MIN_SCORE) -> list[dict]:
        if self.index is None or self.index.ntotal == 0 or not query.strip():
            return []
        q = self.embedder.encode([query])
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [dict(self.chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0])
                if i >= 0 and s >= min_score]

    def stats(self) -> dict:
        per_doc = {}
        for c in self.chunks:
            per_doc[c["doc"]] = per_doc.get(c["doc"], 0) + 1
        return {"documents": len(self.docs), "chunks": len(self.chunks), "per_doc": per_doc}
