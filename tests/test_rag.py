"""Run:  python -m pytest tests -q      (or: python tests/test_rag.py)

Uses a small deterministic hashing embedder so tests run offline and fast.
The real all-MiniLM-L6-v2 model is exercised by tests/smoke_real_model.py (needs internet once).
"""
import hashlib
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import assistant, chunker, config as C, loader, service
from rag.loader import DocumentError
from rag.store import KnowledgeBase


class HashEmbedder:
    """Bag-of-words hashing embedder (stand-in for Sentence Transformers in offline tests)."""
    model_name, calls = "test-hash", 0

    def encode(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), 384), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


def make_pdf(pages):
    from fpdf import FPDF
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=11); pdf.multi_cell(0, 6, t)
    return bytes(pdf.output())


def new_kb(d, emb=None):
    return KnowledgeBase(embedder=emb or HashEmbedder(), index_dir=d)


# ------------------------------------------------------------------ ingestion / extraction
def test_extract_txt_csv_pdf():
    assert "reorder" in loader.extract("a.txt", b"We reorder weekly.")[0].text
    segs = loader.extract("a.csv", b"sku,qty\nA,1\nB,2\n")
    assert "sku: A; qty: 1" in segs[0].text and segs[0].ref == "rows 1-2"
    segs = loader.extract("a.pdf", make_pdf(["Page one about milk.", "Page two about rice."]))
    assert [s.ref for s in segs] == ["page 1", "page 2"] and "rice" in segs[1].text


def test_invalid_documents():
    for name, data in [("a.docx", b"x"), ("a.txt", b""), ("a.pdf", b"not a pdf"), ("a.csv", b"col\n"),
                       ("a.txt", b"   \n  "), ("a.pdf", make_pdf([" "]))]:
        try:
            loader.extract(name, data)
        except DocumentError:
            continue
        raise AssertionError(f"{name} should have been rejected")


# ------------------------------------------------------------------ chunking
def test_chunking_overlap_and_size():
    text = " ".join(f"Sentence number {i} talks about warehouse stock levels." for i in range(60))
    ch = chunker.chunk_text(text)
    assert len(ch) > 3 and all(len(c) <= C.CHUNK_CHARS + C.CHUNK_OVERLAP + 20 for c in ch)
    # overlap: the start of each chunk repeats the end of the previous one
    for a, b in zip(ch, ch[1:]):
        assert b.split()[1] in a, "chunks should overlap"
    assert chunker.chunk_text("short") == ["short"] and chunker.chunk_text("  ") == []
    assert len(chunker.chunk_text("x" * 5000)) >= 6      # no sentence boundaries: hard split still works


def test_segment_refs_kept():
    pairs = chunker.chunk_segments(loader.extract("a.pdf", make_pdf(["alpha " * 5, "beta " * 5])))
    assert [r for _, r in pairs] == ["page 1", "page 2"]


# ------------------------------------------------------------------ FAISS store
def test_index_search_persist_remove():
    with tempfile.TemporaryDirectory() as d:
        emb = HashEmbedder(); kb = new_kb(d, emb)
        assert kb.search("anything") == []                                   # empty KB
        kb.add_document("milk.txt", b"Toned milk is perishable with a three day shelf life. Order milk daily.")
        kb.add_document("rice.txt", b"Basmati rice is a staple with a one year shelf life and weekly review.")
        hits = kb.search("shelf life of milk", k=2)
        assert hits and hits[0]["doc"] == "milk.txt" and hits[0]["score"] > C.MIN_SCORE
        assert kb.search("quantum chromodynamics gluons") == []              # nothing relevant -> empty
        calls = emb.calls
        assert kb.add_document("milk.txt", b"Toned milk is perishable with a three day shelf life. Order milk daily.") == 0
        assert emb.calls == calls                                            # unchanged file: no re-embedding
        # persistence: a fresh instance loads the index from disk and embeds nothing at start-up
        emb2 = HashEmbedder(); kb2 = new_kb(d, emb2)
        assert kb2.stats()["documents"] == 2 and kb2.index.ntotal == len(kb2.chunks)
        assert kb2.search("shelf life of milk")[0]["doc"] == "milk.txt" and emb2.calls == 1  # only the query
        # changed file replaces old chunks
        kb2.add_document("milk.txt", b"Milk policy changed: order milk twice daily from the dairy cooperative.")
        assert "twice daily" in kb2.search("order milk")[0]["text"]
        assert sum(c["doc"] == "milk.txt" for c in kb2.chunks) == 1
        kb2.remove_document("rice.txt")
        assert kb2.stats()["documents"] == 1 and kb2.index.ntotal == len(kb2.chunks)
        # a different embedding model invalidates the stored index (vectors are not comparable)
        other = HashEmbedder(); other.model_name = "other"
        assert new_kb(d, other).stats()["documents"] == 0


def test_corrupt_index_recovers():
    with tempfile.TemporaryDirectory() as d:
        kb = new_kb(d); kb.add_document("a.txt", b"inventory policy text here")
        open(os.path.join(d, "index.faiss"), "wb").write(b"garbage")
        assert new_kb(d).stats()["documents"] == 0                           # no crash; rebuilt by sync()


def test_sync_folder_and_errors():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as src:
        open(os.path.join(src, "ok.txt"), "w").write("vendor delivery policy lead time days")
        open(os.path.join(src, "bad.pdf"), "wb").write(b"junk")
        kb = new_kb(d)
        rep = kb.sync({"ok.txt": os.path.join(src, "ok.txt"), "bad.pdf": os.path.join(src, "bad.pdf"),
                       "missing.txt": os.path.join(src, "missing.txt")})
        assert rep["added"] == ["ok.txt"] and set(rep["errors"]) == {"bad.pdf", "missing.txt"}
        assert kb.sync({"ok.txt": os.path.join(src, "ok.txt")})["skipped"] == ["ok.txt"]
        kb.sync({})                                                          # file vanished -> removed
        assert kb.stats()["documents"] == 0


def test_project_kb_files_indexed_and_csv_rows():
    files = service.kb_files()
    assert any(k.startswith("dataset/vendors.csv") for k in files) and "supply_chain_risk_playbook.txt" in files
    with tempfile.TemporaryDirectory() as d:
        kb = new_kb(d); rep = kb.sync(files)
        assert not rep["errors"], rep
        hit = kb.search("Nashik Satpur Warehouse capacity units")[0]
        assert "Nashik" in hit["text"] and hit["ref"]


# ------------------------------------------------------------------ assistant
def _kb_with_policy(d):
    kb = new_kb(d)
    kb.add_document("policy.txt", b"Purchase orders below 50000 rupees are auto approved by Autopilot.")
    return kb


def test_no_info_does_not_call_llm():
    calls = []
    orig = assistant.explainer.ask
    assistant.explainer.ask = lambda *a, **k: calls.append(1) or "x"
    try:
        with tempfile.TemporaryDirectory() as d:
            r = assistant.answer("Which vendor insurance certificate number applies?", _kb_with_policy(d), data={}, state=None, result=None)
    finally:
        assistant.explainer.ask = orig
    assert r["mode"] == "no_info" and r["text"] == "Insufficient information in the knowledge base." and not calls and not r["sources"]


def test_llm_answer_prompt_and_sources():
    seen = {}
    orig = assistant.explainer.ask
    assistant.explainer.ask = lambda prompt, max_tokens=0, system="": seen.update(p=prompt, s=system) or "Auto-approved below ₹50,000 [D1]."
    try:
        with tempfile.TemporaryDirectory() as d:
            r = assistant.answer("Which purchase orders are auto approved?", _kb_with_policy(d), data={}, state=None, result=None)
    finally:
        assistant.explainer.ask = orig
    assert r["mode"] == "llm" and r["sources"][0]["name"] == "policy.txt" and "[D1] policy.txt" in seen["p"]
    assert "Never invent" in seen["s"]


def test_llm_missing_or_failing_falls_back_to_evidence():
    orig = assistant.explainer.ask
    try:
        with tempfile.TemporaryDirectory() as d:
            kb = _kb_with_policy(d)
            assistant.explainer.ask = lambda *a, **k: None
            r = assistant.answer("auto approved purchase orders", kb, {}, None, None)
            assert r["mode"] == "extractive" and "50000" in r["text"] and r["warning"]
            assistant.explainer.ask = lambda *a, **k: "(AI unavailable: HTTPError. Showing template text instead.)"
            r = assistant.answer("auto approved purchase orders", kb, {}, None, None)
            assert r["mode"] == "extractive" and "HTTPError" in r["warning"] and "50000" in r["text"]
    finally:
        assistant.explainer.ask = orig


def test_embedding_model_failure_is_handled():
    from rag.store import EmbeddingError

    class Broken:
        model_name = "x"
        def encode(self, t): raise EmbeddingError("model offline")

    with tempfile.TemporaryDirectory() as d:
        kb = new_kb(d); kb.add_document("a.txt", b"auto approved purchase orders")
        kb.embedder = Broken()
        r = assistant.answer("auto approved purchase orders", kb, {}, None, None)
    assert r["mode"] == "no_info" and "model offline" in r["warning"]


# ------------------------------------------------------------------ structured data (real project data + agents)
def test_structured_facts_from_project_data():
    from core.state import load_data, State
    from agents.forecast_agent import ForecastAgent
    from agents.orchestrator import Orchestrator
    from rag.structured import live_facts, find_entities
    data = load_data()
    fa = ForecastAgent(data); fa.fit(data["sales"])
    state = State.from_data(data); res = Orchestrator(data, fa).run(state)
    ent = find_entities("milk stock in Pune from Sahyadri", data)
    assert ent == {"products": ["P001"], "warehouses": ["WH-PUN"], "vendors": ["V01"]}, ent
    f = dict(live_facts("Which items are at risk of stockout? What should I reorder?", data, state, res))
    assert "Live inventory risk (Risk Agent)" in f and "System inventory policy (config.py)" in f
    f = dict(live_facts("How much Toned Milk did we sell recently?", data, state, res))
    sales = [v for k, v in f.items() if k.startswith("Sales")][0]
    s = state.sales; s = s[(s.date > s.date.max() - __import__("pandas").Timedelta(days=28)) & (s.product_id == "P001")]
    assert str(int(s.units_sold.sum())) in sales.replace(",", "")           # matches pandas ground truth
    f = dict(live_facts("capacity of Pune warehouse", data, state, res))
    assert "40000" in list(f.values())[0]
    assert live_facts("hello there", data, state, res) == []


if __name__ == "__main__":
    fails = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                fails += 1; print("FAIL", n, type(e).__name__, e)
    sys.exit(fails)
