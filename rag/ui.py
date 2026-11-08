"""Streamlit section: 'Knowledge Assistant' (rendered inside the Stream analytics page)."""
import streamlit as st

from llm import explainer
from rag import config as C
from rag import guardrails, service
from rag.assistant import answer
from rag.loader import DocumentError
from rag.store import EmbeddingError, KnowledgeBase

SAMPLE_QUESTIONS = [
    "Which products are at risk of stockout and what should I reorder?",
    "What is our safety stock and service level policy?",
    "Which vendor is the most reliable, and which is getting worse?",
    "How much did we sell of Toned Milk in the last 28 days?",
    "What is the capacity and current stock level of the Pune warehouse?",
    "When can stock be transferred between warehouses?",
]


@st.cache_resource(show_spinner="Indexing knowledge base (first run downloads the embedding model)…")
def get_kb():
    kb = KnowledgeBase()
    report = {"added": [], "skipped": [], "errors": {}}
    try:
        report = kb.sync(service.kb_files())     # only new/changed files are embedded
    except EmbeddingError as e:
        report["errors"]["embedding model"] = str(e)
    return kb, report


def _sources(srcs):
    if not srcs:
        return
    with st.expander(f"Sources ({len(srcs)})"):
        for s in srcs:
            score = f" · similarity {s['score']:.2f}" if s["score"] is not None else ""
            st.markdown(f"**[{s['label']}]** {s['name']} — _{s['ref']}_{score}")


def _notice(m):
    """Guardrail blocks are shown as a visible warning; other messages stay as a small caption."""
    if m.get("blocked"):
        st.warning(f"🛡️ Blocked by guardrail ({m.get('guardrail')}): {m.get('warning')}")
    elif m.get("warning"):
        st.caption(f"⚠️ {m['warning']}")
    if m.get("note"):
        st.caption(m["note"])


def render(data, S, R):
    ss = st.session_state
    st.divider()
    st.subheader("Knowledge Assistant")
    st.markdown('<p class="note">Ask questions in plain English. Answers combine your documents (PDF, TXT, CSV — '
                'searched with embeddings + FAISS) with live stock, sales and vendor data, and always show their sources. '
                'If the knowledge base has no answer, it says so.</p>', unsafe_allow_html=True)
    if not explainer.provider():
        st.info("No AI key found, so answers show the retrieved evidence instead of a written summary. "
                "Add GROQ_API_KEY or GEMINI_API_KEY to .env to enable written answers.")

    try:
        kb, report = get_kb()
    except Exception as e:     # e.g. faiss / sentence-transformers not installed
        st.error(f"Knowledge Assistant could not start: {type(e).__name__}: {e}. "
                 "Run `pip install -r requirements.txt` and restart.")
        return
    for name, msg in report["errors"].items():
        st.warning(f"{name}: {msg}")

    # ------------------------------------------------ knowledge base: upload + list
    stats = kb.stats()
    with st.expander(f"Knowledge base · {stats['documents']} documents · {stats['chunks']} chunks"):
        files = st.file_uploader("Add your own documents", type=[e.strip('.') for e in C.SUPPORTED],
                                 accept_multiple_files=True, key=f"kb_upload_{ss.get('kb_upload_gen', 0)}")
        if files and st.button("Add to knowledge base", type="primary"):
            with st.spinner("Extracting, chunking and embedding…"):
                for f in files:
                    data_bytes = f.getvalue()
                    check = guardrails.validate_upload(f.name, data_bytes)
                    if not check.ok:
                        st.warning(f"🛡️ Upload blocked ({check.guardrail}): {check.message}")
                        continue
                    try:
                        n = kb.add_document(f"uploads/{f.name}", data_bytes)
                        service.save_upload(f.name, data_bytes)
                        st.success(f"{f.name}: {n} chunks indexed." if n else f"{f.name}: already in the knowledge base.")
                    except (DocumentError, EmbeddingError) as e:
                        st.error(str(e))
            ss.kb_upload_gen = ss.get("kb_upload_gen", 0) + 1     # clears the uploader on next run
        for doc, n in stats["per_doc"].items():
            c1, c2 = st.columns([6, 1])
            c1.caption(f"📄 {doc} · {n} chunks")
            if doc.startswith("uploads/") and c2.button("Remove", key=f"rm_{doc}"):
                kb.remove_document(doc)
                service.delete_upload(doc)
                st.rerun()

    # ------------------------------------------------ chat
    ss.setdefault("kb_chat", [])
    q = None
    cols = st.columns(3)
    for i, text in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(text, key=f"kbq{i}", width="stretch"):
            q = text
    for m in ss.kb_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            _notice(m)
            _sources(m.get("sources"))
    q = st.chat_input("Ask the Knowledge Assistant…", key="kb_input") or q
    if q:
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base…"):
                res = answer(q, kb, data, S, R)
            st.markdown(res["text"])
            _notice(res)
            _sources(res["sources"])
        ss.kb_chat += [{"role": "user", "content": q},
                       {"role": "assistant", "content": res["text"], "sources": res["sources"], "warning": res["warning"],
                        "blocked": res.get("blocked"), "guardrail": res.get("guardrail"), "note": res.get("note")}]
    if ss.kb_chat and st.button("Clear conversation", key="kb_clear"):
        ss.kb_chat = []
        st.rerun()
