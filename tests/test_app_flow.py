"""End-to-end UI test with Streamlit's AppTest: every sidebar page loads, and the Knowledge Assistant works
inside 'Stream analytics'. Uses the offline hash embedder (see test_rag.py) instead of downloading the model.
Run: python tests/test_app_flow.py"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); os.chdir(ROOT)
for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
    os.environ.pop(k, None)

import rag.store as store
from test_rag import HashEmbedder
from streamlit.testing.v1 import AppTest

MODE = sys.argv[1] if len(sys.argv) > 1 else "stub"
if MODE == "stub":
    store.Embedder = HashEmbedder
shutil.rmtree(os.path.join(ROOT, "rag_index"), ignore_errors=True)

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300).run()
assert not at.exception, at.exception
pages = list(at.sidebar.radio[0].options)
print("sidebar:", pages)
assert "Stream analytics" in pages and not any("BDSMA" in p for p in pages) and len(pages) == 10
for p in pages:
    at.sidebar.radio[0].set_value(p).run()
    assert not at.exception, (p, at.exception)
    print("  page ok:", p, "| title:", at.title[0].value if at.title else "-")

at.sidebar.radio[0].set_value("Stream analytics").run()
assert any(s.value == "Knowledge Assistant" for s in at.subheader), "Knowledge Assistant section missing"
at.button(key="kbq0").click().run()                       # sample question button
assert not at.exception, at.exception
msgs = [m for m in at.chat_message]
print("chat messages:", len(msgs))
text = " ".join(str(m.markdown[0].value) for m in msgs if m.markdown)
print(text[:900])
assert len(msgs) == 2
# existing stream functionality still works alongside it
at.button[0].click().run()
assert not at.exception, at.exception
print("stream batch generated; KPI html present:", any("events processed" in m.value for m in at.markdown))
shutil.rmtree(os.path.join(ROOT, "rag_index"), ignore_errors=True)
print("ALL OK" if MODE == "stub" else "REAL-MODE RUN DONE")
