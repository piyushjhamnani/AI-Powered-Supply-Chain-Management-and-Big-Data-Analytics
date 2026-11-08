"""Run once on your machine (needs internet the first time to download all-MiniLM-L6-v2, ~90 MB):
    python tests/smoke_real_model.py
Indexes the starter knowledge base with the real embedding model and prints retrieval results."""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rag import service
from rag.store import KnowledgeBase

kb = KnowledgeBase(index_dir=tempfile.mkdtemp())
print(kb.sync(service.kb_files()))
for q in ["What is the safety stock policy?", "How should a delayed vendor delivery be handled?",
          "Who won the cricket world cup?"]:
    print("\nQ:", q)
    for h in kb.search(q, k=3):
        print(f"  {h['score']:.2f}  {h['doc']} ({h['ref']})")
