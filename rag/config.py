"""RAG settings (kept separate from the main config.py so the original file is untouched)."""
import os
from config import BASE_DIR

KB_DIR = os.path.join(BASE_DIR, "knowledge_base")          # starter documents, indexed automatically
INDEX_DIR = os.path.join(BASE_DIR, "rag_index")            # persisted FAISS index + chunk metadata
UPLOAD_DIR = os.path.join(KB_DIR, "uploads")               # files uploaded from the UI are kept here

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_CHARS = 800          # ~150-200 words per chunk
CHUNK_OVERLAP = 150        # characters shared between neighbouring chunks
CSV_ROWS_PER_CHUNK = 15    # rows per chunk for uploaded/small CSV files (header repeated in each chunk)
TOP_K = 5
MIN_SCORE = 0.25           # cosine similarity below this = "not relevant"
MAX_UPLOAD_MB = 10
SUPPORTED = (".pdf", ".txt", ".csv")

# Dataset CSVs small enough to embed (reference/master data). Big transactional files
# (sales, daily inventory, purchase orders) are answered with pandas instead, see structured.py.
DATASET_CSVS = ("products.csv", "vendors.csv", "warehouses.csv", "vendor_products.csv")
