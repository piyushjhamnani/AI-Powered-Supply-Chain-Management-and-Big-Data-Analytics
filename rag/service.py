"""Glue used by the Streamlit page: builds the knowledge base once and keeps it in sync with the files on disk."""
import os

from rag import config as C
from rag.store import KnowledgeBase
from config import DATA_DIR


def kb_files() -> dict[str, str]:
    """Everything that should be searchable: starter docs, uploads and the small master-data CSVs."""
    files = {}
    for root, _, names in os.walk(C.KB_DIR):
        for n in sorted(names):
            if n.lower().endswith(C.SUPPORTED):
                p = os.path.join(root, n)
                files[os.path.relpath(p, C.KB_DIR).replace(os.sep, "/")] = p
    for n in C.DATASET_CSVS:
        p = os.path.join(DATA_DIR, n)
        if os.path.exists(p):
            files[f"dataset/{n}"] = p
    return files


def save_upload(name: str, data: bytes) -> str:
    os.makedirs(C.UPLOAD_DIR, exist_ok=True)
    safe = os.path.basename(name).replace("..", "_")
    path = os.path.join(C.UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(data)
    return path


def delete_upload(rel_name: str):
    p = os.path.join(C.KB_DIR, rel_name)
    if rel_name.startswith("uploads/") and os.path.exists(p):
        os.remove(p)
