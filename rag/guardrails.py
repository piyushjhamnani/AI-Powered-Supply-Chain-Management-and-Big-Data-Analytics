"""Lightweight, rule-based guardrails for the Knowledge Assistant (regex + simple rules, no extra dependencies).

Pipeline order (see rag/assistant.py and rag/ui.py):
  upload  -> validate_upload()
  query   -> validate_query() -> check_sensitive_request() -> check_domain()
  retrieve-> filter_passages()  (drops injected chunks, applies similarity threshold, redacts secrets)
  context -> has_enough_context()  else INSUFFICIENT
  answer  -> validate_output()  (unsupported numbers, secret leaks, citations, fact-vs-suggestion note)

LIMITATIONS: these are pattern rules. They catch common, obvious attacks and mistakes, not every paraphrase,
other-language or obfuscated attempt, and the number check cannot verify wording or logic. Treat them as one
layer of defence, not a guarantee.
"""
import io
import os
import re
from dataclasses import dataclass

from rag import config as C

INSUFFICIENT = "Insufficient information in the knowledge base."
MAX_QUERY_CHARS = int(os.getenv("RAG_MAX_QUERY_CHARS", "500"))
MIN_QUERY_CHARS = 3
SIMILARITY_THRESHOLD = float(os.getenv("RAG_MIN_SCORE", C.MIN_SCORE))     # configurable via .env


@dataclass
class Check:
    ok: bool
    guardrail: str = ""
    message: str = ""


PASS = Check(True)


def _block(guardrail: str, message: str) -> Check:
    return Check(False, guardrail, message)


# --------------------------------------------------------------------------- 1. prompt injection
_INJECTION = [re.compile(p, re.I) for p in (
    r"\bignore\b.{0,30}\b(previous|prior|above|earlier|all|any|the)\b.{0,30}\b(instruction|prompt|rule|context)s?",
    r"\b(disregard|forget|override|bypass)\b.{0,30}\b(instruction|prompt|rule|guideline|system|safety)s?",
    r"\b(reveal|show|print|repeat|leak|display|tell me)\b.{0,30}\b(system prompt|hidden prompt|your instructions|your prompt|initial prompt)",
    r"\byou are now\b|\bact as\b.{0,20}\b(dan|jailbreak|unrestricted|developer mode)|\bjailbreak\b|\bdeveloper mode\b",
    r"\bnew (instructions?|rules?)\s*:",
    r"(^|\n)\s*(system|assistant)\s*:",
    r"<\s*/?\s*(system|instructions?)\s*>|\[/?INST\]",
    r"\bpretend (that )?you (have no|are not|don't have)\b",
)]


def detect_injection(text: str) -> bool:
    return any(p.search(text or "") for p in _INJECTION)


# --------------------------------------------------------------------------- 4. sensitive information
_SECRET_PATTERNS = [(re.compile(p, re.I), r) for p, r in (
    (r"\bAIza[0-9A-Za-z_\-]{30,}", "[REDACTED_API_KEY]"),                      # Google / Gemini
    (r"\bgsk_[0-9A-Za-z]{20,}", "[REDACTED_API_KEY]"),                         # Groq
    (r"\bsk-[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]"),                       # OpenAI-style
    (r"\b(?:ghp|gho|xox[bap])[_\-][0-9A-Za-z\-]{20,}", "[REDACTED_TOKEN]"),
    (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_API_KEY]"),                           # AWS access key id
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)", "[REDACTED_PRIVATE_KEY]"),
    (r"\b([A-Z][A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*)\s*[=:]\s*\S+", r"\1=[REDACTED]"),
    (r"\b(password|passwd|pwd|passphrase|api[ _-]?key|secret|token)\b(\s*(?:is|=|:)\s*)[^\s,;]+", r"\1\2[REDACTED]"),
)]
_SECRET_REQUEST = re.compile(
    r"\b(api[ _-]?keys?|secret keys?|passwords?|passwd|credentials?|access tokens?|private keys?|"
    r"env(?:ironment)?[ _-]?(?:variables?|vars?|file)|\.env\b|GEMINI_API_KEY|GROQ_API_KEY)\b", re.I)


def redact(text: str) -> str:
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text or "")
    return text


def contains_secret(text: str) -> bool:
    return redact(text) != (text or "")


def check_sensitive_request(question: str) -> Check:
    if _SECRET_REQUEST.search(question):
        return _block("Sensitive information", "I can't share or discuss API keys, passwords or environment variables.")
    if contains_secret(question):
        return _block("Sensitive information", "Your question appears to contain a credential. Please remove it and ask again.")
    return PASS


# --------------------------------------------------------------------------- 3. input validation
def validate_query(question: str) -> Check:
    q = (question or "").strip()
    if len(q) < MIN_QUERY_CHARS:
        return _block("Input validation", "Please type a question.")
    if len(q) > MAX_QUERY_CHARS:
        return _block("Input validation", f"Question is too long ({len(q)} characters). Keep it under {MAX_QUERY_CHARS}.")
    if not re.search(r"[A-Za-z0-9]", q):
        return _block("Input validation", "Please type a question using words.")
    if detect_injection(q):
        return _block("Prompt injection", "This looks like an attempt to override my instructions, so I did not process it. "
                                          "Please ask a supply chain question.")
    return PASS


def validate_upload(name: str, data: bytes) -> Check:
    ext = os.path.splitext(name or "")[1].lower()
    if ext not in C.SUPPORTED:
        return _block("Input validation", f"'{name}': unsupported file type. Allowed: {', '.join(C.SUPPORTED)}.")
    if not data:
        return _block("Input validation", f"'{name}': file is empty.")
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        return _block("Input validation", f"'{name}': larger than the {C.MAX_UPLOAD_MB} MB limit.")
    if ext == ".pdf" and not data.lstrip()[:5] == b"%PDF-":
        return _block("Input validation", f"'{name}': not a valid PDF (wrong file header).")
    if ext in (".txt", ".csv") and b"\x00" in data[:4096]:
        return _block("Input validation", f"'{name}': looks like a binary file, not text.")
    if ext == ".csv":
        try:
            import pandas as pd
            pd.read_csv(io.BytesIO(data), dtype=str, nrows=5)
        except Exception as e:
            return _block("Input validation", f"'{name}': not a valid CSV ({type(e).__name__}).")
    return PASS


# --------------------------------------------------------------------------- 5. domain restriction
DOMAIN_TERMS = (
    "inventory", "stock", "reorder", "re-order", "replenish", "shortage", "shortfall", "stockout", "overstock",
    "safety stock", "service level", "lead time", "vendor", "supplier", "supply", "delivery", "deliver", "shipment",
    "logistic", "transport", "freight", "warehouse", "storage", "capacity", "transfer", "dispatch", "receiving",
    "picking", "sales", "sold", "sell", "demand", "forecast", "revenue", "order", "purchase", "po ", "product",
    "sku", "price", "cost", "margin", "shelf", "perishable", "expiry", "category", "risk", "policy", "procedure",
    "sop", "operation", "promotion", "festival", "anomaly", "spike", "reliab", "on-time", "delay", "overdue",
    "moq", "batch", "unit", "document", "knowledge base", "uploaded", "customer", "feedback", "sentiment", "stream",
)
OUT_OF_DOMAIN = ("I can only help with supply chain questions: inventory, vendors, warehouses, sales and demand, "
                 "logistics and operations. Please ask about one of those.")


def check_domain(question: str, entities: dict | None = None) -> Check:
    """entities: output of structured.find_entities (a mentioned product/vendor/warehouse also counts as on-topic)."""
    q = f" {question.lower()} "
    if any(t in q for t in DOMAIN_TERMS):
        return PASS
    if entities and any(entities.values()):
        return PASS
    return _block("Domain restriction", OUT_OF_DOMAIN)


# --------------------------------------------------------------------------- 1+2. retrieved content
def filter_passages(passages: list[dict], min_score: float | None = None) -> tuple[list[dict], dict]:
    """Retrieved text is untrusted. Drop chunks that contain injection phrases or fall below the similarity
    threshold, and redact secrets in the rest. -> (clean passages, {'injected': n, 'low_score': n})."""
    min_score = SIMILARITY_THRESHOLD if min_score is None else min_score
    kept, dropped = [], {"injected": 0, "low_score": 0}
    for p in passages:
        if p.get("score") is not None and p["score"] < min_score:
            dropped["low_score"] += 1
        elif detect_injection(p["text"]):
            dropped["injected"] += 1
        else:
            kept.append(dict(p, text=redact(p["text"])))
    return kept, dropped


def clean_facts(facts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(label, redact(text)) for label, text in facts]


def has_enough_context(passages: list, facts: list) -> bool:
    return bool(passages or facts)


# --------------------------------------------------------------------------- 2+6. output validation
_CITE = re.compile(r"\[[DS]\d+\]")
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
FACT_NOTE = ("Statements with [D#]/[S#] come from the knowledge base or live data. "
             "Anything labelled 'Suggestion' is advice, not a confirmed fact.")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(_CITE.sub(" ", text)):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unsupported_numbers(answer: str, context_text: str, question: str = "") -> list[float]:
    """Numbers in the answer that appear nowhere in the context or the question (rounding allowed).
    List markers like '1.' at line start are ignored."""
    body = re.sub(r"(?m)^\s*\d+[.)]\s", "", answer)
    allowed = _numbers(context_text) | _numbers(question)
    rounded = {round(a) for a in allowed} | {round(a, 1) for a in allowed}
    return sorted(n for n in _numbers(body)
                  if n not in allowed and round(n) not in rounded and round(n, 1) not in rounded)


def validate_output(answer: str, passages: list[dict], facts: list[tuple[str, str]], question: str = "") -> Check:
    """-> Check(ok, guardrail, message). On ok, message holds the (secret-redacted) answer to display."""
    if not answer or not answer.strip():
        return _block("Output validation", "The model returned an empty answer.")
    if answer.strip().rstrip(".").lower() == INSUFFICIENT.rstrip(".").lower():
        return Check(True, message=INSUFFICIENT)
    ctx = "\n".join(p["text"] for p in passages) + "\n" + "\n".join(t for _, t in facts)
    bad = unsupported_numbers(answer, ctx, question)
    if bad:
        shown = ", ".join(f"{n:g}" for n in bad[:5])
        return _block("Output validation", f"The generated answer contained figures not found in the sources ({shown}), "
                                           "so it was rejected. The retrieved evidence is shown instead.")
    return Check(True, message=redact(answer))


def has_citation(answer: str) -> bool:
    return bool(_CITE.search(answer))
