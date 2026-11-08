"""
LLM layer — the AI only REASONS and EXPLAINS; all numbers come from the agents' code.

Set in a .env file (see .env.example):
    LLM_PROVIDER=groq        # or gemini, or none
    GROQ_API_KEY=...         # https://console.groq.com
    GEMINI_API_KEY=...       # https://aistudio.google.com
    LLM_MODEL=               # optional override of the model name

If no key is set (or the call fails), the app still works using built-in template text.
"""
import json
import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SYSTEM = ("You are the reasoning layer of a supply-chain AI agent for an Indian grocery distributor. "
          "Use ONLY the numbers given to you; never invent figures. Be concise, specific and practical. "
          "Use ₹ for money.")

DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-2.5-flash"}


def provider():
    p = os.getenv("LLM_PROVIDER", "").strip().lower()
    if p in ("groq", "gemini") and os.getenv(f"{p.upper()}_API_KEY"):
        return p
    if not p:   # auto-detect
        for cand in ("groq", "gemini"):
            if os.getenv(f"{cand.upper()}_API_KEY"):
                return cand
    return None


def status_text():
    p = provider()
    if not p:
        return "Offline mode (template explanations). Add an API key in .env to enable AI reasoning."
    return f"AI reasoning on: {p} · {os.getenv('LLM_MODEL') or DEFAULT_MODELS[p]}"


def ask(prompt: str, max_tokens: int = 400, system: str = SYSTEM):
    """Returns the model's text, or None if no LLM is configured / the call fails."""
    p = provider()
    if not p:
        return None
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[p]
    try:
        if p == "groq":
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                              json={"model": model, "max_tokens": max_tokens, "temperature": 0.3,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": prompt}]}, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          params={"key": os.getenv("GEMINI_API_KEY")},
                          json={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3}},
                          timeout=30)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(x.get("text", "") for x in parts).strip()
    except Exception as e:   # network / quota / wrong model name
        return f"(AI unavailable: {type(e).__name__}. Showing template text instead.)"


def explain_vendor_choice(po: dict) -> str:
    opts = po["options"].to_dict("records")
    prompt = (f"Explain in 2-3 sentences, for a warehouse manager, why vendor {po['vendor_name']} was chosen "
              f"for {po['qty']} units of {po['product_name']} at {po['warehouse_id']}. "
              f"Stock runs out in {po['days_to_stockout']} days. Options evaluated (total_expected_cost = purchase + "
              f"expected lost sales + late-delivery risk): {json.dumps(opts, default=str)}")
    return ask(prompt, 220) or po["reason"]


def daily_briefing(context: dict, fallback: str) -> str:
    prompt = ("Write today's supply-chain briefing in 4-6 short bullet points: most urgent risks first, then "
              "actions taken, then things to watch (festivals, vendors). Data:\n" + json.dumps(context, default=str))
    return ask(prompt, 380) or fallback


def chat(question: str, context: dict) -> str | None:
    prompt = (f"Current system data (as JSON):\n{json.dumps(context, default=str)}\n\n"
              f"Manager's question: {question}\nAnswer using only this data. If the data doesn't cover it, say so.")
    return ask(prompt, 450)
