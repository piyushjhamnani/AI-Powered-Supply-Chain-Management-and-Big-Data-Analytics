import os, requests
from dotenv import load_dotenv
load_dotenv()

p = (os.getenv("LLM_PROVIDER") or "").lower()
print("provider:", p, "| model override:", os.getenv("LLM_MODEL"))

if p == "gemini":
    k = os.getenv("GEMINI_API_KEY") or ""
    m = os.getenv("LLM_MODEL") or "gemini-2.5-flash"
    print("key loaded:", bool(k), "| length:", len(k))
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                      params={"key": k}, json={"contents": [{"parts": [{"text": "hi"}]}]}, timeout=30)
else:
    k = os.getenv("GROQ_API_KEY") or ""
    m = os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"
    print("key loaded:", bool(k), "| length:", len(k))
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {k}"},
                      json={"model": m, "messages": [{"role": "user", "content": "hi"}]}, timeout=30)
print(r.status_code, r.text[:400])