import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.get("https://api.groq.com/openai/v1/models",
                 headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}, timeout=30)
print(r.status_code)
for m in sorted(x["id"] for x in r.json().get("data", [])):
    print(m)