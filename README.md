# 📦 AI Supply Chain Management Agent

A multi-agent AI system for **Sahyadri Retail Distribution**, a grocery distributor with 3 warehouses (Mumbai, Pune, Nashik), 14 products and 6 vendors. Every morning the agents check stock, forecast demand, spot risks, pick vendors, draft purchase orders, suggest warehouse transfers and flag anomalies. A manager approves the orders, and a simulator lets time pass so you can watch the whole loop run:

**Detect → Decide → Order → Track → Adjust**

It also includes a real-time **Stream Analytics** page and a **RAG-based Knowledge Assistant** with built-in **AI guardrails**, all in one Streamlit app.

> **Design principle:** code calculates, AI explains. Every number comes from auditable Python and ML code. The LLM only writes explanations, the daily briefing and chat answers, and it works without any API key (template text is used instead).

<!-- Add screenshots here, e.g.
![Control room](docs/control_room.png)
-->

---

## Contents
- [Features](#features)
- [Quick start](#quick-start)
- [Configure the AI key (optional)](#configure-the-ai-key-optional)
- [How the agents work](#how-the-agents-work)
- [Key formulas](#key-formulas)
- [Stream Analytics](#stream-analytics)
- [Knowledge Assistant (RAG)](#knowledge-assistant-rag)
- [AI guardrails](#ai-guardrails)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Results](#results)
- [Suggested demo](#suggested-demo)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## Features

| Page | What it does |
|---|---|
| **Control room** | Today's risks, KPIs and the daily briefing |
| **Agent activity** | Step-by-step log of what each agent did and why |
| **Demand forecast** | 28-day ML forecast, festival watch, AI vs manual accuracy |
| **Purchase orders** | Drafted POs with the vendor comparison, approval and PDF export |
| **Transfers & alerts** | Warehouse-to-warehouse transfers and anomaly alerts |
| **Vendors** | Learned on-time %, delay, fill rate and reliability trend |
| **What-if lab** | Add vendor delays or demand surges and watch the plan change |
| **Simulation** | Autopilot and day-by-day simulation of the full loop |
| **Stream analytics** | Event-stream dashboard, plus the **Knowledge Assistant** at the bottom |
| **Ask the agent** | Chat over the current system data |

## Quick start

Requires Python 3.10+.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Notes:
- The first load takes about 10 seconds while the forecasting model trains.
- `pip install` is slow the first time because of `torch` (needed by Sentence Transformers). Let it finish.
- The first time the Knowledge Assistant starts, it downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB, internet needed once). It is cached afterwards.

## Configure the AI key (optional)

The app works without a key. To enable written explanations, the daily briefing, chat and Knowledge Assistant answers:

```powershell
copy .env.example .env      # Windows   (macOS/Linux: cp .env.example .env)
notepad .env
```

Groq:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
LLM_MODEL=openai/gpt-oss-120b
```

or Google Gemini:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

- Free keys: [Groq](https://console.groq.com/keys) · [Gemini](https://aistudio.google.com/apikey)
- Restart Streamlit after editing `.env`.
- `LLM_MODEL` is optional. Set it if the default model has been retired (Groq changes its model list from time to time).
- **Never commit `.env`.** It is already in `.gitignore`.

## How the agents work

Each agent is a Python class with one job. It reads the shared state (stock, sales, orders), returns a result, and writes its reasoning to a shared activity log shown in the dashboard. The **Orchestrator** runs them in order:

```
Inventory → Forecast → Vendor stats → Risk → Warehouse transfers → Reorder
          → Vendor choice → Cost → Purchase order → Overstock → Anomalies → Briefing
```

| Agent | File | What it does |
|---|---|---|
| Inventory | `agents/inventory_agent.py` | One view of on-hand, confirmed incoming and overdue stock per warehouse and product |
| Forecast | `agents/forecast_agent.py` | One global gradient-boosting model (Poisson loss) forecasts 28 days ahead using calendar, promotion and lagged-demand features; adapts to recent shifts; festival watch |
| Vendor | `agents/vendor_agent.py` | Learns on-time %, delay and fill rate from order history; picks the lowest *total expected cost* vendor |
| Risk | `agents/risk_agent.py` | Day-by-day walk to find when each item runs out versus how fast a vendor can deliver (CRITICAL / HIGH / MEDIUM / OVERSTOCK / LOW) |
| Reorder | `agents/reorder_agent.py` | When and how much: forecast over lead time + review period + dynamic safety stock, with a shelf-life cap for perishables |
| Cost | `agents/reorder_agent.py` | Bulk-discount rounding versus holding cost; overstock capital alerts |
| Purchase order | `agents/po_agent.py` | Structured POs, approval rule (≥ ₹50,000 needs a manager), PDF export |
| Warehouse | `agents/warehouse_agent.py` | Moves surplus stock to a warehouse about to run out |
| Anomaly | `agents/anomaly_agent.py` | Unexplained demand spikes and drops, overdue deliveries, falling vendor reliability, stock write-offs, spoilage |
| Orchestrator | `agents/orchestrator.py` | Runs all agents and writes the activity log |

Supporting code: `core/state.py` (data loading and live state), `core/simulator.py` (approvals and day-by-day simulation), `llm/explainer.py` (Groq/Gemini with offline fallback), `ui.py` (dashboard styling), `config.py` (all tunable numbers), `generate_data.py` (rebuilds the dataset).

**Forecasting details:** every lag feature is at least 28 days old, so the model forecasts 28 days directly instead of feeding its own predictions back in. Stock-out days are excluded from training because sales on those days understate demand. A holdout test (train before 1 Sep 2026, test after) compares the model with a 28-day-average baseline.

**Human in the loop:** orders of ₹50,000 or more always wait for manager approval. Autopilot only approves smaller orders and transfers.

## Key formulas

```
safety stock    = z × σ × √(lead time + review days)        z = 1.65 → 95% service level
order quantity  = forecast over (lead + review) + safety stock − on hand − confirmed arrivals
vendor cost     = purchase cost + expected lost sales (days without stock × daily demand × margin × 1.5)
                  + late-delivery risk
accuracy        = 1 − (total absolute error ÷ total sales)
```

## Stream Analytics

The **Stream analytics** page turns the existing sales history into a stream of sale and sample customer-feedback events.

1. Choose the number of events and click **Generate next event batch**.
2. Explore sales by product and warehouse, feedback sentiment and topics, stock-out signals and unusual sales events (mean + 2σ rule).
3. Download the events as CSV or JSONL.

The demo runs locally with no extra services. For the optional big-data path, the exported JSONL can be processed with **PySpark** and **Kafka**:

```bash
pip install -r requirements-bdsma.txt        # needs Java 17 for PySpark
spark-submit bdsma/pyspark_job.py data/bdsma_stream_events.jsonl

python bdsma/kafka_consumer.py --max-events 0      # terminal 1 (needs a running Kafka broker)
python bdsma/kafka_producer.py data/bdsma_stream_events.jsonl   # terminal 2
```

## Knowledge Assistant (RAG)

At the bottom of the **Stream analytics** page. Ask questions in plain English and get answers grounded in your documents and live system data, with sources.

```
Documents (PDF, TXT, CSV)
  → rag/loader.py     extract text (PDF by page, CSV in row groups)
  → rag/chunker.py    ~800-character sentence-aware chunks, 150-character overlap
  → rag/store.py      all-MiniLM-L6-v2 embeddings → FAISS index (saved in rag_index/)

Question
  → rag/structured.py   live numbers from pandas and the agents (stock risk, sales, vendors, warehouses, reorders)
  → rag/store.py        top-5 similar chunks from FAISS (cosine similarity ≥ threshold)
  → rag/assistant.py    context → llm/explainer.py (Groq/Gemini) → answer with [D#] / [S#] citations
```

- **Starter documents** are in `knowledge_base/` (inventory policy, warehouse SOP, risk playbook). Uploads are saved to `knowledge_base/uploads/`.
- **Persistent index:** the FAISS index is saved to disk. On restart only new or changed files are embedded.
- **Structured data uses pandas, not vector search.** Questions about sales, stock and reorders are answered from the real data and agent output, so the numbers are exact.
- **No answer found:** the assistant replies *"Insufficient information in the knowledge base."*
- **No API key or AI failure:** the retrieved evidence is shown instead of a written answer.

Example questions:
- Which products are at risk of stockout and what should I reorder?
- What is our safety stock and service level policy?
- Which vendor is the most reliable, and which is getting worse?
- How much did we sell of Toned Milk in the last 28 days?
- What is the capacity and current stock level of the Pune warehouse?
- What should we do if a vendor delivery is delayed?

## AI guardrails

All rules live in `rag/guardrails.py` (regex and simple rules, no extra dependencies). A blocked request shows a **🛡️ Blocked by guardrail** warning in the chat.

| # | Guardrail | What it does |
|---|---|---|
| 1 | Prompt injection | Blocks "ignore previous instructions", "reveal your system prompt" and similar phrases. Retrieved chunks are treated as untrusted: chunks containing such phrases are dropped before the prompt is built. |
| 2 | Hallucination prevention | Answers only from retrieved context. Chunks below the similarity threshold (`RAG_MIN_SCORE`, default 0.25) are discarded. With nothing left the reply is *"Insufficient information in the knowledge base."* Sources are always shown. |
| 3 | Input validation | Rejects empty or over-long questions (`RAG_MAX_QUERY_CHARS`, default 500), unsupported file types, empty or oversized (> 10 MB) files, fake PDFs, binary "text" files and unparsable CSVs. |
| 4 | Sensitive information | Refuses questions about API keys, passwords and environment variables; redacts key and password patterns in retrieved text, live data and answers. |
| 5 | Domain restriction | Only supply-chain questions (inventory, vendors, warehouses, sales, logistics, operations) are processed. |
| 6 | Output validation | Every number in a generated answer must appear in the sources, otherwise the answer is rejected and the evidence is shown instead. Advice is labelled "Suggestion:" so it is not mistaken for a confirmed fact. |

**Limitations:** these are pattern rules, not a complete defence. Paraphrased, misspelt, obfuscated or non-English injection attempts can slip through. The domain check is keyword-based and can wrongly allow or refuse edge cases. The output check verifies numbers only, not wording or reasoning, and cannot tell whether a "Suggestion" label was applied honestly. Redaction covers common key formats only. Keep real credentials in `.env`, never in documents.

## Project structure

```
supply-chain-agent/
├── app.py                     # Streamlit app (all pages)
├── config.py                  # tunable numbers (service level, limits, horizons)
├── ui.py                      # styling and UI helpers
├── generate_data.py           # rebuilds the synthetic dataset
├── agents/                    # the agents and the orchestrator
├── core/                      # state loading and the simulator
├── llm/explainer.py           # Groq / Gemini client with offline fallback
├── bdsma/                     # event generation, PySpark job, Kafka producer/consumer
├── rag/                       # loader, chunker, store (FAISS), structured, assistant, guardrails, ui
├── knowledge_base/            # starter documents for the assistant
├── data/                      # CSV datasets
├── tests/                     # test_rag, test_guardrails, test_app_flow, smoke_real_model
├── requirements.txt
├── requirements-bdsma.txt     # optional: PySpark and Kafka
└── .env.example
```

## Tests

```bash
python tests/test_rag.py           # ingestion, chunking, FAISS, persistence, answers, citations
python tests/test_guardrails.py    # the six guardrails
python tests/test_app_flow.py      # every page loads; Knowledge Assistant works inside Stream analytics
python tests/smoke_real_model.py   # real MiniLM embeddings (needs internet once)
```

The first three run offline using a small stand-in embedder and a stubbed AI response. The AI call and the real embedding model are covered only by the smoke test and by running the app with a key.

## Results

Forecast accuracy on 35 unseen days: **80.6% (AI) vs 75.1% (old 28-day average method)**. In a 3-week autopilot simulation, items were out of stock on about **2.3% of days vs 4.7%** in the real Jul–Sep history, while stock value fell from about ₹16 lakh to ₹14 lakh. The simulated demand is drawn around the model's own forecast, so treat this as a simulation result, not a field result.

## Suggested demo

1. **Control room:** an item is out of stock and a delivery truck is overdue.
2. **Agent activity:** walk through each agent's reasoning.
3. **Transfers & alerts:** approve a transfer, then point out a demand spike and a vendor whose reliability is falling.
4. **Purchase orders:** open an order, show why a costlier but faster vendor won, approve it and download the PDF.
5. **What-if lab:** add 3 days of delay to a vendor and watch the agents switch vendors.
6. **Simulation:** turn on Autopilot and click "Next 7 days".
7. **Demand forecast:** show the festival bump and the AI vs manual accuracy chart.
8. **Stream analytics → Knowledge Assistant:** ask a stock question, a policy question, an off-topic question (refused) and an injection attempt (blocked).

## Limitations

- The data is **synthetic**, built from realistic patterns rather than records from a real company.
- The simulator does not model spoilage, and transfers assume 1 day of travel.
- The forecast does not know about future promotions.
- Scanned PDFs are not supported (no OCR).
- Guardrails are rule-based and not a guarantee (see above).

## Troubleshooting

| Problem | Fix |
|---|---|
| "AI unavailable: HTTPError" | Wrong or expired key, retired model (set `LLM_MODEL`), quota used up, or no internet |
| Sidebar says "Offline mode" | `.env` is missing, misnamed (`.env.txt`), or not next to `app.py`; restart after editing |
| Knowledge Assistant cannot start | Run `pip install -r requirements.txt` again and restart |
| Embedding model download fails | Needs internet on the first run only |
| `pip install` seems stuck | It is unpacking `torch`; wait a few minutes |

## License

Add a license of your choice (for example MIT) before publishing.
