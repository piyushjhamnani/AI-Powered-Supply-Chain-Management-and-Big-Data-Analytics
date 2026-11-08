"""Generate realistic demo events from the project's existing supply-chain data."""
from __future__ import annotations

from datetime import datetime, timedelta
import uuid

import numpy as np
import pandas as pd

FEEDBACK_TEMPLATES = [
    ("negative", "delivery", "My order arrived late and the delivery was delayed."),
    ("negative", "quality", "The product quality was poor and the packaging was damaged."),
    ("negative", "availability", "The product was out of stock again. Please improve availability."),
    ("negative", "price", "The price is too high compared with other stores."),
    ("positive", "quality", "Good quality product. Fresh and well packed."),
    ("positive", "delivery", "The order arrived on time. Great delivery service."),
    ("positive", "price", "Good value for money and reasonable pricing."),
    ("neutral", "availability", "Please keep this product available in all warehouses."),
    ("neutral", "delivery", "Delivery was okay, but tracking updates could be better."),
]


def classify_feedback(message: str) -> tuple[str, str]:
    """Small transparent keyword baseline for feedback sentiment and topic."""
    text = str(message).lower()
    positive_words = {"good", "great", "fresh", "well", "reasonable", "value", "excellent", "happy"}
    negative_words = {"late", "delayed", "poor", "damaged", "stock", "out", "high", "bad", "issue"}
    words = set(text.replace(".", "").replace(",", "").split())
    positive_score = len(words & positive_words)
    negative_score = len(words & negative_words)
    sentiment = "positive" if positive_score > negative_score else "negative" if negative_score > positive_score else "neutral"

    topic_terms = {
        "delivery": {"delivery", "arrived", "late", "delayed", "tracking"},
        "quality": {"quality", "fresh", "packaging", "damaged", "packed"},
        "availability": {"stock", "available", "availability", "warehouse"},
        "price": {"price", "pricing", "value", "money", "cost"},
    }
    topic_scores = {topic: len(words & terms) for topic, terms in topic_terms.items()}
    topic = max(topic_scores, key=topic_scores.get) if max(topic_scores.values(), default=0) else "other"
    return sentiment, topic


def build_event_batch(sales: pd.DataFrame, products: pd.DataFrame, n: int = 100,
                      seed: int | None = None) -> pd.DataFrame:
    """Create a batch of sale and customer-feedback events from existing CSV data."""
    if sales.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    product_names = products.set_index("product_id").product_name.to_dict()
    sample = sales.sample(n=n, replace=n > len(sales), random_state=seed).reset_index(drop=True)
    start = datetime.now().replace(microsecond=0)
    events = []

    for i, row in sample.iterrows():
        event_time = start + timedelta(seconds=i * 2)
        pid = str(row.product_id)
        common = {
            "event_id": str(uuid.uuid4())[:12],
            "event_time": event_time.isoformat(sep=" "),
            "product_id": pid,
            "product_name": product_names.get(pid, pid),
            "warehouse_id": str(row.warehouse_id),
        }
        events.append({
            **common,
            "event_type": "sale",
            "units_sold": int(row.units_sold),
            "revenue": float(row.revenue),
            "stockout": bool(row.stockout),
            "feedback_text": "",
            "sentiment": "not_applicable",
            "topic": "sales",
        })

        # Add customer feedback to approximately one out of every four sale events.
        if rng.random() < 0.25:
            _, _, message = FEEDBACK_TEMPLATES[int(rng.integers(0, len(FEEDBACK_TEMPLATES)))]
            sentiment, topic = classify_feedback(message)
            events.append({
                **{**common, "event_id": str(uuid.uuid4())[:12],
                   "event_time": (event_time + timedelta(seconds=1)).isoformat(sep=" ")},
                "event_type": "customer_feedback",
                "units_sold": 0,
                "revenue": 0.0,
                "stockout": False,
                "feedback_text": message,
                "sentiment": sentiment,
                "topic": topic,
            })

    return pd.DataFrame(events).sort_values("event_time").reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> dict:
    """Return dashboard-ready summaries without requiring external services."""
    if events.empty:
        return {
            "total_events": 0, "sales_events": 0, "feedback_events": 0,
            "revenue": 0.0, "units": 0, "stockout_rate": 0.0,
            "sentiment": pd.Series(dtype="int64"),
            "topics": pd.Series(dtype="int64"),
            "top_products": pd.DataFrame(), "warehouse_sales": pd.DataFrame(),
            "anomalies": pd.DataFrame(),
        }

    sales = events[events.event_type == "sale"].copy()
    feedback = events[events.event_type == "customer_feedback"].copy()
    if not sales.empty:
        top_products = (sales.groupby("product_name", as_index=False)
                        .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
                        .sort_values("units_sold", ascending=False).head(10))
        warehouse_sales = (sales.groupby("warehouse_id", as_index=False)
                           .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
                           .sort_values("revenue", ascending=False))
        mean_units = sales.units_sold.mean()
        std_units = sales.units_sold.std(ddof=0)
        threshold = mean_units + (2 * std_units)
        anomalies = sales[sales.units_sold > threshold].copy()
        anomalies["anomaly_reason"] = f"Sale quantity exceeded batch mean + 2 standard deviations ({threshold:.1f})"
    else:
        top_products, warehouse_sales, anomalies = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sentiment = feedback.sentiment.value_counts() if not feedback.empty else pd.Series(dtype="int64")
    topics = feedback.topic.value_counts() if not feedback.empty else pd.Series(dtype="int64")
    return {
        "total_events": len(events),
        "sales_events": len(sales),
        "feedback_events": len(feedback),
        "revenue": float(sales.revenue.sum()) if not sales.empty else 0.0,
        "units": int(sales.units_sold.sum()) if not sales.empty else 0,
        "stockout_rate": float(sales.stockout.mean() * 100) if not sales.empty else 0.0,
        "sentiment": sentiment,
        "topics": topics,
        "top_products": top_products,
        "warehouse_sales": warehouse_sales,
        "anomalies": anomalies,
    }
