"""
Structured-data lookups with pandas. Numbers about live stock, sales, vendors and reorders come from the
project's own data and agents (never from vector search), then are handed to the LLM as trusted facts.
"""
import re

import pandas as pd

import config as cfg

INTENTS = {
    "stock":     ("stock", "shortage", "run out", "runs out", "running out", "out of", "low", "reorder", "re-order",
                  "replenish", "order", "shortfall", "critical", "inventory", "risk", "overstock"),
    "demand":    ("demand", "sales", "sold", "sell", "revenue", "forecast", "selling", "popular", "top"),
    "vendor":    ("vendor", "supplier", "delivery", "deliver", "lead time", "late", "delay", "reliab", "on-time", "on time"),
    "warehouse": ("warehouse", "capacity", "transfer", "storage", "dc", "distribution"),
    "product":   ("price", "cost", "shelf", "expire", "perishable", "category", "margin", "product", "multiple", "moq"),
    "policy":    ("policy", "safety stock", "service level", "review", "threshold", "auto-approve", "approve", "limit"),
}
_GENERIC = {"dairy", "fresh", "supply", "agro", "wholesale", "mart", "traders", "distributors", "beverages", "general",
            "pure", "powder", "tray", "vanilla", "cream", "cow", "tea"}


def _has(q: str, words) -> bool:
    return any(w in q for w in words)


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z]{4,}", s.lower())} - _GENERIC


def find_entities(q: str, data: dict) -> dict:
    ql, qt = q.lower(), set(re.findall(r"[a-z0-9\-]+", q.lower()))
    prods, whs, vens = [], [], []
    for r in data["products"].itertuples():
        if r.product_id.lower() in qt or r.product_name.lower() in ql or (_tokens(r.product_name) & qt):
            prods.append(r.product_id)
    w = data["warehouses"]
    for r in w.itertuples():
        city_tokens = _tokens(r.city) | _tokens(r.warehouse_name)
        if r.warehouse_id.lower() in qt or (city_tokens & qt):
            whs.append(r.warehouse_id)
    vtok = {r.vendor_id: _tokens(r.vendor_name) for r in data["vendors"].itertuples()}
    wh_words = set().union(*[_tokens(c) for c in w.city])
    for r in data["vendors"].itertuples():
        unique = {t for t in vtok[r.vendor_id] if sum(t in o for o in vtok.values()) == 1} - wh_words
        if r.vendor_id.lower() in qt or r.vendor_name.lower() in ql or (unique & qt):
            vens.append(r.vendor_id)
    return {"products": prods, "warehouses": whs, "vendors": vens}


def _fmt(df: pd.DataFrame, n: int = 15) -> str:
    return df.head(n).round(2).to_csv(index=False).strip()


def live_facts(question: str, data: dict, state=None, result=None) -> list[tuple[str, str]]:
    """-> [(source label, text)]. Empty list when the question needs no structured data."""
    q = question.lower()
    ent = find_entities(question, data)
    P, W, V = ent["products"], ent["warehouses"], ent["vendors"]
    want = {k: _has(q, kw) for k, kw in INTENTS.items()}
    facts = []

    def risk_filtered():
        r = result.risk
        if P: r = r[r.product_id.isin(P)]
        if W: r = r[r.warehouse_id.isin(W)]
        return r

    if want["stock"] and result is not None:
        r = risk_filtered()
        if not (P or W):
            r = r[r.risk != "LOW"]
        cols = ["warehouse_id", "product_name", "on_hand", "incoming_confirmed", "avg_daily_forecast",
                "days_to_stockout", "best_lead_days", "risk", "reason"]
        if not r.empty:
            facts.append(("Live inventory risk (Risk Agent)", _fmt(r[cols], 20)))
        pos = [p for p in result.purchase_orders if (not P or p.get("product_id") in P) and (not W or p["warehouse_id"] in W)]
        if pos:
            txt = "\n".join(f"{p['po_id']}: {p['qty']} x {p['product_name']} at {p['warehouse_id']} from {p['vendor_name']} "
                            f"(priority {p['priority']}, ₹{p['total_cost']:,.0f}) - {p['reason']}" for p in pos[:10])
            facts.append(("Reorder recommendations (Reorder/PO Agents)", txt))
        if result.transfers and not P and not W:
            facts.append(("Suggested warehouse transfers", "\n".join(
                f"{t['qty']} x {t['product_name']}: {t['from_warehouse']} -> {t['to_warehouse']} - {t['reason']}"
                for t in result.transfers[:6])))

    if want["demand"] and state is not None:
        s = state.sales
        s = s[s.date > s.date.max() - pd.Timedelta(days=28)]
        if P: s = s[s.product_id.isin(P)]
        if W: s = s[s.warehouse_id.isin(W)]
        names = data["products"].set_index("product_id").product_name
        g = (s.groupby("product_id").agg(units_28d=("units_sold", "sum"), revenue_28d=("revenue", "sum"),
                                         stockout_days=("stockout", "sum")).reset_index())
        g.insert(1, "product_name", g.product_id.map(names))
        g = g.sort_values("units_28d", ascending=False)
        if not g.empty:
            facts.append((f"Sales, last 28 days to {s.date.max():%d %b %Y} (sales_history.csv)", _fmt(g)))
        if result is not None and (P or W):
            f = risk_filtered()[["warehouse_id", "product_name", "avg_daily_forecast", "forecast_28d"]]
            if not f.empty:
                facts.append(("Demand forecast (Forecast Agent)", _fmt(f)))

    if want["vendor"] or V:
        v = data["vendors"]
        if V: v = v[v.vendor_id.isin(V)]
        facts.append(("Vendor master (vendors.csv)", _fmt(v)))
        if result is not None:
            vs = result.vendor_stats
            if V: vs = vs[vs.vendor_name.isin(v.vendor_name)]
            cols = [c for c in ("vendor_name", "reliability_score", "on_time_pct", "recent_on_time_pct", "trend") if c in vs]
            facts.append(("Vendor delivery performance (Vendor Agent)", _fmt(vs[cols])))
        if P or V:
            vp = data["vendor_products"]
            if P: vp = vp[vp.product_id.isin(P)]
            if V: vp = vp[vp.vendor_id.isin(V)]
            facts.append(("Vendor price & lead times (vendor_products.csv)", _fmt(vp, 20)))

    if want["warehouse"] or W:
        w = data["warehouses"]
        if W: w = w[w.warehouse_id.isin(W)]
        if result is not None:
            oh = result.risk.groupby("warehouse_id").agg(units_on_hand=("on_hand", "sum"),
                                                         items_at_risk=("risk", lambda x: int(x.isin(["CRITICAL", "HIGH"]).sum())))
            w = w.merge(oh, left_on="warehouse_id", right_index=True, how="left")
            w["utilisation_pct"] = (100 * w.units_on_hand / w.capacity_units).round(1)
        facts.append(("Warehouses and live stock (warehouses.csv + Inventory Agent)", _fmt(w)))

    if want["product"] or (P and not facts):
        p = data["products"]
        if P:
            facts.append(("Product master (products.csv)", _fmt(p[p.product_id.isin(P)])))

    if want["policy"] or want["stock"]:
        facts.append(("System inventory policy (config.py)", (
            f"Service level z-score {cfg.SERVICE_LEVEL_Z} (95%); review period days by group {cfg.REVIEW_DAYS}; "
            f"overstock above {cfg.OVERSTOCK_DAYS} days of cover (non-perishables); purchase orders under "
            f"₹{cfg.AUTO_APPROVE_LIMIT:,} may be auto-approved by Autopilot; forecast horizon {cfg.FORECAST_HORIZON} days; "
            f"transfers take {cfg.TRANSFER_DAYS} day; donor warehouse keeps >= {cfg.DONOR_KEEP_DAYS} days of cover.")))
    return facts
