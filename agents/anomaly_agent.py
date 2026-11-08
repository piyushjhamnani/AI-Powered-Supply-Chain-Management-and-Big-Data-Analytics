"""
Anomaly Detection Agent — spots things a human should look at:
demand spikes/drops with no promotion or festival to explain them, overdue vendor deliveries,
vendors whose reliability is falling, stock losses (shrinkage) and unusual spoilage.
"""
import pandas as pd
from config import SPIKE_RATIO, DROP_RATIO


class AnomalyAgent:
    name = "Anomaly Agent"

    def __init__(self, data):
        self.products = data["products"].set_index("product_id")
        self.calendar = data["calendar"].set_index("date")
        self.adjustments = data["adjustments"]
        self.inv_daily = data["inventory_daily"]

    def detect(self, state, vstats, log):
        out = []
        t = state.today
        s = state.sales
        last7 = s[(s.date >= t - pd.Timedelta(days=7)) & (s.date < t)]
        prior = s[(s.date >= t - pd.Timedelta(days=35)) & (s.date < t - pd.Timedelta(days=7)) & (~s.on_promotion)]
        fest_days = self.calendar.loc[t - pd.Timedelta(days=7): t - pd.Timedelta(days=1)].festival_window.dropna()
        fest = ", ".join(sorted(set(f for f in fest_days if f)))
        a = last7.groupby(["warehouse_id", "product_id"]).agg(recent=("units_sold", "mean"),
                                                            promo=("on_promotion", "any"), so=("stockout", "sum"))
        b = prior.groupby(["warehouse_id", "product_id"]).units_sold.mean().rename("normal")
        j = a.join(b).dropna()
        j["ratio"] = j.recent / j.normal.clip(lower=0.1)
        for (wh, pid), r in j.iterrows():
            name = self.products.loc[pid].product_name
            if r.ratio >= SPIKE_RATIO:
                expl = "promotion running" if r.promo else (f"festival: {fest}" if fest else "no promotion or festival")
                sev = "MEDIUM" if (r.promo or fest) else "HIGH"
                out.append(dict(type="Demand spike", severity=sev, warehouse_id=wh, product_id=pid,
                                detail=f"{name}: last 7 days avg {r.recent:.0f}/day vs normal {r.normal:.0f}/day "
                                       f"(+{(r.ratio - 1) * 100:.0f}%) — {expl}",
                                action="Confirm with sales team (bulk B2B buyer? competitor stock-out?) before "
                                       "treating it as the new normal"))
            elif r.ratio <= DROP_RATIO and r.so == 0:
                out.append(dict(type="Demand drop", severity="MEDIUM", warehouse_id=wh, product_id=pid,
                                detail=f"{name}: {r.recent:.0f}/day vs normal {r.normal:.0f}/day "
                                       f"({(r.ratio - 1) * 100:.0f}%) with stock available",
                                action="Check pricing, shelf placement or a competitor offer"))
        # overdue deliveries
        late = state.pipeline[(state.pipeline.eta < t) & (state.pipeline.kind == "PO")]
        for x in late.itertuples():
            days = (t - x.eta).days
            out.append(dict(type="Delivery overdue", severity="HIGH" if days >= 2 else "MEDIUM",
                            warehouse_id=x.warehouse_id, product_id=x.product_id,
                            detail=f"{x.ref}: {x.qty} × {self.products.loc[x.product_id].product_name} from {x.source} "
                                   f"is {days} day(s) late",
                            action="Call vendor for a firm ETA; consider emergency order or transfer"))
        # vendor decline
        for v in vstats[vstats.trend == "Declining"].itertuples():
            out.append(dict(type="Vendor reliability falling", severity="HIGH", warehouse_id="", product_id="",
                            detail=f"{v.vendor_name}: on-time {v.recent_on_time_pct:.0f}% in last 60 days vs "
                                   f"{v.on_time_pct:.0f}% overall (listed rating still {v.listed_rating})",
                            action="Shift urgent orders to the alternate vendor; review contract SLA"))
        # shrinkage (last 30 days, real data) — big single losses
        adj = self.adjustments[(self.adjustments.date >= t - pd.Timedelta(days=30))]
        for x in adj.itertuples():
            p = self.products.loc[x.product_id]
            value = -x.qty_change * p.unit_cost
            if value >= 3000:
                out.append(dict(type="Stock loss", severity="MEDIUM", warehouse_id=x.warehouse_id,
                                product_id=x.product_id,
                                detail=f"{x.date:%d %b}: {-x.qty_change} × {p.product_name} written off "
                                       f"(₹{value:,.0f}) — {x.reason}",
                                action="Audit CCTV/access logs and recount high-value items"))
        # spoilage
        inv = self.inv_daily[self.inv_daily.date >= t - pd.Timedelta(days=30)]
        sp = inv.groupby(["warehouse_id", "product_id"])[["spoiled", "sold"]].sum()
        sp = sp[(sp.spoiled > 0)]
        for (wh, pid), r in sp.iterrows():
            rate = r.spoiled / max(1, r.sold)
            if rate > 0.05:
                out.append(dict(type="High spoilage", severity="MEDIUM", warehouse_id=wh, product_id=pid,
                                detail=f"{self.products.loc[pid].product_name}: {r.spoiled:.0f} units spoiled in "
                                       f"30 days ({rate:.1%} of sales)",
                                action="Order smaller, more frequent batches"))
        for a_ in out:
            log.add(self.name, f"{a_['type']}: {a_['detail']}", "critical" if a_["severity"] == "HIGH" else "warn",
                    a_["warehouse_id"], a_["product_id"])
        if not out:
            log.add(self.name, "No anomalies detected", "ok")
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        return sorted(out, key=lambda x: order[x["severity"]])
