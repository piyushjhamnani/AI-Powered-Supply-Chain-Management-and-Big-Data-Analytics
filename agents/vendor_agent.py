"""
Vendor Intelligence Agent
Learns each vendor's real behaviour from purchase-order history (on-time %, average delay,
fill rate, recent trend) and picks the vendor with the lowest *total expected cost*:
    purchase cost  +  expected lost-sales penalty if stock runs out before delivery
                   +  risk cost of late delivery
So a cheaper vendor can lose to a faster, more reliable one when stock is tight.
"""
import numpy as np
import pandas as pd
from config import STOCKOUT_PENALTY_FACTOR


class VendorAgent:
    name = "Vendor Agent"

    def __init__(self, data):
        self.vendors = data["vendors"]
        self.vp = data["vendor_products"]
        self.products = data["products"].set_index("product_id")

    def vendor_stats(self, po_history: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
        d = po_history[po_history.status == "Delivered"].copy()
        # overdue open orders count too: they are already at least this late
        late = po_history[(po_history.status != "Delivered") & (po_history.promised_delivery_date < today)].copy()
        late["delay_days"] = (today - late.promised_delivery_date).dt.days
        late["qty_received"] = late.qty_ordered
        d = pd.concat([d, late], ignore_index=True)
        d["on_time"] = d.delay_days.fillna(0) <= 0
        d["fill"] = d.qty_received / d.qty_ordered
        recent = d[d.order_date >= today - pd.Timedelta(days=60)]
        allv = d.groupby("vendor_id").agg(orders=("po_id", "size"), on_time_pct=("on_time", "mean"),
                                          avg_delay_days=("delay_days", "mean"), fill_rate=("fill", "mean"))
        rec = recent.groupby("vendor_id").agg(recent_on_time_pct=("on_time", "mean"),
                                              recent_avg_delay=("delay_days", "mean"))
        s = allv.join(rec, how="left")
        s["recent_on_time_pct"] = s.recent_on_time_pct.fillna(s.on_time_pct)
        s["recent_avg_delay"] = s.recent_avg_delay.fillna(s.avg_delay_days)
        s["trend"] = np.where(s.recent_on_time_pct < s.on_time_pct - 0.12, "Declining",
                              np.where(s.recent_on_time_pct > s.on_time_pct + 0.08, "Improving", "Stable"))
        # reliability score 0-100 (weights recent behaviour more)
        rel = 0.4 * s.on_time_pct + 0.6 * s.recent_on_time_pct
        s["reliability_score"] = (100 * (0.75 * rel + 0.25 * s.fill_rate)).round(0)
        s = s.reset_index().merge(self.vendors[["vendor_id", "vendor_name", "listed_rating"]], on="vendor_id")
        for c in ["on_time_pct", "recent_on_time_pct", "fill_rate"]:
            s[c] = (100 * s[c]).round(1)
        for c in ["avg_delay_days", "recent_avg_delay"]:
            s[c] = s[c].round(2)
        return s

    def expected_lead(self, vendor_row, stats_row, extra_delay=0):
        """promised lead + learned average delay (recent weighted) + any what-if delay."""
        delay = 0.4 * stats_row.avg_delay_days + 0.6 * stats_row.recent_avg_delay
        return float(vendor_row.promised_lead_time_days + delay + extra_delay)

    def choose(self, pid, qty_fn, shortage_fn, avg_daily, vstats, whatif_delay=None):
        """
        qty_fn(lead_days) -> order quantity needed if this vendor's lead time applies.
        shortage_fn(lead_days) -> expected days without stock before this vendor's delivery lands.
        Returns (best option dict, all options DataFrame).
        """
        whatif_delay = whatif_delay or {}
        p = self.products.loc[pid]
        margin = float(p.selling_price - p.unit_cost)
        vs = vstats.set_index("vendor_id")
        opts = []
        for v in self.vp[self.vp.product_id == pid].itertuples():
            st = vs.loc[v.vendor_id]
            lead = self.expected_lead(v, st, whatif_delay.get(v.vendor_id, 0))
            qty = max(int(qty_fn(lead)), int(v.min_order_qty))
            price = v.unit_price * (1 - v.bulk_discount_pct / 100) if qty >= v.bulk_discount_min_qty else v.unit_price
            purchase = price * qty
            gap = shortage_fn(lead)                                       # days with no stock
            stockout_cost = gap * avg_daily * margin * STOCKOUT_PENALTY_FACTOR
            late_prob = 1 - (0.4 * st.on_time_pct + 0.6 * st.recent_on_time_pct) / 100
            tight = shortage_fn(lead + 2) > gap                            # a 2-day slip would hurt
            late_risk = late_prob * 1.5 * avg_daily * margin * (1 if tight else 0.2)
            opts.append(dict(vendor_id=v.vendor_id, vendor_name=st.vendor_name, qty=qty,
                             unit_price=round(price, 2), bulk_discount=qty >= v.bulk_discount_min_qty,
                             promised_lead=int(v.promised_lead_time_days), expected_lead=round(lead, 1),
                             reliability=st.reliability_score, purchase_cost=round(purchase),
                             stockout_cost=round(stockout_cost), late_risk_cost=round(late_risk)))
        df = pd.DataFrame(opts)
        # compare vendors on the same quantity so a longer lead time (= bigger order) isn't unfairly penalised
        q_ref = df.qty.max()
        df["total_expected_cost"] = (df.unit_price * q_ref + df.stockout_cost + df.late_risk_cost).round()
        df = df.sort_values("total_expected_cost").reset_index(drop=True)
        return df.iloc[0].to_dict(), df

    @staticmethod
    def reason(best, options):
        """Plain-English reason (works without any LLM)."""
        if len(options) < 2:
            return f"{best['vendor_name']} is the only vendor for this product."
        other = options.iloc[1]
        cheaper = other.unit_price < best["unit_price"]
        if cheaper and best["stockout_cost"] < other.stockout_cost:
            return (f"{best['vendor_name']} chosen even though it costs ₹{best['unit_price'] - other.unit_price:.2f}/unit more: "
                    f"it delivers in ~{best['expected_lead']} days vs ~{other.expected_lead} days for {other.vendor_name}, "
                    f"avoiding about ₹{other.stockout_cost - best['stockout_cost']:,.0f} of lost sales.")
        if best["reliability"] > other.reliability + 5:
            return (f"{best['vendor_name']} is cheaper overall and more reliable "
                    f"({best['reliability']:.0f} vs {other.reliability:.0f} reliability score).")
        return (f"{best['vendor_name']} has the lowest total expected cost "
                f"(₹{best['total_expected_cost']:,.0f} vs ₹{other.total_expected_cost:,.0f} for {other.vendor_name}).")
