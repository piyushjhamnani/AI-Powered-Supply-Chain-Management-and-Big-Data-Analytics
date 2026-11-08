"""
Reorder Decision Agent  — WHEN and HOW MUCH to order.
Policy: periodic review, order-up-to level
    need = forecast demand over (lead time + review period)
         + safety stock  (z * sigma * sqrt(lead + review))      <- dynamic: grows with volatility
         - (stock on hand + confirmed deliveries arriving in that window)
Perishables are capped so we never order more than can be sold before expiry.
"""
import math
import numpy as np
import pandas as pd
from config import SERVICE_LEVEL_Z, REVIEW_DAYS, product_group


class ReorderAgent:
    name = "Reorder Agent"

    def __init__(self, data):
        self.products = data["products"].set_index("product_id")

    def make_qty_fn(self, state, row, daily_fc, z=SERVICE_LEVEL_Z, planned_transfer=0):
        p = self.products.loc[row.product_id]
        R = REVIEW_DAYS[product_group(row.product_id)]
        inc = state.incoming(row.warehouse_id, row.product_id).copy()
        # trust on-time orders fully; count orders 1 day overdue at 50% (they may still turn up);
        # ignore later ones so we don't wait on a truck that isn't coming
        overdue = (state.today - inc.eta).dt.days
        inc = inc[overdue <= 1]
        inc["qty"] = inc.qty.where(overdue[overdue <= 1] <= 0, inc.qty * 0.5)
        inc["eta"] = inc.eta.where(inc.eta >= state.today, state.today + pd.Timedelta(days=1))
        avg = float(np.mean(daily_fc)) if len(daily_fc) else 0.0

        def demand_between(a, b):   # forecast units for day a..b-1 (fractional days ok)
            total, d = 0.0, a
            while d < b:
                step = min(1.0, b - d)
                i = int(d)
                total += (daily_fc[i] if i < len(daily_fc) else avg) * step
                d += step
            return total

        def qty_fn(lead):
            window = lead + R
            ss = z * row.sigma * math.sqrt(window)
            arriving = inc[(inc.eta - state.today).dt.days < window].qty.sum()
            need = demand_between(0, window) + ss - row.on_hand - arriving - planned_transfer
            if bool(p.perishable):
                need = min(need, demand_between(lead, lead + p.shelf_life_days - 0.5))
            if need <= 0:
                return 0
            m = int(p.order_multiple)
            return int(math.ceil(need / m) * m)

        def trigger(lead):
            """Reorder point check: will stock + arrivals fall below safety stock before a new order lands?"""
            ss = z * row.sigma * math.sqrt(lead)
            arriving = inc[(inc.eta - state.today).dt.days < lead].qty.sum()
            return demand_between(0, lead) + ss - row.on_hand - arriving - planned_transfer > 0

        return qty_fn, trigger, R

    def decide(self, state, risk_df, fc, log, transfers=()):
        planned = {}
        for t in transfers:
            planned[(t["to_warehouse"], t["product_id"])] = planned.get((t["to_warehouse"], t["product_id"]), 0) + t["qty"]
            planned[(t["from_warehouse"], t["product_id"])] = planned.get((t["from_warehouse"], t["product_id"]), 0) - t["qty"]
        fcg = {k: g.sort_values("date").forecast.values for k, g in fc.groupby(["warehouse_id", "product_id"])}
        decisions = []
        for r in risk_df.itertuples():
            if r.risk == "OVERSTOCK":
                continue
            daily = fcg[(r.warehouse_id, r.product_id)]
            qty_fn, trigger, R = self.make_qty_fn(state, r, daily,
                                                  planned_transfer=planned.get((r.warehouse_id, r.product_id), 0))
            q = qty_fn(r.best_lead_days)
            if q <= 0:
                continue
            # fresh items are topped up daily; for others avoid tiny "dribble" orders unless it's urgent
            batch = r.avg_daily_forecast * R * 0.5
            if R > 1 and q < batch and r.risk not in ("CRITICAL", "HIGH") and not trigger(r.best_lead_days):
                continue
            decisions.append(dict(row=r, qty_fn=qty_fn, review_days=R, base_qty=q))
            if r.risk in ("CRITICAL", "HIGH"):
                log.add(self.name, f"{r.product_name} @ {r.warehouse_id}: reorder needed now — "
                                   f"~{q} units to cover lead time + {R}-day review + safety stock",
                        "action", r.warehouse_id, r.product_id)
        log.add(self.name, f"{len(decisions)} items need replenishment today", "info")
        return decisions


class CostAgent:
    """Cost Optimization Agent — bulk discounts vs holding cost, and overstock clean-up."""
    name = "Cost Agent"

    def __init__(self, data):
        self.products = data["products"].set_index("product_id")
        self.vp = data["vendor_products"]

    def optimise_qty(self, pid, vendor_id, qty, avg_daily, log, wh):
        p = self.products.loc[pid]
        v = self.vp[(self.vp.vendor_id == vendor_id) & (self.vp.product_id == pid)].iloc[0]
        if bool(p.perishable) or qty >= v.bulk_discount_min_qty or avg_daily <= 0:
            return qty
        extra = v.bulk_discount_min_qty - qty
        if extra > 0.3 * qty:
            return qty
        saving = v.bulk_discount_min_qty * v.unit_price * v.bulk_discount_pct / 100
        extra_days = extra / avg_daily
        holding = extra * p.unit_cost * (p.holding_cost_pct_per_year / 100) * extra_days / 365 * 2
        if saving > holding and (qty + extra) / avg_daily < 35:
            log.add(self.name, f"{p.product_name} @ {wh}: rounding {qty} → {int(v.bulk_discount_min_qty)} units unlocks "
                               f"{v.bulk_discount_pct}% bulk discount (saves ₹{saving:,.0f}, extra holding ≈ ₹{holding:,.0f})",
                    "ok", wh, pid)
            return int(v.bulk_discount_min_qty)
        return qty

    def overstock_actions(self, risk_df, log):
        out = []
        for r in risk_df[risk_df.risk == "OVERSTOCK"].itertuples():
            p = self.products.loc[r.product_id]
            excess = max(0, r.on_hand - int(r.avg_daily_forecast * 20))
            locked = excess * p.unit_cost
            monthly_hold = locked * p.holding_cost_pct_per_year / 100 / 12
            out.append(dict(warehouse_id=r.warehouse_id, product_id=r.product_id, product_name=r.product_name,
                            days_cover=r.days_cover, excess_units=excess, capital_locked=round(locked),
                            holding_cost_per_month=round(monthly_hold),
                            suggestion="Transfer to a warehouse that needs it, else run a 10–15% clearance offer "
                                       "and pause reorders"))
            log.add(self.name, f"{r.product_name} @ {r.warehouse_id}: {r.days_cover:.0f} days of stock, "
                               f"₹{locked:,.0f} locked up (≈ ₹{monthly_hold:,.0f}/month holding cost)",
                    "warn", r.warehouse_id, r.product_id)
        return out
