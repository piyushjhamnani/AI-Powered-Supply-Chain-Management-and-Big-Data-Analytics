"""
Stockout Risk Detection Agent
Walks forward day by day: stock - forecast demand + confirmed deliveries,
to find exactly when each item runs out, then compares that with how fast
the best vendor can realistically deliver.
Delayed (overdue) deliveries are NOT trusted in the calculation — they are reported separately.
"""
import numpy as np
import pandas as pd
from config import OVERSTOCK_DAYS


def days_until_stockout(on_hand, daily_fc, arrivals):
    """arrivals: dict day_index -> qty. Returns fractional days until stock hits zero."""
    stock = float(on_hand)
    for d, need in enumerate(daily_fc):
        stock += arrivals.get(d, 0)
        if need <= 0:
            continue
        if stock < need:
            return d + stock / need
        stock -= need
    avg = float(np.mean(daily_fc)) if len(daily_fc) else 0
    return len(daily_fc) + (stock / avg if avg > 0 else 99)


def shortage_days(on_hand, daily_fc, arrivals, horizon):
    """Days of unmet demand (lost-sales days) between now and `horizon` days ahead."""
    stock, short, d = float(on_hand), 0.0, 0
    while d < horizon:
        w = min(1.0, horizon - d)
        stock += arrivals.get(d, 0)
        need = (daily_fc[d] if d < len(daily_fc) else (daily_fc[-1] if len(daily_fc) else 0)) * w
        if need > 0:
            if stock >= need:
                stock -= need
            else:
                short += w * (need - stock) / need
                stock = 0
        d += 1
    return short


class RiskAgent:
    name = "Risk Agent"

    @staticmethod
    def arrivals(state, wh, pid, extra=None):
        """Confirmed (not overdue) deliveries as {day_index: qty}, plus optional extra arrivals."""
        inc = state.incoming(wh, pid)
        out = dict(extra or {})
        for x in inc[inc.eta >= state.today].itertuples():
            di = int((x.eta - state.today).days)
            out[di] = out.get(di, 0) + x.qty
        return out

    def assess(self, state, positions, fc, stats, best_lead):
        """best_lead: dict product_id -> best expected vendor lead time (days)."""
        rows = []
        fcg = {k: g.sort_values("date").forecast.values for k, g in fc.groupby(["warehouse_id", "product_id"])}
        sig = stats.set_index(["warehouse_id", "product_id"]).sigma.to_dict()
        for r in positions.itertuples():
            key = (r.warehouse_id, r.product_id)
            daily = fcg.get(key, np.array([0.0]))
            arrivals = self.arrivals(state, *key)
            dts = days_until_stockout(r.on_hand, daily, arrivals)
            avg7 = float(np.mean(daily[:7]))
            cover = r.on_hand / avg7 if avg7 > 0 else 99
            lead = best_lead[r.product_id]
            sigma = sig.get(key, avg7 * 0.3)
            if r.on_hand <= 0 or dts < 1:
                risk, why = "CRITICAL", "Out of stock or runs out within a day"
            elif dts < lead + 1:
                risk, why = "HIGH", f"Runs out in {dts:.1f} days, fastest realistic delivery ~{lead:.1f} days"
            elif dts < lead + 3:
                risk, why = "MEDIUM", f"Runs out in {dts:.1f} days — reorder window is now"
            elif (not r.perishable and cover > OVERSTOCK_DAYS) or (r.perishable and cover > r.shelf_life_days):
                risk, why = "OVERSTOCK", f"{cover:.0f} days of stock — capital locked up"
            else:
                risk, why = "LOW", f"{cover:.0f} days of cover"
            if r.incoming_delayed > 0:
                why += f" · {r.incoming_delayed} units overdue from vendor"
            rows.append(dict(warehouse_id=r.warehouse_id, product_id=r.product_id, product_name=r.product_name,
                             city=r.city, on_hand=r.on_hand, incoming_confirmed=r.incoming_confirmed,
                             incoming_delayed=r.incoming_delayed, avg_daily_forecast=round(avg7, 1),
                             forecast_28d=round(float(np.sum(daily))), days_cover=round(cover, 1),
                             days_to_stockout=round(dts, 1), best_lead_days=round(lead, 1),
                             sigma=round(sigma, 2), risk=risk, reason=why))
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "OVERSTOCK": 3, "LOW": 4}
        df = pd.DataFrame(rows)
        df["_o"] = df.risk.map(order)
        return df.sort_values(["_o", "days_to_stockout"]).drop(columns="_o").reset_index(drop=True)
