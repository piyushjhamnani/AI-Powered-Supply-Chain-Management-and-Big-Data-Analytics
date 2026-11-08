"""Warehouse Optimization Agent — moves surplus stock to warehouses about to run out."""
import math
from config import TRANSFER_DAYS, DONOR_KEEP_DAYS


class WarehouseAgent:
    name = "Warehouse Agent"

    def __init__(self, data):
        self.products = data["products"].set_index("product_id")

    def plan_transfers(self, risk_df, log):
        transfers = []
        for pid, g in risk_df.groupby("product_id"):
            p = self.products.loc[pid]
            keep_days = 3 if bool(p.perishable) else DONOR_KEEP_DAYS
            needy = g[g.risk.isin(["CRITICAL", "HIGH"])]
            for n in needy.itertuples():
                need = math.ceil(n.avg_daily_forecast * (n.best_lead_days + 2) - n.on_hand - n.incoming_confirmed)
                if need <= 0:
                    continue
                donors = g[(g.warehouse_id != n.warehouse_id) & g.risk.isin(["LOW", "OVERSTOCK"])].copy()
                donors["spare"] = (donors.on_hand - donors.avg_daily_forecast * keep_days).astype(int)
                donors = donors[donors.spare > 0].sort_values("spare", ascending=False)
                if donors.empty:
                    continue
                d = donors.iloc[0]
                qty = int(min(d.spare, need))
                if qty < max(1, n.avg_daily_forecast):
                    continue
                if TRANSFER_DAYS >= n.best_lead_days and n.on_hand > 0:
                    continue   # a vendor order is just as fast
                cost = round(qty * p.unit_cost * 0.02)   # ~2% handling + freight
                transfers.append(dict(product_id=pid, product_name=n.product_name, from_warehouse=d.warehouse_id,
                                      to_warehouse=n.warehouse_id, qty=qty, transit_days=TRANSFER_DAYS,
                                      est_cost=cost, donor_cover_after=round((d.on_hand - qty) / max(d.avg_daily_forecast, .1), 1),
                                      reason=f"{n.warehouse_id} runs out in {n.days_to_stockout} days; "
                                             f"{d.warehouse_id} has {d.days_cover:.0f} days of stock. "
                                             f"Transfer arrives in {TRANSFER_DAYS} day vs ~{n.best_lead_days} days from a vendor."))
                log.add(self.name, f"Transfer {qty} × {n.product_name}: {d.warehouse_id} → {n.warehouse_id} "
                                   f"(arrives tomorrow, ≈ ₹{cost:,} handling)", "action", n.warehouse_id, pid)
        if not transfers:
            log.add(self.name, "No inter-warehouse transfers needed", "ok")
        return transfers
