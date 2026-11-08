"""Loads the CSV data and holds the live 'world state' the agents work on."""
import os
from dataclasses import dataclass, field
import pandas as pd
from config import DATA_DIR


def load_data() -> dict:
    def rd(name, dates=()):
        return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), parse_dates=list(dates))

    return {
        "products": rd("products"),
        "warehouses": rd("warehouses"),
        "vendors": rd("vendors"),
        "vendor_products": rd("vendor_products"),
        "calendar": rd("calendar", ["date"]),
        "promotions": rd("promotions", ["start_date", "end_date"]),
        "sales": rd("sales_history", ["date"]),
        "inventory_daily": rd("inventory_daily", ["date"]),
        "snapshot": rd("inventory_snapshot", ["as_of_date"]),
        "purchase_orders": rd("purchase_orders", ["order_date", "promised_delivery_date", "actual_delivery_date"]),
        "adjustments": rd("inventory_adjustments", ["date"]),
    }


@dataclass
class State:
    """Everything that changes while the system runs (stock, open orders, sales so far)."""
    today: pd.Timestamp
    stock: dict                     # (warehouse_id, product_id) -> units on hand
    pipeline: pd.DataFrame          # open purchase orders / transfers not yet received
    sales: pd.DataFrame             # sales history (grows when we simulate days)
    po_history: pd.DataFrame        # all purchase orders (for vendor scoring)
    approved: list = field(default_factory=list)   # POs approved in this session
    rejected: list = field(default_factory=list)
    transfers_done: list = field(default_factory=list)
    day_log: list = field(default_factory=list)    # what happened on each simulated day

    @classmethod
    def from_data(cls, data: dict) -> "State":
        snap = data["snapshot"]
        today = snap.as_of_date.max() + pd.Timedelta(days=1)
        stock = {(r.warehouse_id, r.product_id): int(r.stock_on_hand) for r in snap.itertuples()}
        po = data["purchase_orders"].copy()
        open_po = po[po.status.isin(["In Transit", "Delayed"])].copy()
        pipeline = pd.DataFrame({
            "ref": open_po.po_id, "warehouse_id": open_po.warehouse_id, "product_id": open_po.product_id,
            "source": open_po.vendor_id, "qty": open_po.qty_ordered, "order_date": open_po.order_date,
            "eta": open_po.promised_delivery_date, "kind": "PO",
        }).reset_index(drop=True)
        return cls(today=today, stock=stock, pipeline=pipeline, sales=data["sales"].copy(), po_history=po)

    # helpers -------------------------------------------------------------------
    def incoming(self, wh, pid):
        p = self.pipeline
        return p[(p.warehouse_id == wh) & (p.product_id == pid)]
