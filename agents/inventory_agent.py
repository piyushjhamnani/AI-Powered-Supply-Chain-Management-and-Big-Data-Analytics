"""Inventory Tracking Agent: one unified view of stock across all warehouses."""
import pandas as pd


class InventoryAgent:
    name = "Inventory Agent"

    def __init__(self, data):
        self.products = data["products"]
        self.warehouses = data["warehouses"]

    def positions(self, state) -> pd.DataFrame:
        rows = []
        for (wh, pid), qty in state.stock.items():
            inc = state.incoming(wh, pid)
            late = inc[inc.eta < state.today]
            on_time = inc[inc.eta >= state.today]
            rows.append(dict(warehouse_id=wh, product_id=pid, on_hand=int(qty),
                             incoming_confirmed=int(on_time.qty.sum()), incoming_delayed=int(late.qty.sum()),
                             next_arrival=on_time.eta.min() if len(on_time) else pd.NaT))
        df = pd.DataFrame(rows).merge(
            self.products[["product_id", "product_name", "unit_cost", "selling_price", "shelf_life_days", "perishable"]],
            on="product_id").merge(self.warehouses[["warehouse_id", "city"]], on="warehouse_id")
        df["stock_value"] = df.on_hand * df.unit_cost
        return df
