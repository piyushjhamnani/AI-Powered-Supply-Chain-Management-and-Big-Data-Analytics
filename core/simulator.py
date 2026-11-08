"""
Actions (approve PO / transfer) and the day-by-day simulator that closes the agentic loop:
    Detect -> Decide -> Order -> Track -> Adjust
Each simulated day: approved orders enter the pipeline, deliveries arrive (or slip, based on each
vendor's real reliability), customers buy (demand drawn around the forecast) and stock updates.
"""
import numpy as np
import pandas as pd


def approve_po(state, po: dict):
    eta = pd.Timestamp(po["expected_delivery"])
    state.pipeline = pd.concat([state.pipeline, pd.DataFrame([{
        "ref": po["po_id"], "warehouse_id": po["warehouse_id"], "product_id": po["product_id"],
        "source": po["vendor_id"], "qty": po["qty"], "order_date": state.today, "eta": eta, "kind": "PO"}])],
        ignore_index=True)
    state.po_history = pd.concat([state.po_history, pd.DataFrame([{
        "po_id": po["po_id"], "order_date": state.today, "warehouse_id": po["warehouse_id"],
        "product_id": po["product_id"], "vendor_id": po["vendor_id"], "qty_ordered": po["qty"],
        "unit_cost": po["unit_price"], "total_cost": po["total_cost"], "promised_delivery_date": eta,
        "actual_delivery_date": pd.NaT, "qty_received": np.nan, "delay_days": np.nan, "status": "In Transit"}])],
        ignore_index=True)
    state.approved.append({**{k: v for k, v in po.items() if k != "options"}, "status": "Approved",
                           "approved_on": state.today.date()})


def approve_transfer(state, tr: dict):
    src = (tr["from_warehouse"], tr["product_id"])
    qty = int(min(tr["qty"], state.stock.get(src, 0)))
    if qty <= 0:
        return
    state.stock[src] -= qty
    ref = f"TR-{state.today:%y%m%d}-{len(state.transfers_done) + 1:02d}"
    state.pipeline = pd.concat([state.pipeline, pd.DataFrame([{
        "ref": ref, "warehouse_id": tr["to_warehouse"], "product_id": tr["product_id"], "source": tr["from_warehouse"],
        "qty": qty, "order_date": state.today, "eta": state.today + pd.Timedelta(days=tr["transit_days"]),
        "kind": "Transfer"}])], ignore_index=True)
    state.transfers_done.append({**tr, "qty": qty, "ref": ref, "approved_on": state.today.date()})


def advance_day(state, result, products, rng, whatif=None, autopilot=False, pending=None):
    """Simulate one day. Returns a summary dict."""
    whatif = whatif or {}
    t = state.today
    approved_now = []
    if autopilot and pending is not None:
        for po in list(pending):
            if not po["needs_human"] and po["status"] == "Pending approval":
                approve_po(state, po); po["status"] = "Approved (autopilot)"; approved_now.append(po["po_id"])
        for tr in result.transfers:
            approve_transfer(state, tr)

    vstats = result.vendor_stats.set_index("vendor_id")
    extra_delay = whatif.get("vendor_delay", {})
    arrived, slipped = [], []
    keep = []
    for x in state.pipeline.itertuples(index=False):
        due = x.eta <= t
        if not due:
            keep.append(x); continue
        if x.kind == "Transfer":
            ok = True
        elif x.source in extra_delay and (t - x.eta).days < extra_delay[x.source]:
            ok = False
        else:
            p_on = vstats.loc[x.source].recent_on_time_pct / 100 if x.source in vstats.index else 0.85
            ok = rng.random() < (p_on if x.eta == t else 0.55)
        if ok:
            key = (x.warehouse_id, x.product_id)
            state.stock[key] = state.stock.get(key, 0) + int(x.qty)
            arrived.append(x.ref)
            if x.kind == "PO":
                m = state.po_history.po_id == x.ref
                state.po_history.loc[m, ["status", "actual_delivery_date", "qty_received", "delay_days"]] = \
                    ["Delivered", t, x.qty, max(0, (t - x.eta).days)]
        else:
            slipped.append(x.ref); keep.append(x)
    state.pipeline = pd.DataFrame(keep, columns=state.pipeline.columns)

    fc_today = result.forecast[result.forecast.date == t].set_index(["warehouse_id", "product_id"]).forecast
    price = products.set_index("product_id").selling_price
    rows, lost, revenue = [], 0, 0.0
    for key, qty in state.stock.items():
        mu = float(fc_today.get(key, 0.0))
        demand = int(rng.poisson(rng.gamma(35, max(mu, 0.01) / 35)))
        sold = min(demand, qty)
        state.stock[key] = qty - sold
        lost += demand - sold
        rev = sold * float(price[key[1]])
        revenue += rev
        rows.append(dict(date=t, warehouse_id=key[0], product_id=key[1], units_sold=sold,
                         unit_price=float(price[key[1]]), on_promotion=False, stockout=demand > qty, revenue=rev))
    new = pd.DataFrame(rows)
    state.sales = pd.concat([state.sales, new], ignore_index=True)
    summary = dict(date=t.date(), units_sold=int(new.units_sold.sum()), lost_sales_units=int(lost),
                   revenue=round(revenue), stockout_positions=int(new.stockout.sum()),
                   deliveries_arrived=len(arrived), deliveries_slipped=len(slipped),
                   autopilot_approved=len(approved_now))
    state.day_log.append(summary)
    state.today = t + pd.Timedelta(days=1)
    return summary
