"""
Supply Chain Orchestrator — runs the agents in order and collects their outputs:
Inventory -> Forecast -> Vendor stats -> Risk -> Reorder -> Vendor choice -> Cost -> PO
          -> Warehouse transfers -> Overstock actions -> Anomalies -> Briefing
"""
from dataclasses import dataclass
import pandas as pd

from agents.base import AgentLog
from agents.inventory_agent import InventoryAgent
from agents.forecast_agent import ForecastAgent
from agents.vendor_agent import VendorAgent
from agents.risk_agent import RiskAgent, shortage_days
from config import TRANSFER_DAYS
from agents.reorder_agent import ReorderAgent, CostAgent
from agents.po_agent import POAgent
from agents.warehouse_agent import WarehouseAgent
from agents.anomaly_agent import AnomalyAgent
from llm import explainer


@dataclass
class RunResult:
    today: pd.Timestamp
    positions: pd.DataFrame
    forecast: pd.DataFrame
    forecast_stats: pd.DataFrame
    festival_watch: pd.DataFrame
    vendor_stats: pd.DataFrame
    risk: pd.DataFrame
    purchase_orders: list
    transfers: list
    overstock: list
    anomalies: list
    log: AgentLog
    kpis: dict
    briefing: str


class Orchestrator:
    name = "Orchestrator"

    def __init__(self, data, forecaster: ForecastAgent):
        self.data = data
        self.inventory = InventoryAgent(data)
        self.forecaster = forecaster
        self.vendor = VendorAgent(data)
        self.risk = RiskAgent()
        self.reorder = ReorderAgent(data)
        self.cost = CostAgent(data)
        self.po = POAgent(data)
        self.warehouse = WarehouseAgent(data)
        self.anomaly = AnomalyAgent(data)

    def run(self, state, whatif=None, ai_briefing=False) -> RunResult:
        whatif = whatif or {}
        vendor_delay = whatif.get("vendor_delay", {})          # {"V02": 3}
        demand_mult = whatif.get("demand_mult", {})            # {"ALL": 1.2} or {"P001": 1.5}
        log = AgentLog()
        t = state.today
        log.add(self.name, f"Starting planning cycle for {t:%a %d %b %Y}"
                + (" — WHAT-IF scenario active" if (vendor_delay or demand_mult) else ""), "info")

        # 1. inventory
        pos = self.inventory.positions(state)
        log.add(self.inventory.name, f"Tracked {len(pos)} stock positions across {pos.warehouse_id.nunique()} warehouses "
                                     f"· stock value ₹{pos.stock_value.sum() / 1e5:.1f} lakh · "
                                     f"{(pos.on_hand == 0).sum()} positions at zero", "info")

        # 2. forecast
        fc, stats = self.forecaster.forecast(state.sales, t)
        if demand_mult:
            mult = fc.product_id.map(lambda p: demand_mult.get(p, demand_mult.get("ALL", 1.0)))
            fc["forecast"] = fc.forecast * mult
        log.add(self.forecaster.name, f"Forecast next 28 days for {len(stats)} series "
                                      f"(ML model, ~{self.forecaster.metrics['ai_accuracy'] if self.forecaster.metrics else '—'}% accurate on holdout)",
                "info")
        hot = stats[stats.level_factor >= 1.2]
        for r in hot.itertuples():
            pname = self.data["products"].set_index("product_id").product_name[r.product_id]
            log.add(self.forecaster.name, f"{pname} @ {r.warehouse_id}: last week ran {r.recent_ratio:.1f}× "
                                          f"above model — adapting forecast ×{r.level_factor:.2f}", "warn",
                    r.warehouse_id, r.product_id)
        fw = self.forecaster.festival_watch(state.sales, t)
        if not fw.empty:
            for (fest, days), g in fw.groupby(["festival", "days_away"]):
                ups = g[g.last_year_uplift > 1]
                if len(ups):
                    items = ", ".join(f"{x.product_name} ×{x.last_year_uplift}" for x in ups.head(4).itertuples())
                    log.add(self.forecaster.name, f"{fest} in {days} days — last year: {items}", "info")

        # 3. vendors
        vstats = self.vendor.vendor_stats(state.po_history, t)
        for v in vstats[vstats.trend == "Declining"].itertuples():
            log.add(self.vendor.name, f"{v.vendor_name} reliability falling: {v.recent_on_time_pct:.0f}% on-time "
                                      f"(last 60 days) vs {v.on_time_pct:.0f}% overall", "warn")
        vs_idx = vstats.set_index("vendor_id")
        best_lead = {}
        for pid, g in self.data["vendor_products"].groupby("product_id"):
            best_lead[pid] = min(self.vendor.expected_lead(v, vs_idx.loc[v.vendor_id], vendor_delay.get(v.vendor_id, 0))
                                 for v in g.itertuples())

        # 4. risk
        risk = self.risk.assess(state, pos, fc, stats, best_lead)
        for r in risk[risk.risk.isin(["CRITICAL", "HIGH"])].itertuples():
            log.add(self.risk.name, f"{r.risk}: {r.product_name} @ {r.warehouse_id} — {r.reason}",
                    "critical", r.warehouse_id, r.product_id)
        counts = risk.risk.value_counts().to_dict()
        log.add(self.risk.name, "Risk summary: " + ", ".join(f"{k} {v}" for k, v in counts.items()), "info")

        # 5. transfers first (fastest fix), then reorders
        transfers = self.warehouse.plan_transfers(risk, log)
        decisions = self.reorder.decide(state, risk, fc, log, transfers)

        # 6. vendor choice + cost optimisation + PO
        pos_list = []
        fcg = {k: g.sort_values("date").forecast.values for k, g in fc.groupby(["warehouse_id", "product_id"])}
        incoming_transfer = {(x["to_warehouse"], x["product_id"]): x["qty"] for x in transfers}
        for i, d in enumerate(decisions, start=1):
            r = d["row"]
            extra = {}
            if (r.warehouse_id, r.product_id) in incoming_transfer:   # a planned transfer buys time
                extra = {TRANSFER_DAYS: incoming_transfer[(r.warehouse_id, r.product_id)]}
            arr = self.risk.arrivals(state, r.warehouse_id, r.product_id, extra)
            daily = fcg[(r.warehouse_id, r.product_id)]
            shortage_fn = (lambda oh, dl, ar: (lambda lead: shortage_days(oh, dl, ar, lead)))(r.on_hand, daily, arr)
            best, options = self.vendor.choose(r.product_id, d["qty_fn"], shortage_fn,
                                               r.avg_daily_forecast, vstats, vendor_delay)
            qty = self.cost.optimise_qty(r.product_id, best["vendor_id"], int(best["qty"]),
                                         r.avg_daily_forecast, log, r.warehouse_id)
            if qty != best["qty"]:
                v = self.data["vendor_products"]
                vr = v[(v.vendor_id == best["vendor_id"]) & (v.product_id == r.product_id)].iloc[0]
                best["unit_price"] = round(vr.unit_price * (1 - vr.bulk_discount_pct / 100), 2)
            reason = VendorAgent.reason(best, options)
            po = self.po.create(t, i, r, best, qty, reason, options)
            pos_list.append(po)
            cheapest = options.sort_values("unit_price").iloc[0].vendor_id
            if r.risk in ("CRITICAL", "HIGH") or (best["vendor_id"] != cheapest and best["stockout_cost"] > 0):
                log.add(self.vendor.name, f"{r.product_name} @ {r.warehouse_id}: {reason}", "action",
                        r.warehouse_id, r.product_id)
        if pos_list:
            total = sum(p["total_cost"] for p in pos_list)
            log.add(self.po.name, f"Generated {len(pos_list)} purchase orders worth ₹{total / 1e5:.2f} lakh "
                                  f"({sum(p['needs_human'] for p in pos_list)} need manager approval)", "action")

        # 7. overstock + anomalies
        overstock = self.cost.overstock_actions(risk, log)
        anomalies = self.anomaly.detect(state, vstats, log)

        kpis = dict(
            critical=int((risk.risk == "CRITICAL").sum()), high=int((risk.risk == "HIGH").sum()),
            medium=int((risk.risk == "MEDIUM").sum()), overstock=int((risk.risk == "OVERSTOCK").sum()),
            stockouts_now=int((risk.on_hand == 0).sum()), po_count=len(pos_list),
            po_value=round(sum(p["total_cost"] for p in pos_list)),
            inventory_value=round(pos.stock_value.sum()),
            locked_capital=sum(o["capital_locked"] for o in overstock),
            anomalies=len(anomalies), transfers=len(transfers),
        )
        fallback = self._template_briefing(risk, pos_list, transfers, anomalies, fw, kpis)
        briefing = fallback
        if ai_briefing:
            ctx = dict(date=str(t.date()), kpis=kpis,
                       top_risks=risk[risk.risk.isin(["CRITICAL", "HIGH"])][
                           ["warehouse_id", "product_name", "on_hand", "days_to_stockout", "reason"]].to_dict("records"),
                       transfers=[{k: x[k] for k in ("product_name", "from_warehouse", "to_warehouse", "qty")} for x in transfers],
                       anomalies=[a["detail"] for a in anomalies[:6]],
                       festivals=fw.head(8).to_dict("records") if not fw.empty else [])
            briefing = explainer.daily_briefing(ctx, fallback)
        log.add(self.name, "Planning cycle complete", "ok")
        return RunResult(t, pos, fc, stats, fw, vstats, risk, pos_list, transfers, overstock, anomalies, log, kpis, briefing)

    @staticmethod
    def _template_briefing(risk, pos_list, transfers, anomalies, fw, k):
        lines = []
        crit = risk[risk.risk.isin(["CRITICAL", "HIGH"])]
        if len(crit):
            top = "; ".join(f"{r.product_name} @ {r.warehouse_id} ({r.days_to_stockout} days)" for r in crit.head(3).itertuples())
            lines.append(f"- **{len(crit)} items at high stockout risk.** Most urgent: {top}.")
        if transfers:
            lines.append(f"- **{len(transfers)} stock transfer(s) recommended** — fastest fix, arrives tomorrow.")
        if pos_list:
            lines.append(f"- **{len(pos_list)} purchase orders drafted** worth ₹{k['po_value'] / 1e5:.2f} lakh, waiting for approval.")
        if k["overstock"]:
            lines.append(f"- **{k['overstock']} overstocked items** lock ₹{k['locked_capital'] / 1e5:.2f} lakh of capital.")
        hi = [a for a in anomalies if a["severity"] == "HIGH"]
        if hi:
            lines.append(f"- **{len(hi)} high-severity anomalies**, e.g. {hi[0]['type'].lower()}: {hi[0]['detail']}.")
        if not fw.empty:
            f = fw.sort_values("days_away").iloc[0]
            lines.append(f"- **Coming up:** {f.festival} in {f.days_away} days — plan for higher demand.")
        return "\n".join(lines) or "- All clear: no urgent risks today."
