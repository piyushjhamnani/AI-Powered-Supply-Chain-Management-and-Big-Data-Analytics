"""
Supply Chain & Inventory Management Agent — Streamlit dashboard
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.state import load_data, State
from core.simulator import advance_day, approve_po, approve_transfer
from agents.forecast_agent import ForecastAgent
from agents.orchestrator import Orchestrator
from agents.base import AGENT_ICONS
from llm import explainer
from ui import inject_css, kpi_strip, alert, agent_log, inr, RISK_COLOR, STEEL, YELLOW, MUTED, INK
from bdsma.streaming import build_event_batch, summarize_events

st.set_page_config(page_title="Supply Chain Agent", page_icon="📦", layout="wide")
inject_css()


# ------------------------------------------------------------------ setup (cached)
@st.cache_resource(show_spinner="Loading 2 years of data…")
def get_data():
    return load_data()


@st.cache_resource(show_spinner="Training the demand forecasting model…")
def get_forecaster(_data):
    fa = ForecastAgent(_data)
    fa.fit(_data["sales"])
    fa.evaluate(_data["sales"])
    return fa


data = get_data()
forecaster = get_forecaster(data)
orch = Orchestrator(data, forecaster)
PRODUCTS = data["products"].set_index("product_id")
WAREHOUSES = data["warehouses"].set_index("warehouse_id")
VENDOR_NAMES = data["vendors"].set_index("vendor_id").vendor_name.to_dict()


def run_agents():
    ss = st.session_state
    wi = ss.whatif if ss.whatif_active else None
    ss.result = orch.run(ss.state, wi)
    ss.pending = [p for p in ss.result.purchase_orders if p["po_id"] not in ss.state.rejected]


def reset():
    ss = st.session_state
    ss.state = State.from_data(data)
    ss.rng = np.random.default_rng(2026)
    ss.whatif = {}
    ss.whatif_active = False
    ss.ai_briefing = {}
    ss.chat = []
    run_agents()


if "state" not in st.session_state:
    reset()

ss = st.session_state
S, R = ss.state, ss.result


def simulate(days):
    for _ in range(days):
        advance_day(ss.state, ss.result, data["products"], ss.rng,
                    ss.whatif if ss.whatif_active else None, ss.autopilot, ss.pending)
        run_agents()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📦 Sahyadri Supply Agent")
    st.markdown(f"**{S.today:%A, %d %B %Y}**")
    st.caption("3 warehouses · 14 products · 6 vendors")
    page = st.radio("Go to", ["Control room", "Agent activity", "Demand forecast", "Purchase orders",
                              "Transfers & alerts", "Vendors", "What-if lab", "Simulation", "Stream analytics", "Ask the agent"],
                    label_visibility="collapsed")
    st.divider()
    ss.autopilot = st.toggle("Autopilot", value=ss.get("autopilot", False),
                             help="When you simulate, routine orders under ₹50,000 and all transfers are approved "
                                  "automatically. Bigger orders still wait for you.")
    c1, c2 = st.columns(2)
    if c1.button("Next day ⏩", width="stretch"):
        simulate(1); st.rerun()
    if c2.button("Next 7 days", width="stretch"):
        with st.spinner("Simulating a week…"):
            simulate(7)
        st.rerun()
    if st.button("Reset to 6 Oct 2026", width="stretch"):
        reset(); st.rerun()
    if ss.whatif_active:
        st.warning("What-if scenario is applied to the live plan.")
    st.divider()
    st.caption(explainer.status_text())

# ------------------------------------------------------------------ pages
if page == "Control room":
    st.title("Control room")
    st.markdown(f'<p class="note">Plan for {S.today:%d %b %Y}, built by 9 agents from live stock, open orders and a '
                f'28-day demand forecast.</p>', unsafe_allow_html=True)
    k = R.kpis
    kpi_strip([
        (k["critical"] + k["high"], "items at high stockout risk", k["critical"] + k["high"] > 0),
        (k["stockouts_now"], "out of stock right now", k["stockouts_now"] > 0),
        (len(ss.pending), "orders waiting for approval", False),
        (inr(sum(p["total_cost"] for p in ss.pending)), "value of pending orders", False),
        (inr(k["inventory_value"]), "stock on hand", False),
        (f'{forecaster.metrics["ai_accuracy"]}%', "forecast accuracy", False),
    ])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Needs attention")
        urgent = R.risk[R.risk.risk.isin(["CRITICAL", "HIGH"])]
        if urgent.empty:
            st.success("No item is at high risk of running out.")
        for r in urgent.itertuples():
            alert(r.risk, f"{r.product_name} · {WAREHOUSES.loc[r.warehouse_id].city}",
                  f"{r.on_hand} on hand, ~{r.avg_daily_forecast:.0f}/day expected. {r.reason}")
        for o in R.overstock:
            alert("OVERSTOCK", f"{o['product_name']} · {WAREHOUSES.loc[o['warehouse_id']].city}",
                  f"{o['days_cover']:.0f} days of stock, {inr(o['capital_locked'])} tied up.")
    with right:
        st.subheader("Today's briefing")
        key = str(S.today.date())
        st.markdown(ss.ai_briefing.get(key, R.briefing))
        if explainer.provider() and key not in ss.ai_briefing:
            if st.button("Rewrite briefing with AI"):
                with st.spinner("Thinking…"):
                    ss.ai_briefing[key] = orch.run(S, ss.whatif if ss.whatif_active else None, ai_briefing=True).briefing
                st.rerun()
        st.subheader("Coming up")
        fw = R.festival_watch
        if fw.empty:
            st.caption("No major festivals in the next 45 days.")
        else:
            for (fest, days), g in fw.groupby(["festival", "days_away"], sort=False):
                ups = [f"{x.product_name} ×{x.last_year_uplift}" for x in g.itertuples() if x.last_year_uplift > 1][:4]
                downs = [f"{x.product_name} ×{x.last_year_uplift}" for x in g.sort_values("last_year_uplift").itertuples()
                         if x.last_year_uplift < 1][:2]
                txt = f"**{fest}** in {days} days."
                if ups:
                    txt += " Last year: " + ", ".join(ups) + "."
                if downs:
                    txt += " Drops: " + ", ".join(downs) + "."
                st.markdown(txt)

    st.subheader("All stock positions")
    f1, f2 = st.columns(2)
    wh_sel = f1.multiselect("Warehouse", list(WAREHOUSES.index), default=list(WAREHOUSES.index))
    rk_sel = f2.multiselect("Risk", list(RISK_COLOR), default=list(RISK_COLOR))
    view = R.risk[R.risk.warehouse_id.isin(wh_sel) & R.risk.risk.isin(rk_sel)]
    st.dataframe(view[["risk", "warehouse_id", "product_name", "on_hand", "incoming_confirmed", "incoming_delayed",
                       "avg_daily_forecast", "days_cover", "days_to_stockout", "reason"]],
                 hide_index=True, width="stretch",
                 column_config={"risk": "Risk", "warehouse_id": "Warehouse", "product_name": "Product",
                                "on_hand": "On hand", "incoming_confirmed": "Arriving",
                                "incoming_delayed": "Overdue", "avg_daily_forecast": "Forecast/day",
                                "days_cover": "Days of cover", "days_to_stockout": "Runs out in (days)",
                                "reason": "Why"})

elif page == "Agent activity":
    st.title("Agent activity")
    st.markdown('<p class="note">Every step the agents took in today\'s planning cycle, in order.</p>',
                unsafe_allow_html=True)
    agents = sorted({e.agent for e in R.log.events})
    pick = st.multiselect("Show agents", agents, default=agents)
    agent_log([e for e in R.log.events if e.agent in pick], AGENT_ICONS)

elif page == "Demand forecast":
    st.title("Demand forecast")
    c1, c2 = st.columns(2)
    wh = c1.selectbox("Warehouse", list(WAREHOUSES.index), format_func=lambda w: f"{w} · {WAREHOUSES.loc[w].city}")
    pid = c2.selectbox("Product", list(PRODUCTS.index), format_func=lambda p: PRODUCTS.loc[p].product_name)
    hist = S.sales[(S.sales.warehouse_id == wh) & (S.sales.product_id == pid)
                   & (S.sales.date >= S.today - pd.Timedelta(days=120))]
    fc = R.forecast[(R.forecast.warehouse_id == wh) & (R.forecast.product_id == pid)]
    stt = R.forecast_stats.set_index(["warehouse_id", "product_id"]).loc[(wh, pid)]
    band = 1.65 * stt.sigma
    fig = go.Figure()
    # festival windows (shaded) inside the forecast period
    cal = data["calendar"]
    fwin = cal[(cal.date >= hist.date.min()) & (cal.date <= fc.date.max()) & cal.festival_window.notna()]
    for name, g in fwin.groupby((fwin.festival_window != fwin.festival_window.shift()).cumsum()):
        label = g.festival_window.iloc[0] if len(g) >= 4 else ""   # skip labels on 1–3 day windows (avoid overlap)
        fig.add_vrect(x0=g.date.min(), x1=g.date.max() + pd.Timedelta(days=1), fillcolor=YELLOW, opacity=.13,
                      line_width=0, annotation_text=label, annotation_position="top left",
                      annotation_font_size=11)
    fig.add_trace(go.Scatter(x=hist.date, y=hist.units_sold, name="Actual sales", line=dict(color=INK, width=1.6)))
    so = hist[hist.stockout]
    fig.add_trace(go.Scatter(x=so.date, y=so.units_sold, mode="markers", name="Stock-out day",
                             marker=dict(color="#C0392B", size=7, symbol="x")))
    fig.add_trace(go.Scatter(x=list(fc.date) + list(fc.date[::-1]),
                             y=list(fc.forecast + band) + list((fc.forecast - band).clip(lower=0)[::-1]),
                             fill="toself", fillcolor="rgba(47,93,138,.15)", line=dict(width=0),
                             name="95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.forecast, name="AI forecast", line=dict(color=STEEL, width=2.4)))
    fig.update_layout(template="simple_white", height=420, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Barlow, Arial"), legend=dict(orientation="h", y=1.08),
                      yaxis_title="Units per day")
    st.plotly_chart(fig, width="stretch")
    note = f"Next 7 days: ~{fc.forecast.head(7).mean():.0f}/day · next 28 days: {fc.forecast.sum():,.0f} units."
    if abs(stt.level_factor - 1) >= 0.05:
        note += (f" Last week sales ran {stt.recent_ratio:.2f}× the model's expectation, so the agent adjusted "
                 f"its forecast by ×{stt.level_factor:.2f}.")
    st.markdown(f'<p class="note">{note}</p>', unsafe_allow_html=True)

    m = forecaster.metrics
    st.subheader("How good is the forecast?")
    st.markdown(f"Tested on the last {m['test_days']} days of real data the model never saw. "
                f"**AI model: {m['ai_accuracy']}% accurate** vs **{m['manual_accuracy']}%** for the old manual method "
                f"(28-day average). Accuracy = 1 − total error ÷ total sales.")
    bp = m["by_product"].sort_values("ai_accuracy_%")
    fig2 = go.Figure()
    fig2.add_bar(y=bp.product_name, x=bp["manual_accuracy_%"], name="Manual (28-day average)", orientation="h",
                 marker_color="#B8BEB4")
    fig2.add_bar(y=bp.product_name, x=bp["ai_accuracy_%"], name="AI model", orientation="h", marker_color=STEEL)
    fig2.update_layout(template="simple_white", barmode="group", height=480, margin=dict(l=10, r=10, t=10, b=10),
                       font=dict(family="Barlow, Arial"), xaxis=dict(range=[50, 100], title="Accuracy %"),
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, width="stretch")

elif page == "Purchase orders":
    st.title("Purchase orders")
    pend = [p for p in ss.pending if p["status"] == "Pending approval"]
    st.markdown(f'<p class="note">{len(pend)} orders drafted today. Orders of ₹50,000 or more always need a '
                f'manager\'s approval.</p>', unsafe_allow_html=True)
    routine = [p for p in pend if not p["needs_human"]]
    if routine and st.button(f"Approve all {len(routine)} routine orders"):
        for p in routine:
            approve_po(S, p)
        run_agents(); st.rerun()
    for p in pend:
        icon = "🔴" if p["risk"] == "CRITICAL" else "🟠" if p["risk"] == "HIGH" else "🟢"
        label = (f"{icon} {p['product_name']} → {p['warehouse_id']} · {p['qty']} units from {p['vendor_name']} · "
                 f"{inr(p['total_cost'])}" + (" · needs approval" if p["needs_human"] else ""))
        with st.expander(label, expanded=p["risk"] in ("CRITICAL",)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Priority", p["priority"])
            c2.metric("Runs out in", "Out of stock" if p["days_to_stockout"] <= 0 else f"{p['days_to_stockout']} days")
            c3.metric("Expected delivery", f"{p['expected_lead_days']} days")
            c4.metric("Unit price", f"₹{p['unit_price']:,.2f}")
            st.markdown(f"**Why this vendor:** {p.get('ai_explanation') or p['reason']}")
            st.caption("Vendors compared (total expected cost = purchase + expected lost sales + late-delivery risk)")
            st.dataframe(p["options"][["vendor_name", "qty", "unit_price", "expected_lead", "reliability",
                                       "purchase_cost", "stockout_cost", "late_risk_cost", "total_expected_cost"]],
                         hide_index=True, width="stretch",
                         column_config={"vendor_name": "Vendor", "qty": "Qty needed", "unit_price": "Unit price ₹",
                                        "expected_lead": "Expected days", "reliability": "Reliability",
                                        "purchase_cost": "Purchase ₹", "stockout_cost": "Lost-sales risk ₹",
                                        "late_risk_cost": "Late risk ₹", "total_expected_cost": "Total expected ₹"})
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Approve", key="ap" + p["po_id"], type="primary"):
                approve_po(S, p); run_agents(); st.rerun()
            if b2.button("Reject", key="rj" + p["po_id"]):
                S.rejected.append(p["po_id"]); run_agents(); st.rerun()
            if explainer.provider() and b3.button("Explain with AI", key="ex" + p["po_id"]):
                with st.spinner("Asking the AI…"):
                    p["ai_explanation"] = explainer.explain_vendor_choice(p)
                st.rerun()
            b4.download_button("PO as PDF", orch.po.to_pdf(p), file_name=f"{p['po_id']}.pdf",
                               mime="application/pdf", key="pdf" + p["po_id"])
    if not pend:
        st.success("Nothing waiting. New orders appear when the agents see a need.")

    st.subheader("Approved orders")
    if S.approved:
        ap = pd.DataFrame(S.approved)
        st.dataframe(ap[["po_id", "approved_on", "warehouse_id", "product_name", "vendor_name", "qty", "total_cost",
                         "expected_delivery", "priority"]], hide_index=True, width="stretch")
    else:
        st.caption("No orders approved yet in this session.")

elif page == "Transfers & alerts":
    st.title("Transfers & alerts")
    st.subheader("Stock transfers between warehouses")
    if not R.transfers:
        st.caption("No transfers needed right now.")
    for i, t in enumerate(R.transfers):
        alert("HIGH", f"Move {t['qty']} × {t['product_name']}: {t['from_warehouse']} → {t['to_warehouse']}",
              f"{t['reason']} Handling cost ≈ {inr(t['est_cost'])}. Sender keeps {t['donor_cover_after']} days of stock.")
        if st.button("Approve transfer", key=f"tr{i}"):
            approve_transfer(S, t); run_agents(); st.rerun()
    st.subheader("Overstock")
    if R.overstock:
        st.dataframe(pd.DataFrame(R.overstock)[["warehouse_id", "product_name", "days_cover", "excess_units",
                                                "capital_locked", "holding_cost_per_month", "suggestion"]],
                     hide_index=True, width="stretch")
    else:
        st.caption("No overstocked items.")
    st.subheader("Anomalies")
    for a in R.anomalies:
        alert("CRITICAL" if a["severity"] == "HIGH" else "MEDIUM", a["type"], f"{a['detail']}. Suggested: {a['action']}.",
              tag=a["severity"].title())

elif page == "Vendors":
    st.title("Vendors")
    st.markdown('<p class="note">Reliability is learned from every past delivery, weighting the last 60 days more. '
                'Compare it with the rating the vendor lists.</p>', unsafe_allow_html=True)
    vs = R.vendor_stats
    st.dataframe(vs[["vendor_name", "reliability_score", "listed_rating", "on_time_pct", "recent_on_time_pct",
                     "avg_delay_days", "fill_rate", "trend", "orders"]],
                 hide_index=True, width="stretch",
                 column_config={"vendor_name": "Vendor", "reliability_score": st.column_config.ProgressColumn(
                     "Reliability", min_value=0, max_value=100, format="%d"),
                     "listed_rating": "Listed rating", "on_time_pct": "On-time % (all)",
                     "recent_on_time_pct": "On-time % (60 days)", "avg_delay_days": "Avg delay (days)",
                     "fill_rate": "Fill rate %", "trend": "Trend", "orders": "Orders"})
    po = S.po_history[S.po_history.status == "Delivered"].copy()
    po["month"] = po.order_date.dt.to_period("M").dt.to_timestamp()
    mon = po.groupby(["month", "vendor_id"]).delay_days.apply(lambda x: 100 * (x <= 0).mean()).reset_index()
    fig = go.Figure()
    for v, g in mon.groupby("vendor_id"):
        fig.add_trace(go.Scatter(x=g.month, y=g.delay_days, name=VENDOR_NAMES[v], mode="lines+markers"))
    fig.update_layout(template="simple_white", height=380, yaxis_title="On-time deliveries %",
                      font=dict(family="Barlow, Arial"), margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.subheader("On-time delivery by month")
    st.caption("Monsoon months dip for everyone; a vendor that keeps falling afterwards is a real problem.")
    st.plotly_chart(fig, width="stretch")

elif page == "What-if lab":
    st.title("What-if lab")
    st.markdown('<p class="note">Change one thing and see how the agents re-plan. Nothing is ordered here.</p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vendor delays")
        vsel = st.multiselect("Vendors", list(VENDOR_NAMES), format_func=lambda v: VENDOR_NAMES[v])
        vdays = st.slider("Extra delay (days)", 0, 7, 3)
    with c2:
        st.subheader("Demand change")
        psel = st.selectbox("Product", ["ALL"] + list(PRODUCTS.index),
                            format_func=lambda p: "All products" if p == "ALL" else PRODUCTS.loc[p].product_name)
        pct = st.slider("Change in demand (%)", -50, 200, 0, step=10)
    wi = {}
    if vsel and vdays:
        wi["vendor_delay"] = {v: vdays for v in vsel}
    if pct:
        wi["demand_mult"] = {psel: 1 + pct / 100}
    b1, b2 = st.columns(2)
    run = b1.button("Run scenario", type="primary", disabled=not wi)
    if b2.button("Apply to live plan" if not ss.whatif_active else "Remove from live plan"):
        ss.whatif_active = not ss.whatif_active
        ss.whatif = wi if ss.whatif_active else {}
        run_agents(); st.rerun()
    if run:
        base = orch.run(S)
        scen = orch.run(S, wi)
        kb, ks = base.kpis, scen.kpis
        kpi_strip([
            (f"{kb['critical'] + kb['high']} → {ks['critical'] + ks['high']}", "items at high risk",
             ks["critical"] + ks["high"] > kb["critical"] + kb["high"]),
            (f"{kb['po_count']} → {ks['po_count']}", "purchase orders", False),
            (f"{inr(kb['po_value'])} → {inr(ks['po_value'])}", "order value", False),
            (f"{kb['transfers']} → {ks['transfers']}", "transfers", False),
        ])
        m = base.risk[["warehouse_id", "product_name", "risk", "days_to_stockout"]].merge(
            scen.risk[["warehouse_id", "product_name", "risk", "days_to_stockout"]],
            on=["warehouse_id", "product_name"], suffixes=(" (now)", " (scenario)"))
        changed = m[m["risk (now)"] != m["risk (scenario)"]]
        st.subheader("Risk changes")
        st.dataframe(changed, hide_index=True, width="stretch") if len(changed) else st.caption("No risk levels changed.")
        bv = {(p["warehouse_id"], p["product_id"]): p for p in base.purchase_orders}
        sw = [dict(item=f"{p['product_name']} @ {p['warehouse_id']}", before=bv[k]["vendor_name"] if k in bv else "no order",
                   after=p["vendor_name"], qty_before=bv[k]["qty"] if k in bv else 0, qty_after=p["qty"])
              for p in scen.purchase_orders for k in [(p["warehouse_id"], p["product_id"])]
              if k not in bv or bv[k]["vendor_id"] != p["vendor_id"] or bv[k]["qty"] != p["qty"]]
        st.subheader("Orders the agents changed")
        st.dataframe(pd.DataFrame(sw), hide_index=True, width="stretch") if sw else st.caption("Same orders as now.")
        with st.expander("Agent log for this scenario"):
            agent_log(scen.log.events, AGENT_ICONS)

elif page == "Stream analytics":
    st.title("Stream analytics · Real-time supply chain events")
    st.markdown(
        '<p class="note">A streaming-data mini-project built on the existing supply-chain dataset. '
        'Generate sale and customer-feedback events, inspect live-style analytics, and export the event stream '
        'for optional Kafka and PySpark processing.</p>', unsafe_allow_html=True)

    if "bdsma_events" not in ss:
        ss.bdsma_events = pd.DataFrame()

    top_left, top_right = st.columns([1, 2])
    with top_left:
        batch_size = st.select_slider("Events to generate", options=[25, 50, 100, 250, 500, 1000], value=100)
        if st.button("Generate next event batch", type="primary", width="stretch"):
            new_events = build_event_batch(S.sales, data["products"], n=int(batch_size))
            if not new_events.empty:
                ss.bdsma_events = pd.concat([ss.bdsma_events, new_events], ignore_index=True)
                ss.bdsma_events = ss.bdsma_events.drop_duplicates("event_id").sort_values("event_time").reset_index(drop=True)
            st.rerun()
        if st.button("Clear stream", width="stretch"):
            ss.bdsma_events = pd.DataFrame()
            st.rerun()
    with top_right:
        st.info("Demo mode uses your existing historical sales records and generates sample customer-feedback events. "
                "It runs locally without requiring a Kafka or Hadoop installation.")

    events = ss.bdsma_events.copy()
    summary = summarize_events(events)
    kpi_strip([
        (f"{summary['total_events']:,}", "events processed", False),
        (f"{summary['sales_events']:,}", "sales events", False),
        (f"{summary['feedback_events']:,}", "customer feedback events", False),
        (f"{summary['units']:,}", "units sold in stream", False),
        (inr(summary["revenue"]), "stream revenue", False),
        (f"{summary['stockout_rate']:.1f}%", "stockout event rate", summary["stockout_rate"] >= 5),
    ])

    if events.empty:
        st.warning("No events yet. Select a batch size and click **Generate next event batch** to start the stream.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Sales by product")
            if not summary["top_products"].empty:
                fig = go.Figure(go.Bar(
                    x=summary["top_products"].units_sold,
                    y=summary["top_products"].product_name,
                    orientation="h", marker_color=STEEL,
                    hovertemplate="%{y}<br>Units: %{x:,}<extra></extra>"))
                fig.update_layout(template="simple_white", height=360, yaxis=dict(autorange="reversed"),
                                  margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Units sold")
                st.plotly_chart(fig, width="stretch")
        with right:
            st.subheader("Customer feedback sentiment")
            if not summary["sentiment"].empty:
                sent_df = summary["sentiment"].rename_axis("sentiment").reset_index(name="count")
                fig = go.Figure(go.Pie(labels=sent_df.sentiment, values=sent_df["count"], hole=.55))
                fig.update_layout(template="simple_white", height=360, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("No customer feedback events in this batch yet.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Warehouse-wise sales")
            if not summary["warehouse_sales"].empty:
                st.dataframe(summary["warehouse_sales"].style.format({"revenue": "₹{:,.0f}"}),
                             hide_index=True, width="stretch")
        with c2:
            st.subheader("Customer complaint topics")
            if not summary["topics"].empty:
                topic_df = summary["topics"].rename_axis("topic").reset_index(name="events")
                st.dataframe(topic_df, hide_index=True, width="stretch")
            else:
                st.caption("No feedback topics available yet.")

        st.subheader("Supply-chain signals for AAI agents")
        sales_events = events[events.event_type == "sale"]
        feedback_events = events[events.event_type == "customer_feedback"]
        signals = []
        if not sales_events.empty:
            stockout_products = (sales_events[sales_events.stockout]
                                 .groupby(["product_id", "product_name", "warehouse_id"]).size()
                                 .rename("stockout_events").reset_index())
            if not stockout_products.empty:
                stockout_products["signal"] = "Stockout observed in stream"
                signals.append(stockout_products)
        if not feedback_events.empty:
            negative = (feedback_events[feedback_events.sentiment == "negative"]
                        .groupby(["product_id", "product_name", "warehouse_id"]).size()
                        .rename("negative_feedback").reset_index())
            if not negative.empty:
                negative["signal"] = "Negative customer feedback"
                signals.append(negative)
        if signals:
            signal_df = pd.concat(signals, ignore_index=True, sort=False).fillna(0)
            aai_cols = ["product_id", "warehouse_id", "risk", "days_to_stockout", "on_hand"]
            signal_df = signal_df.merge(R.risk[aai_cols], on=["product_id", "warehouse_id"], how="left")
            st.dataframe(signal_df, hide_index=True, width="stretch")
            st.caption("The stream signals are cross-referenced with the existing AAI Risk Agent's stockout assessment.")
        else:
            st.success("No stockout or negative-feedback signals detected in the current event batch.")

        if not summary["anomalies"].empty:
            st.subheader("Unusually high sales events")
            st.dataframe(summary["anomalies"][
                ["event_time", "product_name", "warehouse_id", "units_sold", "anomaly_reason"]
            ], hide_index=True, width="stretch")
        else:
            st.caption("No unusually high sales events detected using the current batch mean + 2σ rule.")

        st.subheader("Recent stream events")
        st.dataframe(events.tail(30).sort_values("event_time", ascending=False),
                     hide_index=True, width="stretch")
        csv_bytes = events.to_csv(index=False).encode("utf-8")
        jsonl_bytes = events.to_json(orient="records", lines=True, force_ascii=False).encode("utf-8")
        d1, d2 = st.columns(2)
        d1.download_button("Download events CSV", csv_bytes, "bdsma_stream_events.csv", "text/csv", width="stretch")
        d2.download_button("Download events JSONL (Kafka / Spark)", jsonl_bytes,
                            "bdsma_stream_events.jsonl", "application/x-ndjson", width="stretch")

        with st.expander("Optional big-data execution: PySpark and Kafka"):
            st.markdown("""
            **PySpark local analytics**

            1. Save the downloaded JSONL file inside the project's `data` folder as `bdsma_stream_events.jsonl`.
            2. Install Java 17 and the optional packages: `pip install -r requirements-bdsma.txt`.
            3. Run from the project root:

            ```bash
            spark-submit bdsma/pyspark_job.py data/bdsma_stream_events.jsonl
            ```

            **Kafka ingestion (optional)**

            Start a Kafka broker, then run the producer and consumer scripts in separate terminals:

            ```bash
            python bdsma/kafka_consumer.py --max-events 0
            python bdsma/kafka_producer.py data/bdsma_stream_events.jsonl
            ```

            The consumer writes captured Kafka messages to `data/kafka_captured_events.jsonl`, which can also
            be analysed with the PySpark job. The dashboard's local demo does not require these services.
            """)

    # ---- RAG: Supply Chain Knowledge Assistant (lives inside this page, no separate sidebar item)
    from rag.ui import render as render_knowledge_assistant
    render_knowledge_assistant(data, S, R)

elif page == "Simulation":
    st.title("Simulation")
    st.markdown('<p class="note">Use “Next day” in the sidebar to let time pass: approved orders travel, vendors '
                'deliver (or slip, based on their real reliability), customers buy, and the agents re-plan every '
                'morning. Turn on Autopilot to watch the full loop run by itself.</p>', unsafe_allow_html=True)
    if not S.day_log:
        st.info("No days simulated yet. Press “Next day” or “Next 7 days” in the sidebar.")
    else:
        dl = pd.DataFrame(S.day_log)
        hist = S.sales[(S.sales.date >= "2026-07-01") & (S.sales.date < "2026-10-06")]
        sim = S.sales[S.sales.date >= "2026-10-06"]
        kpi_strip([(len(dl), "days simulated", False), (f"{dl.units_sold.sum():,}", "units sold", False),
                   (inr(dl.revenue.sum()), "revenue", False), (f"{dl.lost_sales_units.sum():,}", "units of lost sales", False),
                   (f"{100 * sim.stockout.mean():.1f}%", "stock-out rate with agents", False),
                   (f"{100 * hist.stockout.mean():.1f}%", "stock-out rate before (Jul–Sep)", False)])
        fig = go.Figure()
        fig.add_bar(x=dl.date, y=dl.lost_sales_units, name="Lost sales (units)", marker_color="#C0392B")
        fig.add_trace(go.Scatter(x=dl.date, y=dl.deliveries_arrived, name="Deliveries arrived", yaxis="y2",
                                 line=dict(color=STEEL)))
        fig.update_layout(template="simple_white", height=340, font=dict(family="Barlow, Arial"),
                          yaxis2=dict(overlaying="y", side="right", title="Deliveries"),
                          margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(dl, hide_index=True, width="stretch")

elif page == "Ask the agent":
    st.title("Ask the agent")
    if not explainer.provider():
        st.info("Chat needs an AI key. Add GROQ_API_KEY or GEMINI_API_KEY to a .env file and restart. "
                "Until then, try the quick answers below.")
    ctx = dict(
        date=str(S.today.date()), kpis=R.kpis,
        risks=R.risk[R.risk.risk != "LOW"][["warehouse_id", "product_name", "on_hand", "avg_daily_forecast",
                                            "days_to_stockout", "risk", "reason"]].to_dict("records"),
        pending_orders=[{k: p[k] for k in ("po_id", "warehouse_id", "product_name", "vendor_name", "qty",
                                           "total_cost", "priority", "reason")} for p in ss.pending],
        transfers=[{k: t[k] for k in ("product_name", "from_warehouse", "to_warehouse", "qty", "reason")} for t in R.transfers],
        anomalies=[a["detail"] for a in R.anomalies],
        vendors=R.vendor_stats[["vendor_name", "reliability_score", "on_time_pct", "recent_on_time_pct", "trend"]].to_dict("records"),
        festivals=R.festival_watch.to_dict("records") if not R.festival_watch.empty else [],
    )
    quick = ["Which items will run out this week?", "Which vendor should I worry about?",
             "What should I prepare for Diwali?", "Summarise today's orders"]
    cols = st.columns(len(quick))
    q = None
    for c, text in zip(cols, quick):
        if c.button(text, width="stretch"):
            q = text
    typed = st.chat_input("Ask about stock, orders, vendors or festivals…")
    q = typed or q
    for m in ss.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    if q:
        st.chat_message("user").markdown(q)
        ans = explainer.chat(q, ctx) if explainer.provider() else None
        if not ans:   # offline quick answers from the agents' own data
            ql = q.lower()
            if "run out" in ql or "stock" in ql:
                rows = R.risk[R.risk.days_to_stockout <= 7].head(10)
                ans = "\n".join(f"- {r.product_name} @ {r.warehouse_id}: runs out in {r.days_to_stockout} days ({r.risk})"
                                for r in rows.itertuples()) or "Nothing runs out within 7 days."
            elif "vendor" in ql:
                ans = "\n".join(f"- {v.vendor_name}: reliability {v.reliability_score:.0f}, {v.recent_on_time_pct:.0f}% on-time "
                                f"lately ({v.trend})" for v in R.vendor_stats.sort_values("reliability_score").itertuples())
            elif "diwali" in ql or "festival" in ql:
                fw = R.festival_watch
                ans = "\n".join(f"- {x.festival} ({x.days_away} days): {x.product_name} ×{x.last_year_uplift} last year"
                                for x in fw.itertuples()) if not fw.empty else "No festivals in the next 45 days."
            else:
                ans = R.briefing
        st.chat_message("assistant").markdown(ans)
        ss.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
