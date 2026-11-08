"""
generate_data.py
Realistic synthetic dataset for PS-08: Supply Chain & Inventory Management Agent.

Simulates "Sahyadri Retail Distribution" — a grocery distributor with 3 warehouses in
Maharashtra (Bhiwandi/Mumbai, Pune, Nashik), 14 products and 6 vendors, from
2024-10-01 to 2026-10-05.

The demand model includes: weekday patterns, monthly salary-cycle effect, seasonal
(summer / monsoon / winter) patterns, real Indian festival effects (Diwali, Ganesh
Chaturthi, Navratri, Holi, Eid, Christmas...), Shravan dip for eggs, yearly growth,
price hikes, promotions with post-promo dip, and random noise.

The supply side is a day-by-day inventory simulation with FIFO batches, expiry/spoilage
for perishables, a (human-style) reorder policy, vendor lead-time delays, short deliveries,
monsoon disruptions and stock-outs. So every file is consistent with every other file.

Run:  python generate_data.py      (output -> ./data)
"""
import os
import math
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
START = pd.Timestamp("2024-10-01")
END = pd.Timestamp("2026-10-05")          # "today" for the project = 2026-10-06
CAL_END = pd.Timestamp("2026-12-31")      # calendar extends into the future for forecasting
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
dates = pd.date_range(START, END, freq="D")
N = len(dates)
didx = {d: i for i, d in enumerate(dates)}

# ----------------------------------------------------------------------------- master data
warehouses = pd.DataFrame([
    ["WH-MUM", "Bhiwandi Distribution Centre", "Mumbai (Bhiwandi)", "Maharashtra", 19.2967, 73.0631, 60000, 1.00],
    ["WH-PUN", "Pune Chakan Warehouse", "Pune", "Maharashtra", 18.7606, 73.8636, 40000, 0.70],
    ["WH-NSK", "Nashik Satpur Warehouse", "Nashik", "Maharashtra", 19.9975, 73.7898, 25000, 0.42],
], columns=["warehouse_id", "warehouse_name", "city", "state", "latitude", "longitude",
            "capacity_units", "_scale"])

# id, name, category, unit, cost, price, shelf_life, base_demand(MUM/day), order_multiple, perishable, group
P = [
    ["P001", "Toned Milk 1L", "Dairy", "pouch", 50, 56, 3, 230, 12, 1, "fresh"],
    ["P002", "Brown Bread 400g", "Bakery", "packet", 37, 45, 5, 85, 10, 1, "fresh"],
    ["P003", "Eggs (Tray of 30)", "Dairy", "tray", 165, 195, 21, 34, 5, 1, "fresh"],
    ["P004", "Basmati Rice 5kg", "Staples", "bag", 520, 625, 365, 22, 4, 0, "staple"],
    ["P005", "Toor Dal 1kg", "Staples", "packet", 142, 172, 270, 42, 10, 0, "staple"],
    ["P006", "Wheat Atta 10kg", "Staples", "bag", 385, 460, 120, 26, 2, 0, "staple"],
    ["P007", "Sugar 1kg", "Staples", "packet", 41, 48, 365, 62, 25, 0, "staple"],
    ["P008", "Sunflower Oil 1L", "Edible Oil", "pouch", 132, 160, 270, 46, 12, 0, "staple"],
    ["P009", "Tea Powder 500g", "Beverages", "packet", 228, 275, 365, 28, 12, 0, "staple"],
    ["P010", "Glucose Biscuits 800g", "Snacks", "packet", 44, 55, 180, 72, 24, 0, "impulse"],
    ["P011", "Pure Cow Ghee 1L", "Dairy", "jar", 555, 650, 270, 12, 6, 0, "staple"],
    ["P012", "Vanilla Ice Cream 1L", "Frozen", "tub", 158, 210, 180, 18, 6, 0, "impulse"],
    ["P013", "Cola 2L", "Beverages", "bottle", 84, 100, 180, 40, 6, 0, "impulse"],
    ["P014", "Detergent Powder 1kg", "Home Care", "packet", 94, 120, 730, 30, 12, 0, "staple"],
]
products = pd.DataFrame(P, columns=["product_id", "product_name", "category", "unit", "_cost0", "_price0",
                                    "shelf_life_days", "_base", "order_multiple", "perishable", "_group"])

vendors = pd.DataFrame([
    ["V01", "Sahyadri Dairy Co-operative", "Pune", "Dairy & frozen", 4.6, 30],
    ["V02", "Mumbai Fresh Distributors", "Mumbai", "Fresh & dairy", 4.3, 15],
    ["V03", "Deccan Agro Traders", "Nashik", "Grains & pulses", 4.4, 45],
    ["V04", "Konkan Wholesale Mart", "Bhiwandi", "General grocery wholesale", 3.9, 30],
    ["V05", "Western FMCG Supply Co.", "Mumbai", "Packaged FMCG", 4.2, 30],
    ["V06", "CoolChain Beverages", "Thane", "Beverages & frozen", 4.0, 21],
], columns=["vendor_id", "vendor_name", "city", "specialization", "listed_rating", "payment_terms_days"])

# hidden "true" vendor behaviour (NOT exported) -> the agent must learn it from purchase_orders.csv
VB = {  # on-time probability, max delay days, fill-rate range
    "V01": dict(ontime=0.93, maxdelay=1, fill=(0.97, 1.0)),
    "V02": dict(ontime=0.80, maxdelay=3, fill=(0.88, 1.0)),
    "V03": dict(ontime=0.90, maxdelay=3, fill=(0.95, 1.0)),
    "V04": dict(ontime=0.72, maxdelay=4, fill=(0.85, 1.0)),
    "V05": dict(ontime=0.88, maxdelay=2, fill=(0.94, 1.0)),
    "V06": dict(ontime=0.82, maxdelay=3, fill=(0.90, 1.0)),
}
# vendor, product, price factor vs product cost, promised lead days, MOQ, bulk qty, bulk disc %
VP = [
    ["V01", "P001", 1.00, 1, 48, 480, 2.0], ["V02", "P001", 0.97, 1, 48, 480, 1.5],
    ["V01", "P003", 1.02, 2, 10, 100, 2.0], ["V02", "P003", 0.98, 1, 10, 100, 1.0],
    ["V02", "P002", 0.98, 1, 20, 200, 1.0], ["V05", "P002", 1.02, 2, 20, 200, 2.0],
    ["V03", "P004", 0.96, 4, 40, 200, 3.0], ["V04", "P004", 1.00, 2, 20, 100, 2.0],
    ["V03", "P005", 0.95, 4, 50, 300, 3.0], ["V04", "P005", 1.00, 2, 30, 150, 2.0],
    ["V03", "P006", 0.97, 4, 20, 120, 2.5], ["V04", "P006", 1.00, 2, 10, 60, 1.5],
    ["V03", "P007", 0.96, 4, 100, 500, 2.0], ["V04", "P007", 1.00, 2, 50, 250, 1.5],
    ["V04", "P008", 1.00, 2, 48, 240, 2.0], ["V05", "P008", 0.98, 3, 48, 240, 2.5],
    ["V04", "P009", 1.01, 2, 24, 120, 1.5], ["V05", "P009", 0.97, 3, 24, 144, 3.0],
    ["V04", "P010", 1.00, 2, 48, 240, 2.0], ["V05", "P010", 0.97, 3, 48, 288, 3.0],
    ["V01", "P011", 0.97, 2, 12, 60, 3.0], ["V05", "P011", 1.00, 3, 12, 60, 2.0],
    ["V01", "P012", 1.00, 2, 12, 60, 2.0], ["V06", "P012", 0.96, 2, 18, 90, 3.0],
    ["V05", "P013", 1.01, 3, 18, 108, 2.0], ["V06", "P013", 0.95, 2, 18, 108, 3.0],
    ["V04", "P014", 1.00, 2, 24, 120, 2.0], ["V05", "P014", 0.96, 3, 24, 144, 3.0],
]
vendor_products = pd.DataFrame(VP, columns=["vendor_id", "product_id", "_pf", "promised_lead_time_days",
                                            "min_order_qty", "bulk_discount_min_qty", "bulk_discount_pct"])

# ----------------------------------------------------------------------------- calendar & effects
FEST = [  # name, date, (window start offset, end offset), product multipliers
    ("Navratri", "2024-10-03", (0, 8), {"P003": .72, "P001": 1.08, "P011": 1.2}),
    ("Dussehra", "2024-10-12", (-2, 0), {"P011": 1.3, "P007": 1.2}),
    ("Diwali", "2024-10-31", (-14, 1), "DIWALI"),
    ("Christmas & New Year", "2024-12-24", (0, 7), "XMAS"),
    ("Makar Sankranti", "2025-01-14", (-2, 0), {"P007": 1.35, "P011": 1.15}),
    ("Holi", "2025-03-14", (-2, 0), {"P013": 1.6, "P001": 1.3, "P007": 1.2}),
    ("Gudi Padwa", "2025-03-30", (-2, 0), {"P007": 1.25, "P011": 1.25, "P001": 1.1}),
    ("Eid al-Fitr", "2025-03-31", (-3, 0), {"P001": 1.4, "P007": 1.3, "P011": 1.4, "P003": 1.2}),
    ("Eid al-Adha", "2025-06-07", (-2, 0), {"P004": 1.4, "P008": 1.2, "P011": 1.2}),
    ("Raksha Bandhan", "2025-08-09", (-2, 0), {"P007": 1.2, "P011": 1.25}),
    ("Ganesh Chaturthi", "2025-08-27", (-3, 10), "GANESH"),
    ("Navratri", "2025-09-22", (0, 8), {"P003": .72, "P001": 1.08, "P011": 1.2}),
    ("Dussehra", "2025-10-02", (-2, 0), {"P011": 1.3, "P007": 1.2}),
    ("Diwali", "2025-10-20", (-14, 1), "DIWALI"),
    ("Christmas & New Year", "2025-12-24", (0, 7), "XMAS"),
    ("Makar Sankranti", "2026-01-14", (-2, 0), {"P007": 1.35, "P011": 1.15}),
    ("Holi", "2026-03-04", (-2, 0), {"P013": 1.6, "P001": 1.3, "P007": 1.2}),
    ("Gudi Padwa", "2026-03-19", (-2, 0), {"P007": 1.25, "P011": 1.25, "P001": 1.1}),
    ("Eid al-Fitr", "2026-03-20", (-3, 0), {"P001": 1.4, "P007": 1.3, "P011": 1.4, "P003": 1.2}),
    ("Eid al-Adha", "2026-05-27", (-2, 0), {"P004": 1.4, "P008": 1.2, "P011": 1.2}),
    ("Raksha Bandhan", "2026-08-28", (-2, 0), {"P007": 1.2, "P011": 1.25}),
    ("Ganesh Chaturthi", "2026-09-14", (-3, 10), "GANESH"),
    # upcoming (future) — present in calendar.csv so the forecasting agent can anticipate them
    ("Navratri", "2026-10-11", (0, 8), {"P003": .72, "P001": 1.08, "P011": 1.2}),
    ("Dussehra", "2026-10-20", (-2, 0), {"P011": 1.3, "P007": 1.2}),
    ("Diwali", "2026-11-08", (-14, 1), "DIWALI"),
    ("Christmas & New Year", "2026-12-24", (0, 7), "XMAS"),
]
PROFILES = {
    "DIWALI": {"P011": 2.6, "P007": 1.9, "P008": 1.7, "P006": 1.35, "P013": 1.4, "P010": 1.35,
               "P014": 1.6, "P001": 1.15, "P005": 1.1, "P004": 1.15, "P009": 1.1, "P012": 1.1},
    "GANESH": {"P011": 1.6, "P007": 1.5, "P001": 1.2, "P013": 1.2, "P010": 1.15, "P003": .75},
    "XMAS": {"P013": 1.5, "P012": 1.4, "P003": 1.3, "P002": 1.25, "P010": 1.15},
}
SHRAVAN = [("2025-07-25", "2025-08-23"), ("2026-08-12", "2026-09-10")]
MONSOON = [("2025-06-10", "2025-09-30"), ("2026-06-08", "2026-09-30")]

SEASON = {  # monthly multipliers (Jan..Dec)
    "P001": [1.0, 1.0, 1.05, 1.12, 1.12, .98, .96, .96, .98, 1.03, 1.03, 1.0],
    "P003": [1.15, 1.1, 1.0, .92, .9, .95, 1.0, 1.0, 1.0, 1.02, 1.08, 1.15],
    "P009": [1.2, 1.1, .95, .85, .85, 1.1, 1.18, 1.18, 1.12, 1.0, 1.1, 1.2],
    "P012": [.6, .8, 1.4, 1.9, 2.0, 1.1, .7, .7, .8, 1.0, .7, .75],
    "P013": [.9, 1.0, 1.3, 1.6, 1.7, 1.1, .8, .8, .9, 1.0, 1.0, 1.15],
    "P002": [1.0, 1.0, 1.0, .97, .97, 1.03, 1.05, 1.05, 1.02, 1.0, 1.0, 1.02],
}
WEEKDAY = {  # Mon..Sun
    "fresh": [1.0, .98, .98, 1.0, 1.02, 1.05, 1.12],
    "staple": [.9, .9, .95, .95, 1.05, 1.2, 1.25],
    "impulse": [.85, .85, .85, .9, 1.1, 1.3, 1.4],
}
for k, v in WEEKDAY.items():
    m = np.mean(v); WEEKDAY[k] = [x / m for x in v]
GROWTH = {"P012": .12, "P013": .10, "P009": .0, "P002": .05}  # default 8%/yr
ELASTIC = {"fresh": 1.6, "staple": 2.2, "impulse": 3.5}

calendar = pd.DataFrame({"date": pd.date_range(START, CAL_END, freq="D")})
calendar["day_of_week"] = calendar.date.dt.day_name()
calendar["is_weekend"] = calendar.date.dt.dayofweek >= 5
calendar["month"] = calendar.date.dt.month
calendar["season"] = calendar.month.map(lambda m: "Summer" if m in (3, 4, 5) else
                                        "Monsoon" if m in (6, 7, 8, 9) else
                                        "Post-monsoon" if m in (10, 11) else "Winter")
calendar["festival"] = ""
calendar["festival_window"] = ""
calendar["is_monsoon_disruption_period"] = False
calendar["is_shravan"] = False
for name, ds, (a, b), _ in FEST:
    d = pd.Timestamp(ds)
    calendar.loc[calendar.date == d, "festival"] = name
    m = (calendar.date >= d + pd.Timedelta(days=a)) & (calendar.date <= d + pd.Timedelta(days=b))
    calendar.loc[m & (calendar.festival_window == ""), "festival_window"] = name
for s, e in MONSOON:
    calendar.loc[(calendar.date >= s) & (calendar.date <= e), "is_monsoon_disruption_period"] = True
for s, e in SHRAVAN:
    calendar.loc[(calendar.date >= s) & (calendar.date <= e), "is_shravan"] = True
HOL = {"01-26": "Republic Day", "08-15": "Independence Day", "05-01": "Maharashtra Day"}
for d in ["2025-01-26", "2025-08-15", "2026-01-26", "2026-08-15", "2025-05-01", "2026-05-01"]:
    m = (calendar.date == d) & (calendar.festival == "")
    calendar.loc[m, "festival"] = HOL[d[5:]]
calendar["is_salary_week"] = calendar.date.dt.day <= 7


def festival_mult(pid):
    m = np.ones(N)
    for name, ds, (a, b), prof in FEST:
        prof = PROFILES[prof] if isinstance(prof, str) else prof
        if pid not in prof:
            continue
        peak = prof[pid]; d = pd.Timestamp(ds)
        for off in range(a, b + 1):
            day = d + pd.Timedelta(days=off)
            if day in didx:
                # long windows ramp up toward the festival day, short ones are flat
                if (b - a) <= 4:
                    w = 1.0
                elif off <= 0 and a < 0:
                    w = 0.45 + 0.55 * (off - a) / (-a)
                else:
                    w = 0.85
                m[didx[day]] *= 1 + (peak - 1) * min(1.0, w)
        if name == "Diwali":  # post-Diwali slump (people travel / pantry already full)
            for off in range(2, 8):
                day = d + pd.Timedelta(days=off)
                if day in didx:
                    m[didx[day]] *= 0.85
    return m


# price hikes (both cost and selling price)
def price_factor(d, pid):
    f = 1.0
    if d >= pd.Timestamp("2025-04-01"): f *= 1.04
    if d >= pd.Timestamp("2026-04-01"): f *= 1.035
    if pid in ("P001",) and d >= pd.Timestamp("2025-05-01"): f *= 1.035   # milk price revision
    if pid in ("P008",) and d >= pd.Timestamp("2025-09-01"): f *= 1.06     # edible oil duty change
    return f


# ----------------------------------------------------------------------------- promotions
promos = []
pid_n = 1
for _, p in products.iterrows():
    if p.product_id == "P001":
        continue
    n = rng.integers(3, 6)
    starts = sorted(rng.choice(np.arange(20, N - 30), size=n, replace=False))
    for s in starts:
        length = int(rng.integers(5, 11))
        disc = float(rng.choice([0.08, 0.10, 0.12, 0.15, 0.20]))
        whs = "ALL" if rng.random() < 0.5 else str(rng.choice(warehouses.warehouse_id))
        promos.append([f"PR{pid_n:03d}", p.product_id, whs, dates[s].date(),
                       (dates[s] + pd.Timedelta(days=length - 1)).date(), disc,
                       rng.choice(["Weekend Bonanza", "Festive Offer", "Bulk Saver", "Brand Push", "Clearance"])])
        pid_n += 1
promotions = pd.DataFrame(promos, columns=["promo_id", "product_id", "warehouse_id", "start_date", "end_date",
                                           "discount_pct", "promo_name"])

# ----------------------------------------------------------------------------- demand
dow = np.array([d.dayofweek for d in dates]); mon = np.array([d.month for d in dates])
dom = np.array([d.day for d in dates]); t_years = np.arange(N) / 365.0
shravan = np.zeros(N, bool)
for s, e in SHRAVAN:
    shravan |= (dates >= s) & (dates <= e)
monsoon = np.zeros(N, bool)
for s, e in MONSOON:
    monsoon |= (dates >= s) & (dates <= e)

demand = {}   # (wh, pid) -> array of true demand
promo_flag = {}; disc_arr = {}
for _, p in products.iterrows():
    pid = p.product_id; g = p._group
    base = (p._base * np.array(SEASON.get(pid, [1] * 12))[mon - 1]
            * np.array(WEEKDAY[g])[dow]
            * (1 + GROWTH.get(pid, .08)) ** t_years
            * festival_mult(pid))
    if g == "staple":
        base *= np.where(dom <= 7, 1.12, np.where(dom >= 25, .93, 1.0))
    if pid == "P003":
        base *= np.where(shravan, .7, 1.0)
    for _, w in warehouses.iterrows():
        mu = base * w._scale
        # each warehouse has its own small local taste differences
        mu = mu * rng.uniform(0.92, 1.08)
        pf = np.zeros(N, bool); dsc = np.zeros(N)
        for _, pr in promotions[promotions.product_id == pid].iterrows():
            if pr.warehouse_id not in ("ALL", w.warehouse_id):
                continue
            m = (dates >= pd.Timestamp(pr.start_date)) & (dates <= pd.Timestamp(pr.end_date))
            pf |= m; dsc[m] = pr.discount_pct
            mu = np.where(m, mu * (1 + ELASTIC[g] * pr.discount_pct * rng.uniform(.8, 1.2)), mu)
            if not p.perishable:  # post-promo dip (customers stocked up)
                e = np.where(m)[0].max()
                post = np.arange(e + 1, min(N, e + 8))
                mu[post] *= 0.86
        # scenario: unexplained detergent spike in Pune over the last week (anomaly for the agent)
        if pid == "P014" and w.warehouse_id == "WH-PUN":
            mu[dates >= "2026-09-29"] *= 3.1
        k = 35.0  # gamma-poisson overdispersion
        lam = rng.gamma(k, mu / k)
        demand[(w.warehouse_id, pid)] = rng.poisson(lam)
        promo_flag[(w.warehouse_id, pid)] = pf; disc_arr[(w.warehouse_id, pid)] = dsc

# ----------------------------------------------------------------------------- supply simulation
vp_by_prod = {pid: g for pid, g in vendor_products.groupby("product_id")}
preferred = {}
for pid, g in vp_by_prod.items():  # the planner's "habit" vendor = usually cheapest
    preferred[pid] = g.sort_values("_pf").vendor_id.iloc[0]
COVER = {"fresh": 1, "staple": 12, "impulse": 9}

sales_rows, inv_rows, po_rows, adj_rows = [], [], [], []
po_counter = 1


def vendor_delay(vid, day_i):
    b = VB[vid]
    ontime = b["ontime"]
    if monsoon[day_i]: ontime -= 0.12
    if vid == "V02" and dates[day_i] >= pd.Timestamp("2026-08-01"): ontime -= 0.25   # V02 getting worse lately
    if rng.random() < ontime:
        return 0
    return int(rng.integers(1, b["maxdelay"] + 1 + (1 if monsoon[day_i] else 0)))


for _, w in warehouses.iterrows():
    wh = w.warehouse_id
    for _, p in products.iterrows():
        pid = p.product_id; g = p._group; shelf = int(p.shelf_life_days)
        dem = demand[(wh, pid)]
        cands = vp_by_prod[pid]
        lead_ref = int(cands.promised_lead_time_days.max()) + 1
        avg0 = dem[:14].mean()
        lots = [[int(avg0 * (lead_ref + COVER[g] / 2)) + 5, shelf]]   # [qty, expiry_day_index]
        pipeline = []   # [arrival_i, qty, po_index]
        hist = []
        for i, d in enumerate(dates):
            opening = sum(q for q, _ in lots)
            received = 0
            for pp in [x for x in pipeline if x[0] == i]:
                lots.append([pp[1], i + shelf]); received += pp[1]
                po_rows[pp[2]]["status"] = "Delivered"
                pipeline.remove(pp)
            spoiled = sum(q for q, e in lots if e <= i)
            lots = [[q, e] for q, e in lots if e > i]
            # forced events --------------------------------------------------------
            adj = 0
            if wh == "WH-NSK" and pid == "P011" and d == pd.Timestamp("2026-09-25"):
                adj = -min(18, sum(q for q, _ in lots))
                adj_rows.append([d.date(), wh, pid, adj, "Cycle count mismatch — suspected pilferage"])
            elif rng.random() < 0.0025 and not p.perishable:
                avail = sum(q for q, _ in lots)
                if avail > 10:
                    adj = -int(rng.integers(1, max(2, avail // 40)))
                    adj_rows.append([d.date(), wh, pid, adj, rng.choice(
                        ["Damaged in handling", "Cycle count mismatch", "Packaging leak", "Rodent damage"])])
            if adj < 0:
                rem = -adj
                for lot in lots:
                    take = min(lot[0], rem); lot[0] -= take; rem -= take
                lots = [l for l in lots if l[0] > 0]
            # sales (FIFO) ----------------------------------------------------------
            avail = sum(q for q, _ in lots)
            sold = int(min(dem[i], avail)); rem = sold
            for lot in lots:
                take = min(lot[0], rem); lot[0] -= take; rem -= take
            lots = [l for l in lots if l[0] > 0]
            closing = sum(q for q, _ in lots)
            stockout = dem[i] > avail
            hist.append(sold if not stockout else max(sold, dem[i] * 0.9))  # planner sees roughly true demand
            price = round(round(p._price0 * price_factor(d, pid)) * (1 - disc_arr[(wh, pid)][i]), 2)   # MRP in whole rupees
            sales_rows.append([d.date(), wh, pid, sold, price, bool(promo_flag[(wh, pid)][i]), bool(stockout)])
            in_transit = sum(x[1] for x in pipeline)
            inv_rows.append([d.date(), wh, pid, opening, received, sold, spoiled, adj, closing, in_transit])
            # reorder decision (manual planner policy: trailing-average, no festival foresight)
            look = np.array(hist[-28:]); avg = look.mean(); sd = look.std() if len(look) > 3 else avg * .3
            prow = cands[cands.vendor_id == preferred[pid]].iloc[0]
            L = int(prow.promised_lead_time_days) + 1
            z = 1.4 if p.perishable else 1.0
            ss = z * sd * math.sqrt(L)
            rop = avg * L + ss
            # planner usually uses the habit vendor, sometimes the alternative
            v_choice = preferred[pid] if rng.random() < 0.72 else str(cands[cands.vendor_id != preferred[pid]].vendor_id.iloc[0])
            row = cands[cands.vendor_id == v_choice].iloc[0]
            position = closing + in_transit
            forced_qty = None; forced_delay = None; forced_fill = None
            # scenario: Milk at Mumbai — V02 short-delivers & then a delayed PO is still in transit
            if wh == "WH-MUM" and pid == "P001" and d >= pd.Timestamp("2026-10-01"):
                if d == pd.Timestamp("2026-10-01"):
                    v_choice = "V02"; row = cands[cands.vendor_id == "V02"].iloc[0]
                    forced_qty = int(math.ceil(avg * 2.2 / 12) * 12); forced_delay = 0; forced_fill = 0.75
                elif d == pd.Timestamp("2026-10-03"):
                    v_choice = "V02"; row = cands[cands.vendor_id == "V02"].iloc[0]
                    forced_qty = int(math.ceil(avg * 3 / 12) * 12); forced_delay = 3; forced_fill = 1.0
                elif d == pd.Timestamp("2026-10-04"):   # emergency top-up from the reliable dairy
                    v_choice = "V01"; row = cands[cands.vendor_id == "V01"].iloc[0]
                    forced_qty = int(math.ceil(avg * 2.6 / 12) * 12); forced_delay = 0; forced_fill = 1.0
                else:
                    position = 10 ** 9   # planner waits for the delayed truck
            # scenario: Rice — Mumbai PO with V04 badly delayed, Nashik over-ordered
            if pid == "P004" and d >= pd.Timestamp("2026-09-21"):
                if wh == "WH-MUM":
                    if d == pd.Timestamp("2026-09-24"):
                        v_choice = "V04"; row = cands[cands.vendor_id == "V04"].iloc[0]
                        forced_qty = int(math.ceil(avg * 9 / 4) * 4); forced_delay = 10; forced_fill = 1.0
                    else:
                        position = 10 ** 9   # planner relies on the open PO
                if wh == "WH-NSK" and d == pd.Timestamp("2026-09-26"):
                    v_choice = "V03"; row = cands[cands.vendor_id == "V03"].iloc[0]
                    forced_qty = int(math.ceil(avg * 55 / 4) * 4); forced_delay = 0; forced_fill = 1.0
            # scenario: Ice cream Pune — big off-season order (overstock)
            if wh == "WH-PUN" and pid == "P012" and d == pd.Timestamp("2026-09-10"):
                v_choice = "V06"; row = cands[cands.vendor_id == "V06"].iloc[0]
                forced_qty = int(math.ceil(avg * 70 / 6) * 6); forced_delay = 0; forced_fill = 1.0
            if forced_qty is not None or position <= rop:
                if forced_qty is not None:
                    qty = forced_qty
                else:
                    S = rop + avg * COVER[g]
                    if p.perishable:   # never order more than can be sold before expiry
                        S = min(S, avg * (shelf - 0.3) + ss)
                    qty = max(S - position, row.min_order_qty)
                    m = int(p.order_multiple); qty = int(math.ceil(qty / m) * m)
                delay = vendor_delay(v_choice, i) if forced_delay is None else forced_delay
                lo, hi = VB[v_choice]["fill"]
                fill = forced_fill if forced_fill is not None else (1.0 if rng.random() < 0.8 else rng.uniform(lo, hi))
                recv = int(round(qty * fill))
                promised = i + int(row.promised_lead_time_days)
                actual = promised + delay
                unit_cost = p._cost0 * row._pf * price_factor(d, pid)
                if qty >= row.bulk_discount_min_qty:
                    unit_cost *= 1 - row.bulk_discount_pct / 100
                po_rows.append(dict(
                    po_id=None, order_date=d.date(), warehouse_id=wh, product_id=pid, vendor_id=v_choice,
                    qty_ordered=int(qty), unit_cost=round(unit_cost, 2),
                    total_cost=round(unit_cost * qty, 2),
                    promised_delivery_date=(START + pd.Timedelta(days=promised)).date(),
                    actual_delivery_date=(START + pd.Timedelta(days=actual)).date() if actual <= N - 1 else None,
                    qty_received=recv if actual <= N - 1 else None,
                    delay_days=delay if actual <= N - 1 else None,
                    status="In Transit"))
                if actual <= N - 1:
                    pipeline.append([actual, recv, len(po_rows) - 1])
                else:
                    pipeline.append([10 ** 9, recv, len(po_rows) - 1])
                    po_rows[-1]["qty_received"] = None

# ----------------------------------------------------------------------------- assemble & save
po = pd.DataFrame(po_rows).sort_values(["order_date", "warehouse_id", "product_id"]).reset_index(drop=True)
po["po_id"] = [f"PO-{i:05d}" for i in range(1, len(po) + 1)]
today = END.date()
late = (po.status == "In Transit") & (pd.to_datetime(po.promised_delivery_date) < pd.Timestamp(today))
po.loc[late, "status"] = "Delayed"
po = po[["po_id", "order_date", "warehouse_id", "product_id", "vendor_id", "qty_ordered", "unit_cost",
         "total_cost", "promised_delivery_date", "actual_delivery_date", "qty_received", "delay_days", "status"]]
po["qty_received"] = po.qty_received.astype("Int64"); po["delay_days"] = po.delay_days.astype("Int64")

sales = pd.DataFrame(sales_rows, columns=["date", "warehouse_id", "product_id", "units_sold", "unit_price",
                                          "on_promotion", "stockout"])
sales["revenue"] = (sales.units_sold * sales.unit_price).round(2)
inv = pd.DataFrame(inv_rows, columns=["date", "warehouse_id", "product_id", "opening_stock", "received",
                                      "sold", "spoiled", "adjustment", "closing_stock", "in_transit"])
adj = pd.DataFrame(adj_rows, columns=["date", "warehouse_id", "product_id", "qty_change", "reason"])
snap = inv[inv.date == today][["warehouse_id", "product_id", "closing_stock", "in_transit"]].rename(
    columns={"closing_stock": "stock_on_hand"})
snap.insert(0, "as_of_date", today)

prod_out = products.copy()
prod_out["unit_cost"] = [round(r._cost0 * price_factor(END, r.product_id), 2) for _, r in products.iterrows()]
prod_out["selling_price"] = [float(round(r._price0 * price_factor(END, r.product_id))) for _, r in products.iterrows()]
prod_out["holding_cost_pct_per_year"] = np.where(prod_out.perishable == 1, 30, 22)
prod_out["perishable"] = prod_out.perishable.astype(bool)
prod_out = prod_out[["product_id", "product_name", "category", "unit", "unit_cost", "selling_price",
                     "shelf_life_days", "perishable", "order_multiple", "holding_cost_pct_per_year"]]
vp_out = vendor_products.merge(prod_out[["product_id", "unit_cost"]], on="product_id")
vp_out["unit_price"] = (vp_out._pf * vp_out.unit_cost).round(2)
vp_out = vp_out[["vendor_id", "product_id", "unit_price", "promised_lead_time_days", "min_order_qty",
                 "bulk_discount_min_qty", "bulk_discount_pct"]]
wh_out = warehouses.drop(columns="_scale")
cal_out = calendar.copy(); cal_out["date"] = cal_out.date.dt.date

for name, df in [("products", prod_out), ("warehouses", wh_out), ("vendors", vendors),
                 ("vendor_products", vp_out), ("calendar", cal_out), ("promotions", promotions),
                 ("sales_history", sales), ("inventory_daily", inv), ("inventory_snapshot", snap),
                 ("purchase_orders", po), ("inventory_adjustments", adj)]:
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"{name:22s} {len(df):>7,d} rows")
