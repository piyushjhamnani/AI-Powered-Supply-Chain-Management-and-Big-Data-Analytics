"""
Demand Forecasting Agent
- One global ML model (HistGradientBoosting, Poisson loss) for all warehouse x product series.
- Features: calendar (weekday, month, salary week, Shravan, festival window), promotions,
  lagged demand (28/35/42 days), rolling means and last-year same-week demand.
- Because every lag is >= 28 days, it forecasts 28 days ahead directly (no error snowballing).
- Recent-level correction: if the last 7 days ran hotter/colder than the model expected,
  the forecast is nudged (half-way, capped) toward the new level.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from config import FORECAST_HORIZON, HOLDOUT_START

FEATURES = ["wh_code", "pid_code", "dow", "month", "dom", "is_weekend", "is_salary_week", "is_shravan",
            "fest_code", "on_promotion", "lag28", "lag35", "lag42", "roll7_l28", "roll28_l28", "yoy7"]
CATEGORICAL = ["wh_code", "pid_code", "fest_code"]


class ForecastAgent:
    name = "Forecast Agent"

    def __init__(self, data):
        self.products = data["products"]
        self.calendar = data["calendar"].copy()
        self.promotions = data["promotions"]
        self.wh_codes = {w: i for i, w in enumerate(sorted(data["warehouses"].warehouse_id))}
        self.pid_codes = {p: i for i, p in enumerate(sorted(self.products.product_id))}
        fests = sorted(self.calendar.festival_window.fillna("").unique())
        self.fest_codes = {f: i for i, f in enumerate(fests)}
        self.model = None
        self.metrics = None

    # ------------------------------------------------------------------ features
    def _panel(self, sales: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
        series = sales[["warehouse_id", "product_id"]].drop_duplicates()
        dates = pd.DataFrame({"date": pd.date_range(sales.date.min(), end_date)})
        grid = series.merge(dates, how="cross")
        df = grid.merge(sales[["date", "warehouse_id", "product_id", "units_sold", "stockout", "on_promotion"]],
                        on=["date", "warehouse_id", "product_id"], how="left")
        df = df.sort_values(["warehouse_id", "product_id", "date"]).reset_index(drop=True)
        g = df.groupby(["warehouse_id", "product_id"]).units_sold
        df["lag28"] = g.shift(28)
        df["lag35"] = g.shift(35)
        df["lag42"] = g.shift(42)
        df["roll7_l28"] = g.transform(lambda s: s.shift(28).rolling(7, min_periods=3).mean())
        df["roll28_l28"] = g.transform(lambda s: s.shift(28).rolling(28, min_periods=7).mean())
        df["yoy7"] = g.transform(lambda s: s.shift(361).rolling(7, min_periods=3).mean())
        cal = self.calendar[["date", "is_weekend", "is_salary_week", "is_shravan", "festival_window"]]
        df = df.merge(cal, on="date", how="left")
        df["dow"] = df.date.dt.dayofweek
        df["month"] = df.date.dt.month
        df["dom"] = df.date.dt.day
        df["fest_code"] = df.festival_window.fillna("").map(self.fest_codes).fillna(0).astype(int)
        df["wh_code"] = df.warehouse_id.map(self.wh_codes)
        df["pid_code"] = df.product_id.map(self.pid_codes)
        # planned promotions are known in advance; past promo flag comes from sales
        df["on_promotion"] = df.on_promotion.astype("float").fillna(0)
        for b in ["is_weekend", "is_salary_week", "is_shravan"]:
            df[b] = df[b].astype(float)
        return df

    def _new_model(self):
        cat_mask = [f in CATEGORICAL for f in FEATURES]
        return HistGradientBoostingRegressor(loss="poisson", max_iter=350, learning_rate=0.06,
                                             max_leaf_nodes=31, min_samples_leaf=20,
                                             categorical_features=cat_mask, random_state=7)

    def _train(self, panel, cutoff):
        tr = panel[(panel.date < cutoff) & panel.units_sold.notna() & (panel.stockout != True)
                   & panel.lag28.notna()]
        m = self._new_model()
        m.fit(tr[FEATURES], tr.units_sold)
        return m

    # ------------------------------------------------------------------ public API
    def fit(self, sales):
        end = sales.date.max()
        panel = self._panel(sales, end)
        self.model = self._train(panel, end + pd.Timedelta(days=1))
        return self

    def _level_and_sigma(self, panel, model, start):
        hist = panel[(panel.date < start) & (panel.date >= start - pd.Timedelta(days=56))
                     & panel.units_sold.notna()].copy()
        hist["pred"] = model.predict(hist[FEATURES])
        ok = hist[hist.stockout != True]
        out = []
        for (wh, pid), h in ok.groupby(["warehouse_id", "product_id"]):
            last7 = h[h.date >= start - pd.Timedelta(days=7)]
            ratio = last7.units_sold.sum() / max(1e-6, last7.pred.sum()) if len(last7) >= 3 else 1.0
            level = float(np.clip(1 + 0.5 * (ratio - 1), 0.8, 1.6))
            sigma = float(np.std(h.units_sold - h.pred)) if len(h) > 7 else float(h.units_sold.std())
            out.append(dict(warehouse_id=wh, product_id=pid, level_factor=round(level, 3),
                            recent_ratio=round(ratio, 2), sigma=max(sigma, 1.0)))
        return pd.DataFrame(out)

    def forecast(self, sales, start, horizon=FORECAST_HORIZON):
        """Returns (daily forecast frame, per-series stats frame)."""
        end = start + pd.Timedelta(days=horizon - 1)
        panel = self._panel(sales, end)
        fut = panel[(panel.date >= start) & (panel.date <= end)].copy()
        fut["base_forecast"] = self.model.predict(fut[FEATURES])
        stats = self._level_and_sigma(panel, self.model, start)
        fut = fut.merge(stats[["warehouse_id", "product_id", "level_factor"]], on=["warehouse_id", "product_id"],
                        how="left").fillna({"level_factor": 1.0})
        fut["forecast"] = (fut.base_forecast * fut.level_factor).clip(lower=0)
        return fut[["warehouse_id", "product_id", "date", "base_forecast", "forecast", "festival_window"]], stats

    def evaluate(self, sales):
        """Holdout test: train before HOLDOUT_START, forecast to end, compare with the old manual method."""
        start = pd.Timestamp(HOLDOUT_START)
        end = sales.date.max()
        panel = self._panel(sales, end)
        model = self._train(panel, start)
        stats = self._level_and_sigma(panel, model, start)
        test = panel[(panel.date >= start) & (panel.date <= end) & (panel.stockout != True)].copy()
        test["model"] = model.predict(test[FEATURES])
        test = test.merge(stats[["warehouse_id", "product_id", "level_factor"]], on=["warehouse_id", "product_id"])
        test["model"] *= test.level_factor
        # baseline = what the human planner did: average of last 28 days, frozen at the start of the test
        base = (sales[(sales.date < start) & (sales.date >= start - pd.Timedelta(days=28))]
                .groupby(["warehouse_id", "product_id"]).units_sold.mean().rename("baseline"))
        test = test.merge(base, on=["warehouse_id", "product_id"])
        test["err_m"] = (test.units_sold - test.model).abs()
        test["err_b"] = (test.units_sold - test.baseline).abs()
        by = test.groupby("product_id").agg(actual=("units_sold", "sum"), em=("err_m", "sum"), eb=("err_b", "sum"))
        by["ai_accuracy_%"] = (100 * (1 - by.em / by.actual)).round(1)
        by["manual_accuracy_%"] = (100 * (1 - by.eb / by.actual)).round(1)
        by = by.reset_index().merge(self.products[["product_id", "product_name"]], on="product_id")
        overall_m = 100 * (1 - test.err_m.sum() / test.units_sold.sum())
        overall_b = 100 * (1 - test.err_b.sum() / test.units_sold.sum())
        self.metrics = dict(ai_accuracy=round(overall_m, 1), manual_accuracy=round(overall_b, 1),
                            by_product=by[["product_id", "product_name", "ai_accuracy_%", "manual_accuracy_%"]],
                            test_days=int((end - start).days + 1))
        return self.metrics

    def festival_watch(self, sales, today, days_ahead=45):
        """Upcoming festivals + which products jumped last time (seasonal intelligence)."""
        cal = self.calendar
        upcoming = cal[(cal.date >= today) & (cal.date <= today + pd.Timedelta(days=days_ahead))
                       & cal.festival.fillna("").ne("")]
        rows = []
        for r in upcoming.itertuples():
            name = r.festival
            past = cal[(cal.festival == name) & (cal.date < today - pd.Timedelta(days=60))]
            if past.empty:
                continue
            last = past.date.max()
            win = cal[(cal.festival_window == name) & (cal.date >= last - pd.Timedelta(days=20))
                      & (cal.date <= last + pd.Timedelta(days=12))].date
            if win.empty:
                continue
            w0 = win.min()
            s_win = sales[sales.date.isin(win)].groupby("product_id").units_sold.mean()
            s_pre = sales[(sales.date < w0) & (sales.date >= w0 - pd.Timedelta(days=28))
                          ].groupby("product_id").units_sold.mean()
            uplift = (s_win / s_pre).dropna()
            for pid, u in uplift[(uplift >= 1.2) | (uplift <= 0.8)].sort_values(ascending=False).items():
                rows.append(dict(festival=name, festival_date=r.date.date(),
                                 days_away=(r.date - today).days, product_id=pid, last_year_uplift=round(u, 2),
                                 window_starts=(r.date - (last - w0)).date()))
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.merge(self.products[["product_id", "product_name"]], on="product_id")
        return df
