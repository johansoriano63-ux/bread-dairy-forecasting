"""FreshCast – forecasting and ordering agent for bread and dairy (Streamlit app).

Run:  streamlit run app/app.py
The app only reads the CSV files exported by src/final_experiments.py (app/data), so it needs no GPU or PyTorch.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="FreshCast – Bread & Dairy agent", page_icon="🥖", layout="wide")


@st.cache_data
def load():
    fc = pd.read_csv(DATA / "forecasts.csv", parse_dates=["date"])
    att = pd.read_csv(DATA / "attributions.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA / "stores.csv")
    met = pd.read_csv(DATA / "metrics_test.csv")
    meta = json.loads((DATA / "meta.json").read_text())
    return fc, att, stores, met, meta


fc, att, stores, met, meta = load()
FAMILIES = {"BREAD/BAKERY": "Bread / bakery", "DAIRY": "Dairy"}
REPO = "https://github.com/johansoriano63-ux/bread-dairy-forecasting"
TEAM = [("Stephany Trujillo", "Student", "stephany.trujillo@estudiantesunibague.edu.co"),
        ("Johan Soriano", "Student", "johan.soreano@estudiantesunibague.edu.co"),
        ("Cristian Cardona", "Student", "cristian.cardona@estudiantesunibague.edu.co"),
        ("Dr. Alexandra La Cruz", "Instructor", "alexandra.lacruz@unibague.edu.co")]
LINKS = [("📄 Report (paper)", f"{REPO}/tree/main/paper"), ("💻 Code on GitHub", REPO),
         ("🧪 Experiments in Colab", "https://colab.research.google.com/github/johansoriano63-ux/bread-dairy-forecasting/blob/main/notebooks/Final_Experiments_Bread_Dairy.ipynb"),
         ("🗂️ Dataset (Google Drive)", ""), ("🎬 Promotional video", "")]


def quantile_at(q10, q50, q90, tau):
    """Linear interpolation between the three quantiles in log(1+x) space (same rule as the experiments)."""
    tau = float(np.clip(tau, 0.1, 0.9))
    l10, l50, l90 = np.log1p(q10), np.log1p(q50), np.log1p(q90)
    v = l10 + (tau - 0.1) / 0.4 * (l50 - l10) if tau <= 0.5 else l50 + (tau - 0.5) / 0.4 * (l90 - l50)
    return np.expm1(v)


def plan_orders(levels, expected, life):
    """Order-up-to plan with FIFO perishable stock: order = target level - usable stock on hand."""
    stock = np.zeros(life); orders, on_hand_list = [], []
    for lv, d in zip(levels, expected):
        on_hand = stock.sum(); on_hand_list.append(on_hand)
        o = max(lv - on_hand, 0.0); orders.append(o); stock[-1] += o
        rem = d
        for a in range(life):
            take = min(stock[a], rem); stock[a] -= take; rem -= take
        stock[:-1] = stock[1:]; stock[-1] = 0
    return np.array(orders), np.array(on_hand_list)


# ---------------- sidebar ----------------
st.sidebar.title("🥖 FreshCast")
st.sidebar.caption("Forecasting and ordering agent for perishable families · Corporación Favorita (Kaggle)")
mode = st.sidebar.radio("Period", ["Forecast · Aug 16–31, 2017", "Backtest · Jul 31–Aug 15, 2017"])
future = mode.startswith("Forecast")
family = st.sidebar.selectbox("Family", list(FAMILIES), format_func=FAMILIES.get)
sinfo = stores.set_index("store_nbr")
store = st.sidebar.selectbox("Store", sorted(fc.store.unique()), index=43,
                             format_func=lambda s: f"{s} · {sinfo.loc[s, 'city']} (type {sinfo.loc[s, 'type']})")
model = "LSTM"
if not future:
    model = st.sidebar.radio("Model", ["LSTM", "LightGBM"], horizontal=True,
                             help="Seed ensembles with conformally recalibrated 10–90 % intervals")
st.sidebar.markdown("**Cost of a shortage vs. a leftover**")
ratio = st.sidebar.slider("c_u : c_o", 1.0, 6.0, 2.0, 0.5,
                          help="How much worse is a missing unit than a unit that is not sold")
tau = ratio / (ratio + 1)
life = 1
if family == "DAIRY":
    life = st.sidebar.slider("Dairy shelf life (days)", 2, 14, int(meta.get("dairy_life", 7)))
st.sidebar.caption(f"Critical quantile τ* = c_u/(c_u+c_o) = **{tau:.2f}**")

kind = "future" if future else ("backtest" if model == "LSTM" else "backtest_lgb")
d = fc[(fc.family == family) & (fc.store == store)]
hist = d[d.kind == "history"]
f = d[d.kind == kind].copy()
if future:
    hist = pd.concat([hist, d[d.kind == "backtest"]])          # the backtest period is real history here
hist = hist[hist.date < f.date.min()].tail(42)

f["target"] = quantile_at(f.q10.values, f.q50.values, f.q90.values, tau)
if life > 1:
    f["order"], f["stock_on_hand"] = plan_orders(f.target.values, f.q50.values, life)
else:
    f["order"], f["stock_on_hand"] = f.target, 0.0
if not future:
    f["outside"] = ((f.sales < f.q10) | (f.sales > f.q90)) & (f.open > 0)
    width = np.log1p(f.q90) - np.log1p(f.q10)
    dist = np.maximum(np.log1p(f.q10) - np.log1p(f.sales), np.log1p(f.sales) - np.log1p(f.q90))
    f["alert"] = np.where((dist > 0.5 * width) & (f.open > 0), "critical",
                          np.where(f.outside & f.outside.shift(1, fill_value=False), "warning", ""))

# ---------------- header ----------------
if meta.get("smoke"):
    st.warning("⚠️ Demo data from a short test run: the numbers are not the final results yet.")
st.title("🥖 FreshCast · forecasting and ordering agent for bread and dairy")
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏠 About the project", "📈 Forecast & order", "🔍 Why?",
                                        "🚨 Alerts (all stores)", "📊 Model quality"])

with tab1:
    st.header(f"{FAMILIES[family]} · store {store}")
    st.caption(f"{sinfo.loc[store, 'city']}, {sinfo.loc[store, 'state']} · type {sinfo.loc[store, 'type']} · "
               f"cluster {sinfo.loc[store, 'cluster']} · {'16-day forecast' if future else 'backtest against real sales'}")
    c1, c2, c3, c4 = st.columns(4)
    first = f.iloc[0]
    c1.metric(f"Order for {first.date:%b %d}", f"{first.order:,.0f} units",
              help="Newsvendor quantity" + (" minus usable stock (order-up-to rule)" if life > 1 else ""))
    c2.metric("Expected sales, 16 days", f"{f.q50.sum():,.0f}", f"80 % range {f.q10.sum():,.0f}–{f.q90.sum():,.0f}",
              delta_color="off")
    c3.metric("Days with promotion", int((f.promo > 0).sum()))
    if future:
        c4.metric("Paydays / holidays", f"{int(f.payday.sum())} / {int((f.holiday > 0).sum())}")
    else:
        real = f.loc[f.open > 0, "sales"].sum()
        wape = 100 * (f.loc[f.open > 0, "sales"] - f.loc[f.open > 0, "q50"]).abs().sum() / max(real, 1)
        c4.metric("Alerts in the window", int((f.alert != "").sum()), f"WAPE {wape:.1f} %", delta_color="off")

with tab0:
    st.markdown("## Less waste, fewer empty shelves")
    st.markdown(
        "Bread is baked and thrown away the same day; milk and yogurt last only a few days more. Every morning each "
        "store has to decide **how much to have on the shelf**: too much is waste, too little is a lost sale. "
        "**FreshCast** is an intelligent agent that learns the daily demand of the **BREAD/BAKERY** and **DAIRY** "
        "families of 54 Corporación Favorita supermarkets (Ecuador) and turns it into decisions that store staff can use.")
    a1, a2, a3, a4 = st.columns(4)
    a1.markdown("##### 📈 Forecast\n16 days ahead, as a range (10–90 %) and not a single number.")
    a2.markdown("##### 🛒 Order\nHow much to have each day, given how costly a shortage is.")
    a3.markdown("##### 🔍 Explain\nWhich factors move each day's forecast: promotions, weekday, paydays, holidays.")
    a4.markdown("##### 🚨 Alert\nWarns when real sales leave the expected range.")

    def kpi(model, fam, col="RMSLE"):
        r = met[(met.model == model) & (met.family == fam)]
        return float(r[col].iloc[0]) if len(r) else np.nan
    st.markdown("### Results on the holdout test (Jul 31 – Aug 15, 2017)")
    k = st.columns(4)
    for j, fam in enumerate(FAMILIES):
        lstm, naive = kpi("LSTM ensemble + conformal", fam), kpi("Seasonal naive", fam)
        k[2 * j].metric(f"{FAMILIES[fam]}: error vs. naive", f"{100 * (lstm - naive) / naive:+.0f} %",
                        help="RMSLE reduction of the LSTM with respect to repeating last week's sales")
        k[2 * j + 1].metric(f"{FAMILIES[fam]}: interval coverage", f"{kpi('LSTM ensemble + conformal', fam, 'Coverage_pct'):.0f} %",
                            help="Share of days whose real sales fell inside the 10–90 % range (target 80 %)")

    c_left, c_right = st.columns([1, 1.2])
    with c_left:
        st.markdown("### How it works")
        st.markdown(
            "1. **Data**: 108 daily series (54 stores × 2 families), 2013–2017, with promotions, national, regional "
            "and local holidays, paydays, oil price and store transactions ([Kaggle Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting)).\n"
            "2. **Neural core**: an **LSTM encoder–decoder** reads 8 weeks of history and the known future "
            "(promotions, holidays, calendar) and outputs three quantiles per day that can never cross.\n"
            "3. **Calibration**: 5 training seeds are averaged and the range is corrected with **conformal "
            "prediction**, so that 80 % of real sales fall inside it.\n"
            "4. **Decisions**: the **newsvendor** quantile gives the bread order; for dairy an **order-up-to** rule "
            "discounts the stock carried from previous days.\n"
            "5. **Explanations**: **integrated gradients**, checked with occlusion and retraining ablations.\n\n"
            "It was compared, under one protocol, with a seasonal naive model, ETS, quantile LightGBM, a GRU and a "
            "Temporal Fusion Transformer.")
    with c_right:
        st.image(str(Path(__file__).parent / "assets" / "architecture.png"), caption="Architecture of the agent")
    st.markdown("### What the data show")
    st.image(str(Path(__file__).parent / "assets" / "eda_dow.png"), width=520,
             caption="Relative sales by day of week: Sunday sells about 30 % above the mean, Thursday about 20 % below.")

    st.markdown("### Team")
    t = st.columns(4)
    for col, (name, role, mail) in zip(t, TEAM):
        col.markdown(f"**{name}**  \n{role}  \n[{mail}](mailto:{mail})")
    st.caption("Intelligent Systems course · Faculty of Engineering, Universidad de Ibagué, Colombia · 2026")
    st.markdown("### Resources")
    st.markdown(" · ".join(f"[{label}]({url})" if url else f"{label} (coming soon)" for label, url in LINKS))
    st.info("👈 Use the panel on the left to pick a store and a family, then open the **Forecast & order** tab.")

with tab1:
    band = alt.Chart(f).mark_area(opacity=.25, color="#1f77b4").encode(
        x=alt.X("date:T", title=None), y=alt.Y("q10:Q", title="units"), y2="q90:Q")
    med = alt.Chart(f).mark_line(color="#1f77b4").encode(x="date:T", y="q50:Q",
                                                         tooltip=["date:T", alt.Tooltip("q10:Q", format=",.0f"),
                                                                  alt.Tooltip("q50:Q", format=",.0f"), alt.Tooltip("q90:Q", format=",.0f")])
    order = alt.Chart(f).mark_line(color="#d62728", strokeDash=[4, 3]).encode(x="date:T", y="target:Q")
    hline = alt.Chart(hist).mark_line(color="black", strokeWidth=1).encode(x="date:T", y="sales:Q")
    layers = [band, med, order, hline]
    if not future:
        layers.append(alt.Chart(f).mark_line(color="black", point=True, strokeWidth=1).encode(x="date:T", y="sales:Q"))
        al = f[f.alert != ""]
        if len(al):
            layers.append(alt.Chart(al).mark_point(shape="triangle-up", size=160, filled=True).encode(
                x="date:T", y="sales:Q", color=alt.Color("alert:N", scale=alt.Scale(domain=["warning", "critical"],
                                                                                   range=["orange", "red"])), tooltip=["date:T", "alert:N"]))
    st.altair_chart(alt.layer(*layers).properties(height=340), use_container_width=True)
    st.caption("Black: real sales · blue: median and 10–90 % interval · red dashed: availability target at τ*")
    show = f[["date", "q10", "q50", "q90", "target", "stock_on_hand", "order", "promo", "payday", "holiday"]
             + ([] if future else ["sales", "alert"])].copy()
    show["date"] = show.date.dt.strftime("%a %b %d")
    show = show.rename(columns={"q10": "low (10 %)", "q50": "median", "q90": "high (90 %)", "target": f"target (τ={tau:.2f})",
                                "stock_on_hand": "stock carried", "order": "recommended order", "promo": "items on promo"})
    if life == 1: show = show.drop(columns=["stock carried"])
    st.dataframe(show.style.format(precision=0), use_container_width=True, hide_index=True)
    if life > 1:
        st.info(f"Dairy keeps about {life} days, so the agent orders only what is missing to reach the target, "
                "assuming median sales on the previous days (order-up-to rule).")

with tab2:
    a = att[(att.family == family) & (att.store == store) & (att.kind == ("future" if future else "backtest"))]
    day = st.select_slider("Day", options=list(f.date.dt.date), format_func=lambda x: x.strftime("%a %b %d"))
    ad = a[a.date.dt.date == day].copy()
    ad["effect_pct"] = 100 * (np.exp(ad.attribution) - 1)
    ad["direction"] = np.where(ad.effect_pct >= 0, "raises the forecast", "lowers the forecast")
    ch = alt.Chart(ad).mark_bar().encode(
        x=alt.X("effect_pct:Q", title="approx. effect on the median forecast (%)"),
        y=alt.Y("factor:N", sort="-x", title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises the forecast", "lowers the forecast"],
                                                       range=["#2ca02c", "#d62728"]), legend=None),
        tooltip=["factor", alt.Tooltip("effect_pct:Q", format="+.1f")])
    st.altair_chart(ch.properties(height=220), use_container_width=True)
    top = ad.reindex(ad.effect_pct.abs().sort_values(ascending=False).index).head(2)
    msg = "; ".join(f"**{r.factor}** {r.direction} by about {abs(r.effect_pct):.0f} %" for r in top.itertuples())
    st.markdown(f"On {day:%A %b %d}: {msg}, compared with an average week without promotions, holidays or paydays.")
    st.caption("Integrated-gradient attributions of the LSTM (seed 42). They were checked against occlusion and "
               "retraining ablations; see the report for their limits.")

with tab3:
    if future:
        st.info("Alerts need real sales. Switch to the backtest period to see them.")
    else:
        allf = fc[fc.kind == kind].copy()
        lo, hi, y = np.log1p(allf.q10), np.log1p(allf.q90), np.log1p(allf.sales)
        allf["outside"] = ((y < lo) | (y > hi)) & (allf.open > 0)
        allf["critical"] = (np.maximum(lo - y, y - hi) > 0.5 * (hi - lo)) & (allf.open > 0)
        allf = allf.sort_values(["family", "store", "date"])
        allf["warning"] = allf.outside & allf.groupby(["family", "store"]).outside.shift(1, fill_value=False)
        summ = (allf.groupby(["family", "store"]).agg(days_outside=("outside", "sum"), warnings=("warning", "sum"),
                                                      critical=("critical", "sum")).reset_index())
        summ = summ[(summ.warnings + summ.critical) > 0].sort_values(["critical", "warnings"], ascending=False)
        summ["city"] = summ.store.map(sinfo.city)
        st.markdown(f"**{len(summ)}** of 108 store–family series raised at least one alert with the {model} model.")
        st.dataframe(summ, use_container_width=True, hide_index=True)
        st.caption("Warning: real sales outside the 10–90 % interval two days in a row. "
                   "Critical: outside by more than half the interval width on one day.")

with tab4:
    m = met[met.model.isin(["Seasonal naive", "ETS", "LightGBM ensemble + conformal", "GRU ensemble + conformal",
                            "LSTM ensemble + conformal", "TFT"])]
    st.dataframe(m[["model", "family", "RMSLE", "WAPE_pct", "Pinball_log", "Coverage_pct"]]
                 .sort_values(["family", "RMSLE"]), use_container_width=True, hide_index=True)
    st.caption("Holdout test window Jul 31 – Aug 15, 2017 (open days). Coverage target: 80 %.")
