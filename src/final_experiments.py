# %% [markdown]
# # Final Experiments – Probabilistic Forecasting of Bakery and Dairy Sales
# **Intelligent Systems – Universidad de Ibagué** · Stephany Trujillo, Johan Soriano, Cristian Cardona · Instructor: Dr. Alexandra La Cruz
#
# This notebook reproduces every result of the final report under one protocol. Compared with phase 3 it adds:
#
# | Step | What it does |
# |---|---|
# | 1–3 | Data, covariates (including the unlabeled Aug 16–31, 2017 period of the Kaggle test file) and metrics |
# | 4 | Exploratory analysis |
# | 5–6 | Baselines (seasonal naive, ETS) and quantile LightGBM |
# | 7 | LSTM / GRU encoder–decoder with embeddings and non-crossing quantile head |
# | 8 | **Hyperparameter tuning** of the LSTM on the validation windows |
# | 9 | **Several seeds** for LSTM, GRU and LightGBM, and seed ensembles |
# | 10 | **Conformal recalibration** of the 10–90 % intervals |
# | 11 | Results tables and figures |
# | 12 | Newsvendor order, **multi-day order-up-to rule for dairy**, anomaly alerts |
# | 13 | Integrated gradients and their **validation by occlusion and retraining ablation** |
# | 14 | Temporal Fusion Transformer |
# | 15 | Forecast of Aug 16–31, 2017 and **export of the data used by the Streamlit app** |
#
# **Colab:** `Runtime → Change runtime type → T4 GPU`, upload `train.csv, test.csv, stores.csv, oil.csv,
# holidays_events.csv, transactions.csv` to `/content/data` (or use the Google Drive / Kaggle options in step 1) and `Run all`.
# **Locally:** `python src/final_experiments.py` with the CSV files in `data/`.

# %%
import os, sys, time, copy, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib
IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB and not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    display
except NameError:
    def display(x): print(x.to_string() if hasattr(x, "to_string") else x)

SEED = 42
SEEDS = [42, 7, 123, 2024, 31415][: int(os.environ.get("N_SEEDS", 5))]
DATA_DIR = os.environ.get("DATA_DIR", "/content/data" if IN_COLAB else "data")
OUT_DIR = os.environ.get("OUT_DIR", "/content/results" if IN_COLAB else "results")
# Colab: keep results (and the resume cache) in Google Drive, so a disconnection does not lose finished steps.
# After a disconnection, simply run all cells again: finished models are loaded from the cache.
SAVE_TO_DRIVE = True
if IN_COLAB and SAVE_TO_DRIVE:
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"): drive.mount("/content/drive")
        OUT_DIR = "/content/drive/MyDrive/freshcast_results"
        print("Results will be saved in Google Drive:", OUT_DIR)
    except Exception as e:
        print("WARNING: Google Drive could not be mounted (", e, ").\n"
              "Results are saved in /content/results instead; if Colab disconnects, the run starts again.")
APP_DIR = os.environ.get("APP_DIR", f"{OUT_DIR}/app_data")
CACHE_DIR = f"{OUT_DIR}/cache"
for d in (DATA_DIR, OUT_DIR, APP_DIR, CACHE_DIR): os.makedirs(d, exist_ok=True)

FAMILIES = ["BREAD/BAKERY", "DAIRY"]
FAM_SHORT = ["Bread", "Dairy"]
ENC_LEN, H = 56, 16
VAL_ORIGINS = ["2017-05-28", "2017-06-13", "2017-06-29", "2017-07-15"]
TEST_ORIGIN = "2017-07-31"
FUTURE_ORIGIN = "2017-08-16"              # Kaggle test period: no public labels
QUANTILES = [0.1, 0.5, 0.9]
RUN_TUNING = os.environ.get("RUN_TUNING", "1") == "1"
RUN_ABLATION = os.environ.get("RUN_ABLATION", "1") == "1"
RUN_TFT = os.environ.get("RUN_TFT", "1") == "1"
SMOKE = os.environ.get("SMOKE", "0") == "1"   # quick end-to-end check with tiny training budgets
DAIRY_LIFE = 7                            # assumed shelf life of dairy (days) for the inventory simulation
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.grid": True, "grid.alpha": .3})

def cached(name, fn):
    """Store expensive results so the script can be resumed."""
    path = f"{CACHE_DIR}/{name}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f: return pickle.load(f)
    out = fn()
    with open(path, "wb") as f: pickle.dump(out, f)
    return out

def savefig(name):
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight"); plt.show(); plt.close("all")

# %% [markdown]
# ## 1. Data

# %%
# Option A (Colab): copy the files from Google Drive
USE_DRIVE = False
DRIVE_FOLDER = "/content/drive/MyDrive/store-sales-time-series-forecasting"
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"): drive.mount("/content/drive")
    os.system(f"cp {DRIVE_FOLDER}/*.csv {DATA_DIR}/ 2>/dev/null; cd {DATA_DIR} && unzip -o -q {DRIVE_FOLDER}/*.zip 2>/dev/null")
# Option B (Colab): Kaggle API (needs kaggle.json)
USE_KAGGLE_API = False
if IN_COLAB and USE_KAGGLE_API:
    from google.colab import files
    files.upload()
    os.system("mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    os.system(f"kaggle competitions download -c store-sales-time-series-forecasting -p {DATA_DIR}")
    os.system(f"cd {DATA_DIR} && unzip -o -q store-sales-time-series-forecasting.zip")

needed = ["train.csv", "test.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv"]
# Option C: upload the Kaggle zip (store-sales-time-series-forecasting.zip) to DATA_DIR; it is extracted here
import glob, zipfile
for z in glob.glob(os.path.join(DATA_DIR, "*.zip")) + glob.glob("/content/*.zip"):
    if any(not os.path.exists(os.path.join(DATA_DIR, f)) for f in needed):
        with zipfile.ZipFile(z) as zf:
            for f in needed:
                if f in zf.namelist(): zf.extract(f, DATA_DIR)
missing = [f for f in needed if not os.path.exists(os.path.join(DATA_DIR, f))]
assert not missing, f"Put these files in {DATA_DIR}: {missing}"

# %%
train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["date"])
stores = pd.read_csv(f"{DATA_DIR}/stores.csv")
oil = pd.read_csv(f"{DATA_DIR}/oil.csv", parse_dates=["date"])
hol = pd.read_csv(f"{DATA_DIR}/holidays_events.csv", parse_dates=["date"])
trans = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["date"])

LAST_OBS = train.date.max()
dates = pd.date_range(train.date.min(), test.date.max(), freq="D")    # observed period + 16 unlabeled days
T = len(dates); T_OBS = int((dates <= LAST_OBS).sum())
date_idx = {d: i for i, d in enumerate(dates)}
store_ids = sorted(stores.store_nbr.unique())
series = [(f, s) for f in FAMILIES for s in store_ids]
S = len(series)
print(f"{T_OBS} observed days ({dates[0].date()} – {LAST_OBS.date()}) + {T - T_OBS} future days, {S} series")

store_day = (train.groupby(["store_nbr", "date"])["sales"].sum()
             .unstack("date").reindex(index=store_ids, columns=dates).fillna(0))
CLOSED_STORE = (store_day.values == 0).astype(np.float32)
CLOSED_STORE[:, T_OBS:] = 0                                            # future days: assume open

sub = pd.concat([train[train.family.isin(FAMILIES)], test[test.family.isin(FAMILIES)].assign(sales=0.0)])
mi = pd.MultiIndex.from_tuples(series, names=["family", "store_nbr"])
def to_matrix(col):
    return (sub.pivot_table(index=["family", "store_nbr"], columns="date", values=col, aggfunc="sum")
               .reindex(index=mi, columns=dates).fillna(0).values.astype(np.float32))
SALES = to_matrix("sales")
PROMO_RAW = to_matrix("onpromotion")
Y = np.log1p(SALES)

st_pos = np.array([store_ids.index(s) for _, s in series])
CLOSED = CLOSED_STORE[st_pos]
FIRST = np.array([np.argmax(r > 0) if (r > 0).any() else T for r in SALES])
OPEN = ((np.arange(T)[None, :] >= FIRST[:, None]) & (CLOSED == 0)).astype(np.float32)
OPEN[:, T_OBS:] = 0                                                    # no labels in the future period
print("Series whose store opened after Feb 2013:", int((FIRST > 31).sum()))

# %% [markdown]
# ## 2. Covariates

# %%
tr_m = (trans.pivot_table(index="store_nbr", columns="date", values="transactions", aggfunc="sum")
             .reindex(index=store_ids, columns=dates).fillna(0).values)
TRANS = (np.log1p(tr_m)[st_pos] / 10).astype(np.float32)
OIL = (oil.set_index("date")["dcoilwtico"].reindex(dates)
          .interpolate(limit_direction="both").values / 100).astype(np.float32)

DOW, DOM, MONTH = dates.dayofweek.values, dates.day.values, dates.month.values
PAYDAY = ((DOM == 15) | dates.is_month_end).astype(np.float32)
QUAKE = ((dates >= "2016-04-16") & (dates <= "2016-05-15")).astype(np.float32)

h = hol[(hol.transferred.astype(str).str.lower() != "true") & (hol.type != "Work Day")]
nat_days = set(h.loc[h.locale == "National", "date"])
HOL = np.zeros((S, T, 3), np.float32)
HOL[:, :, 0] = np.array([d in nat_days for d in dates], np.float32)
sinfo = stores.set_index("store_nbr")
reg, loc = h[h.locale == "Regional"], h[h.locale == "Local"]
for i, (_, s) in enumerate(series):
    for d in reg.loc[reg.locale_name == sinfo.loc[s, "state"], "date"]:
        if d in date_idx: HOL[i, date_idx[d], 1] = 1
    for d in loc.loc[loc.locale_name == sinfo.loc[s, "city"], "date"]:
        if d in date_idx: HOL[i, date_idx[d], 2] = 1
PROMO = (np.log1p(PROMO_RAW) / 5).astype(np.float32)

# KNOWN covariates (18 channels): 0-6 day of week | 7-8 day of month sin/cos | 9-10 month sin/cos | 11 payday
#                                 12 national holiday | 13 regional | 14 local | 15 promotion | 16 closed | 17 earthquake
cal = np.concatenate([np.eye(7)[DOW],
                      np.sin(2*np.pi*DOM/31)[:, None], np.cos(2*np.pi*DOM/31)[:, None],
                      np.sin(2*np.pi*(MONTH-1)/12)[:, None], np.cos(2*np.pi*(MONTH-1)/12)[:, None],
                      PAYDAY[:, None]], axis=1).astype(np.float32)
KNOWN = np.concatenate([np.broadcast_to(cal, (S, T, 12)), HOL, PROMO[..., None], CLOSED[..., None],
                        np.broadcast_to(QUAKE[None, :, None], (S, T, 1))], axis=-1).astype(np.float32)
PAST = np.stack([TRANS, np.broadcast_to(OIL, (S, T))], axis=-1).astype(np.float32)

FAM_IDX = np.array([FAMILIES.index(f) for f, _ in series])
STORE_IDX = st_pos
TYPE_IDX = pd.Categorical(sinfo.loc[[s for _, s in series], "type"]).codes.astype(int)
CLUSTER_IDX = pd.Categorical(sinfo.loc[[s for _, s in series], "cluster"]).codes.astype(int)
CARD = (len(store_ids), int(TYPE_IDX.max())+1, int(CLUSTER_IDX.max())+1, len(FAMILIES))
VAL_O = [date_idx[pd.Timestamp(d)] for d in VAL_ORIGINS]
TEST_O = date_idx[pd.Timestamp(TEST_ORIGIN)]
FUT_O = date_idx[pd.Timestamp(FUTURE_ORIGIN)]
assert FUT_O == T_OBS and FUT_O + H == T
print("KNOWN", KNOWN.shape, "PAST", PAST.shape, "cardinalities", CARD)

# %% [markdown]
# ## 3. Metrics (identical for every model, open days only)

# %%
def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)

def metrics(preds, fam):
    """preds: {origin: [S, H, 3] log(1+x) quantiles}"""
    rows = FAM_IDX == fam
    os_ = list(preds)
    y = np.stack([Y[rows, o:o+H] for o in os_]); m = np.stack([OPEN[rows, o:o+H] for o in os_]) > 0
    q = np.maximum(np.sort(np.stack([preds[o][rows] for o in os_]), axis=-1), 0)
    yl, q10, q50, q90 = y[m], q[..., 0][m], q[..., 1][m], q[..., 2][m]
    yu, pu = np.expm1(yl), np.expm1(q50)
    return dict(RMSLE=np.sqrt(np.mean((q50 - yl) ** 2)),
                WAPE_pct=100 * np.abs(yu - pu).sum() / yu.sum(),
                MAE=np.abs(yu - pu).mean(),
                Pinball_log=np.mean([pinball(yl, qq, t).mean() for qq, t in zip([q10, q50, q90], QUANTILES)]),
                Coverage_pct=100 * np.mean((yl >= q10) & (yl <= q90)),
                Width_log=np.mean(q90 - q10))

def evaluate(name, preds, extra=None):
    out = []
    for split, os_ in [("validation", VAL_O), ("test", [TEST_O])]:
        if not all(o in preds for o in os_): continue
        for fam in range(2):
            out.append(dict(model=name, split=split, family=FAMILIES[fam], **(extra or {}),
                            **metrics({o: preds[o] for o in os_}, fam)))
    return out

PREDS = {}

# %% [markdown]
# ## 4. Exploratory analysis

# %%
obs = slice(0, T_OBS)
fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
for i, f in enumerate(FAMILIES):
    tot = pd.Series(SALES[FAM_IDX == i, obs].sum(0), index=dates[obs])
    ax[i].plot(tot.rolling(7).mean(), lw=1); ax[i].set_title(f"{f}: total daily sales, 54 stores (7-day mean)")
    ax[i].axvspan(pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-15"), color="red", alpha=.15, label="earthquake")
    ax[i].legend(loc="upper left")
savefig("fig_eda_total_sales")

eda = []
mask = OPEN > 0
rel = SALES / np.maximum(np.nanmean(np.where(mask, SALES, np.nan), axis=1, keepdims=True), 1e-9)
for i, f in enumerate(FAMILIES):
    r = (FAM_IDX == i)[:, None] & mask
    by_dow = [rel[r & (DOW[None] == d)].mean() for d in range(7)]
    eda.append(dict(family=f,
        weekend_vs_weekday=np.mean(by_dow[5:]) / np.mean(by_dow[:5]),
        payday_vs_other=rel[r & (PAYDAY[None] == 1)].mean() / rel[r & (PAYDAY[None] == 0)].mean(),
        national_holiday_vs_other=rel[r & (HOL[..., 0] == 1)].mean() / rel[r & (HOL[..., 0] == 0)].mean(),
        promo_vs_no_promo=rel[r & (PROMO_RAW > 0)].mean() / rel[r & (PROMO_RAW == 0)].mean(),
        **{f"dow_{d}": v for d, v in zip(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], by_dow)}))
EDA = pd.DataFrame(eda).round(3); EDA.to_csv(f"{OUT_DIR}/eda_summary.csv", index=False); display(EDA)

plt.figure(figsize=(5, 2.8))
for i, f in enumerate(FAMILIES):
    plt.plot(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], EDA.loc[i, [c for c in EDA if c.startswith("dow_")]].values, marker="o", label=f)
plt.ylabel("sales / series mean"); plt.legend()
savefig("fig_eda_dow")

# %% [markdown]
# ## 5. Baselines: seasonal naive and ETS

# %%
Z = 1.2816
def seasonal_naive(o):
    P = np.zeros((S, H, 3), np.float32)
    hist, op = Y[:, o-364:o], OPEN[:, o-364:o] > 0
    d, ok = hist[:, 7:] - hist[:, :-7], op[:, 7:] & op[:, :-7]
    for s in range(S):
        dd = d[s][ok[s]]
        lo, hi = np.quantile(dd, [0.1, 0.9]) if len(dd) > 20 else (-0.3, 0.3)
        fc = Y[s, o-7:o][np.arange(H) % 7]
        P[s, :, 0], P[s, :, 1], P[s, :, 2] = fc + lo, fc, fc + hi
    return P

from statsmodels.tsa.holtwinters import ExponentialSmoothing
def ets(o):
    P = np.zeros((S, H, 3), np.float32)
    naive = seasonal_naive(o)
    for s in range(S):
        y, m = Y[s, o-364:o].astype(float).copy(), OPEN[s, o-364:o] > 0
        for t in range(7, len(y)):
            if not m[t]: y[t] = y[t-7]
        try:
            fit = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=7,
                                       initialization_method="estimated").fit()
            fc, sd = fit.forecast(H), np.std(fit.resid[28:])
            P[s, :, 0], P[s, :, 1], P[s, :, 2] = fc - Z*sd, fc, fc + Z*sd
        except Exception:
            P[s] = naive[s]
    return P

PREDS["Seasonal naive"] = {o: seasonal_naive(o) for o in VAL_O + [TEST_O]}
PREDS["ETS"] = cached("ets", lambda: {o: ets(o) for o in VAL_O + [TEST_O]})

# %% [markdown]
# ## 6. Quantile LightGBM (global, direct multi-horizon)

# %%
import lightgbm as lgb
CAT_COLS = ["store", "type", "cluster", "family"]

def lgb_features(o):
    s_idx, hh = np.repeat(np.arange(S), H), np.tile(np.arange(H), S)
    t = o + hh
    rep = lambda v: np.repeat(v, H)
    X = {"h": hh, "store": STORE_IDX[s_idx], "type": TYPE_IDX[s_idx], "cluster": CLUSTER_IDX[s_idx],
         "family": FAM_IDX[s_idx], "dow": DOW[t], "dom": DOM[t], "month": MONTH[t], "payday": PAYDAY[t],
         "hol_nat": HOL[s_idx, t, 0], "hol_reg": HOL[s_idx, t, 1], "hol_loc": HOL[s_idx, t, 2],
         "promo": PROMO[s_idx, t], "closed": CLOSED[s_idx, t], "quake": QUAKE[t],
         "last1": rep(Y[:, o-1]), "std28": rep(Y[:, o-28:o].std(1)),
         "promo_hist28": rep(PROMO[:, o-28:o].mean(1)), "trans7": rep(TRANS[:, o-7:o].mean(1)),
         "oil_last": np.full(S*H, OIL[o-1])}
    for k in (7, 14, 28, 56):
        X[f"mean{k}"] = rep(Y[:, o-k:o].mean(1))
    k0 = hh // 7 + 1
    sw = np.stack([Y[s_idx, t - 7*(k0 + j)] for j in range(4)], 1)
    X["sdow_last"], X["sdow_mean4"] = sw[:, 0], sw.mean(1)
    X = pd.DataFrame(X)
    keep = (OPEN[s_idx, t] > 0) & (o - ENC_LEN >= FIRST[s_idx])
    return X, Y[s_idx, t], keep

LGB_START = date_idx[pd.Timestamp("2015-01-01")]
def lgb_dataset(last_target, step=4):
    Xs, ys = [], []
    for o in range(max(ENC_LEN, LGB_START), last_target - H + 2, step):
        X, y, k = lgb_features(o); Xs.append(X[k]); ys.append(y[k])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)

def lgb_predict(models, o):
    X, _, _ = lgb_features(o)
    return np.sort(np.stack([m.predict(X) for m in models], -1).reshape(S, H, 3), -1).astype(np.float32)

LGB_PARAMS = dict(objective="quantile", n_estimators=1500, learning_rate=0.06, num_leaves=63,
                  min_child_samples=100, max_bin=127, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  verbose=-1, n_jobs=os.cpu_count())
if SMOKE: LGB_PARAMS["n_estimators"] = 40
LGB_DATA = {}
def run_lgb(seed):
    t0 = time.time()
    if "val" not in LGB_DATA:
        Xtr, ytr = lgb_dataset(VAL_O[0] - 1)
        _va = [lgb_features(o) for o in VAL_O]
        Xva = pd.concat([X[k] for X, _, k in _va], ignore_index=True); yva = np.concatenate([y[k] for _, y, k in _va])
        LGB_DATA["val"] = (Xtr, ytr, Xva, yva); LGB_DATA["test"] = lgb_dataset(TEST_O - 1)
    Xtr, ytr, Xva, yva = LGB_DATA["val"]
    mv, iters = [], []
    for tau in QUANTILES:
        m = lgb.LGBMRegressor(alpha=tau, random_state=seed, **LGB_PARAMS)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], categorical_feature=CAT_COLS,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        mv.append(m); iters.append(max(m.best_iteration_ or LGB_PARAMS["n_estimators"], 50))
    P = {o: lgb_predict(mv, o) for o in VAL_O}
    Xtr2, ytr2 = LGB_DATA["test"]
    mt = []
    for tau, it in zip(QUANTILES, iters):
        m = lgb.LGBMRegressor(alpha=tau, random_state=seed, **{**LGB_PARAMS, "n_estimators": it})
        m.fit(Xtr2, ytr2, categorical_feature=CAT_COLS); mt.append(m)
    P[TEST_O] = lgb_predict(mt, TEST_O)
    imp = pd.Series(mt[1].feature_importances_, index=Xtr2.columns)
    print(f"LightGBM seed {seed}: iterations {iters}, {(time.time()-t0)/60:.1f} min")
    return dict(preds=P, iters=iters, importance=imp)

# %% [markdown]
# ## 7. LSTM / GRU encoder–decoder with embeddings and non-crossing quantile head

# %%
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu": torch.set_num_threads(os.cpu_count())
print("device:", DEVICE)
Y_T, OPEN_T = torch.tensor(Y, device=DEVICE), torch.tensor(OPEN, device=DEVICE)
KNOWN_T, PAST_T = torch.tensor(KNOWN, device=DEVICE), torch.tensor(PAST, device=DEVICE)
CATS_T = torch.tensor(np.stack([STORE_IDX, TYPE_IDX, CLUSTER_IDX, FAM_IDX], 1), dtype=torch.long, device=DEVICE)
OFFS = torch.arange(-ENC_LEN, H, device=DEVICE)
QT = torch.tensor(QUANTILES, device=DEVICE)
N_KNOWN, N_PAST = KNOWN.shape[-1], PAST.shape[-1]

# Factor groups used by the explanations and the ablations (channels of KNOWN)
GROUPS = {"Recent history": None, "Day of week": list(range(7)), "Payday": [11],
          "Holiday": [12, 13, 14], "Promotion": [15]}

def ablate_inputs(enc, dec, group):
    """Replace one factor group by its neutral value (same baseline used by integrated gradients)."""
    enc, dec = enc.clone(), dec.clone()
    if group == "Recent history":
        enc[:, :, 0] = enc[:, :, 0].mean(1, keepdim=True)
    else:
        ch = GROUPS[group]; val = 1/7 if group == "Day of week" else 0.0
        enc[:, :, [1 + N_PAST + c for c in ch]] = val; dec[:, :, ch] = val
    return enc, dec

def make_batch(s, o, ablate=None):
    tidx = o[:, None] + OFFS[None]
    sidx = s[:, None].expand_as(tidx)
    y, op, kn, pa = Y_T[sidx, tidx], OPEN_T[sidx, tidx], KNOWN_T[sidx, tidx], PAST_T[sidx, tidx]
    ye, oe = y[:, :ENC_LEN], op[:, :ENC_LEN]
    scale = ((ye * oe).sum(1) / oe.sum(1).clamp(min=1)).clamp(min=1.0)
    ys = y / scale[:, None]
    enc = torch.cat([ys[:, :ENC_LEN, None], pa[:, :ENC_LEN], kn[:, :ENC_LEN]], -1)
    dec = kn[:, ENC_LEN:]
    if ablate: enc, dec = ablate_inputs(enc, dec, ablate)
    return enc, dec, CATS_T[s], ys[:, ENC_LEN:], op[:, ENC_LEN:], scale

class Seq2SeqQuantile(nn.Module):
    def __init__(self, cell="lstm", hidden=64, layers=2, dropout=0.2, emb=(8, 3, 4, 2)):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, e) for c, e in zip(CARD, emb)])
        E, RNN = sum(emb), (nn.LSTM if cell == "lstm" else nn.GRU)
        self.encoder = RNN(1 + N_PAST + N_KNOWN + E, hidden, layers, batch_first=True, dropout=dropout)
        self.decoder = RNN(N_KNOWN + E, hidden, layers, batch_first=True, dropout=dropout)
        self.drop, self.head = nn.Dropout(dropout), nn.Linear(hidden, 3)
    def forward(self, enc, dec, cats):
        e = torch.cat([m(cats[:, i]) for i, m in enumerate(self.embs)], -1)
        enc = torch.cat([enc, e[:, None].expand(-1, enc.size(1), -1)], -1)
        dec = torch.cat([dec, e[:, None].expand(-1, dec.size(1), -1)], -1)
        _, state = self.encoder(enc)
        out, _ = self.decoder(dec, state)
        z = self.head(self.drop(out))
        med = z[..., 0]
        return torch.stack([med - F.softplus(z[..., 1]), med, med + F.softplus(z[..., 2])], -1)

def qloss(pred, y, m):
    d = y[..., None] - pred
    return (torch.maximum(QT * d, (QT - 1) * d).mean(-1) * m).sum() / m.sum().clamp(min=1)

def train_pairs(last_target):
    ss, oo = [], []
    for s in range(S):
        start, end = FIRST[s] + ENC_LEN, last_target - H + 1
        if end >= start:
            o = np.arange(start, end + 1); ss.append(np.full(len(o), s)); oo.append(o)
    return torch.tensor(np.concatenate(ss), device=DEVICE), torch.tensor(np.concatenate(oo), device=DEVICE)

def all_series(o):
    return torch.arange(S, device=DEVICE), torch.full((S,), o, device=DEVICE, dtype=torch.long)

def train_nn(cell, last_target, val_origins=None, epochs=40, steps=300, bs=256, lr=1e-3, patience=6,
             seed=SEED, ablate=None, verbose=False, **kw):
    if SMOKE: epochs, steps = min(epochs, 2), 10
    torch.manual_seed(seed); np.random.seed(seed)
    model = Seq2SeqQuantile(cell, **kw).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ps, po = train_pairs(last_target)
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    best, best_state, best_ep, wait, hist = 1e9, None, epochs, 0, []
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for _ in range(steps):
            i = torch.randint(0, len(ps), (bs,), device=DEVICE, generator=g)
            enc, dec, cats, yt, mt, _ = make_batch(ps[i], po[i], ablate)
            loss = qloss(model(enc, dec, cats), yt, mt)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item()
        row = {"epoch": ep, "train": tl / steps}
        if val_origins:
            model.eval()
            with torch.no_grad():
                vl = []
                for o in val_origins:
                    enc, dec, cats, yt, mt, _ = make_batch(*all_series(o), ablate)
                    vl.append(qloss(model(enc, dec, cats), yt, mt).item())
                row["val"] = float(np.mean(vl))
            if row["val"] < best:
                best, best_state, best_ep, wait = row["val"], copy.deepcopy(model.state_dict()), ep, 0
            else:
                wait += 1
        hist.append(row)
        if verbose: print(f"epoch {ep:02d}  train {row['train']:.4f}" + (f"  val {row['val']:.4f}" if val_origins else ""))
        if val_origins and wait >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, best_ep, best, pd.DataFrame(hist)

@torch.no_grad()
def nn_predict(model, o, ablate=None):
    model.eval()
    enc, dec, cats, _, _, scale = make_batch(*all_series(o), ablate)
    return (model(enc, dec, cats) * scale[:, None, None]).clamp(min=0).cpu().numpy()

def run_nn(cell, seed, cfg, future=False):
    """Stage 1: train up to May 27 with early stopping on the 4 validation windows.
       Stage 2: retrain from scratch up to Jul 30 for the selected epochs and forecast the test window.
       Stage 3 (optional): retrain up to Aug 15 and forecast the unlabeled Aug 16-31 period."""
    t0 = time.time()
    mv, best_ep, best_val, hist = train_nn(cell, VAL_O[0] - 1, val_origins=VAL_O, seed=seed, **cfg)
    P = {o: nn_predict(mv, o) for o in VAL_O}
    mt, _, _, _ = train_nn(cell, TEST_O - 1, epochs=best_ep, seed=seed, **cfg)
    P[TEST_O] = nn_predict(mt, TEST_O)
    out = dict(preds=P, best_ep=best_ep, best_val=best_val, hist=hist,
               state_val=copy.deepcopy(mv.state_dict()), state_test=copy.deepcopy(mt.state_dict()))
    if future:
        mf, _, _, _ = train_nn(cell, FUT_O - 1, epochs=best_ep, seed=seed, **cfg)
        out["future"] = nn_predict(mf, FUT_O); out["state_future"] = copy.deepcopy(mf.state_dict())
    print(f"{cell.upper()} seed {seed} {cfg}: best epoch {best_ep}, val loss {best_val:.5f}, {(time.time()-t0)/60:.1f} min")
    return out

def load_model(cell, cfg, state):
    m = Seq2SeqQuantile(cell, **{k: v for k, v in cfg.items() if k in ("hidden", "layers", "dropout", "emb")}).to(DEVICE)
    m.load_state_dict(state); m.eval(); return m

# %% [markdown]
# ## 8. Hyperparameter tuning of the LSTM (validation windows only, seed 42)

# %%
BASE_CFG = dict(hidden=64, dropout=0.2, lr=1e-3)
GRID = [BASE_CFG, dict(hidden=128, dropout=0.2, lr=1e-3), dict(hidden=64, dropout=0.1, lr=1e-3),
        dict(hidden=128, dropout=0.3, lr=1e-3), dict(hidden=64, dropout=0.2, lr=2e-3)]
if SMOKE: GRID = GRID[:2]
def tune():
    rows = []
    for cfg in GRID:
        mv, ep, best, _ = train_nn("lstm", VAL_O[0] - 1, val_origins=VAL_O, seed=SEED, **cfg)
        P = {o: nn_predict(mv, o) for o in VAL_O}
        r = dict(**cfg, best_epoch=ep, val_pinball=best)
        for fam in range(2): r[f"val_RMSLE_{FAM_SHORT[fam]}"] = metrics(P, fam)["RMSLE"]
        print(r); rows.append(r)
    return pd.DataFrame(rows)
if RUN_TUNING:
    TUNE = cached("tuning", tune); TUNE.to_csv(f"{OUT_DIR}/tuning_lstm.csv", index=False); display(TUNE.round(4))
    BEST_CFG = {k: TUNE.loc[TUNE.val_pinball.idxmin(), k] for k in ("hidden", "dropout", "lr")}
    BEST_CFG = dict(hidden=int(BEST_CFG["hidden"]), dropout=float(BEST_CFG["dropout"]), lr=float(BEST_CFG["lr"]))
else:
    BEST_CFG = BASE_CFG
print("selected configuration:", BEST_CFG)

# %% [markdown]
# ## 9. Several seeds and seed ensembles

# %%
RUNS = {"LSTM": {}, "GRU": {}, "LightGBM": {}}
for seed in SEEDS:
    RUNS["LSTM"][seed] = cached(f"lstm_{seed}", lambda: run_nn("lstm", seed, BEST_CFG, future=True))
    RUNS["GRU"][seed] = cached(f"gru_{seed}", lambda: run_nn("gru", seed, BEST_CFG))
    RUNS["LightGBM"][seed] = cached(f"lgb_{seed}", lambda: run_lgb(seed))

def ensemble(runs):
    keys = list(next(iter(runs.values()))["preds"])
    return {o: np.sort(np.mean([r["preds"][o] for r in runs.values()], 0), -1) for o in keys}

for name in RUNS:
    PREDS[f"{name} (seed {SEED})"] = RUNS[name][SEED]["preds"]
    PREDS[f"{name} ensemble"] = ensemble(RUNS[name])
FUTURE_LSTM = np.sort(np.mean([r["future"] for r in RUNS["LSTM"].values()], 0), -1)

SEED_ROWS = []
for name, runs in RUNS.items():
    for seed, r in runs.items():
        SEED_ROWS += evaluate(name, r["preds"], {"seed": seed})
SEED_RES = pd.DataFrame(SEED_ROWS); SEED_RES.to_csv(f"{OUT_DIR}/results_by_seed.csv", index=False)
SEED_SUM = (SEED_RES.groupby(["split", "family", "model"])[["RMSLE", "WAPE_pct", "Pinball_log", "Coverage_pct"]]
            .agg(["mean", "std"]).round(4))
SEED_SUM.to_csv(f"{OUT_DIR}/results_seed_summary.csv"); display(SEED_SUM)
print("epochs selected by seed:", {n: [r.get("best_ep") for r in RUNS[n].values()] for n in ("LSTM", "GRU")})

# Is the LSTM better than LightGBM? paired comparison by seed on the test window
cmp = SEED_RES[SEED_RES.split == "test"].pivot_table(index=["family", "seed"], columns="model", values="RMSLE")
cmp["LSTM_minus_LGB"] = cmp["LSTM"] - cmp["LightGBM"]
display(cmp.round(4)); cmp.to_csv(f"{OUT_DIR}/lstm_vs_lgb_by_seed.csv")

# %% [markdown]
# ## 10. Conformal recalibration of the 10–90 % interval
# Conformalized quantile regression (asymmetric): on the four validation windows we measure how far the real
# value falls below q0.1 and above q0.9 and widen (or narrow) each bound, per family, so that each tail holds
# 10 % of the validation days. The same correction is applied to the test forecasts.

# %%
def conformal_offsets(val_preds):
    adj = {}
    for fam in range(2):
        rows = FAM_IDX == fam
        y = np.stack([Y[rows, o:o+H] for o in VAL_O]); m = np.stack([OPEN[rows, o:o+H] for o in VAL_O]) > 0
        q = np.sort(np.stack([val_preds[o][rows] for o in VAL_O]), -1)
        lo, hi = (q[..., 0] - y)[m], (y - q[..., 2])[m]
        n = len(lo); level = min(1.0, 0.9 * (1 + 1 / n))
        adj[fam] = (float(np.quantile(lo, level)), float(np.quantile(hi, level)))
    return adj

def apply_conformal(P, adj):
    P = P.copy()
    for fam, (a_lo, a_hi) in adj.items():
        r = FAM_IDX == fam
        P[r, :, 0] = np.minimum(P[r, :, 0] - a_lo, P[r, :, 1])
        P[r, :, 2] = np.maximum(P[r, :, 2] + a_hi, P[r, :, 1])
    return P

CONF = {}
for name in ["LSTM ensemble", "GRU ensemble", "LightGBM ensemble", "ETS"]:
    adj = conformal_offsets(PREDS[name]); CONF[name] = adj
    PREDS[f"{name} + conformal"] = {o: apply_conformal(P, adj) for o, P in PREDS[name].items()}
pd.DataFrame([dict(model=n, family=FAMILIES[f], adj_lower=a[0], adj_upper=a[1]) for n, d in CONF.items() for f, a in d.items()]
             ).round(4).to_csv(f"{OUT_DIR}/conformal_offsets.csv", index=False)
MAIN = "LSTM ensemble + conformal"

# %% [markdown]
# ## 11. Results

# %%
ROWS = []
for name, p in PREDS.items(): ROWS += evaluate(name, p)
RES = pd.DataFrame(ROWS); RES.to_csv(f"{OUT_DIR}/results_final.csv", index=False)
cols = ["model", "family", "RMSLE", "WAPE_pct", "MAE", "Pinball_log", "Coverage_pct", "Width_log"]
for split in ["validation", "test"]:
    print(f"\n===== {split.upper()} ====="); display(RES[RES.split == split][cols].sort_values(["family", "RMSLE"]).round(4))

# Figure: test RMSLE by seed (box) for the three learned models
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for fam in range(2):
    d = SEED_RES[(SEED_RES.split == "test") & (SEED_RES.family == FAMILIES[fam])]
    data = [d[d.model == m].RMSLE.values for m in ["LightGBM", "LSTM", "GRU"]]
    ax[fam].boxplot(data, widths=.5); ax[fam].set_xticks([1, 2, 3]); ax[fam].set_xticklabels(["LightGBM", "LSTM", "GRU"])
    for j, m in enumerate(["LightGBM", "LSTM", "GRU"]):
        ens = RES[(RES.split == "test") & (RES.family == FAMILIES[fam]) & (RES.model == f"{m} ensemble")].RMSLE.values
        ax[fam].scatter([j + 1], ens, marker="*", s=90, c="C3", zorder=3, label="seed ensemble" if j == 0 else None)
    ax[fam].set_title(f"{FAMILIES[fam]} – test RMSLE ({len(SEEDS)} seeds)"); ax[fam].legend(fontsize=7)
savefig("fig_seeds_rmsle")

# Figure: coverage before / after conformal recalibration
fig, ax = plt.subplots(figsize=(6.5, 2.8))
names = ["ETS", "LightGBM ensemble", "LSTM ensemble", "GRU ensemble"]
lab = ["ETS", "LightGBM", "LSTM", "GRU"]
x = np.arange(len(names)); w = .2
for k, (fam, post) in enumerate([(0, ""), (0, " + conformal"), (1, ""), (1, " + conformal")]):
    v = [RES[(RES.split == "test") & (RES.family == FAMILIES[fam]) & (RES.model == n + post)].Coverage_pct.values[0] for n in names]
    ax.bar(x + (k - 1.5) * w, v, w, label=f"{FAM_SHORT[fam]}{' recalibrated' if post else ' raw'}",
           color=["C0", "C0", "C1", "C1"][k], alpha=[.45, 1, .45, 1][k])
ax.axhline(80, ls="--", c="k", lw=1); ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("coverage of 10–90 % (%)")
ax.set_ylim(50, 100); ax.legend(fontsize=7, ncol=2); ax.set_title("Test window: interval coverage before and after recalibration")
savefig("fig_coverage_conformal")

# Figure: training curve of the selected LSTM (seed 42)
hist = RUNS["LSTM"][SEED]["hist"]; best_ep = RUNS["LSTM"][SEED]["best_ep"]
hist.to_csv(f"{OUT_DIR}/training_curve_lstm.csv", index=False)
plt.figure(figsize=(4.5, 2.6)); plt.plot(hist.epoch, hist.train, label="train"); plt.plot(hist.epoch, hist.val, label="validation")
plt.axvline(best_ep, ls="--", c="gray"); plt.xlabel("epoch"); plt.ylabel("pinball loss (scaled)"); plt.legend()
savefig("fig_training_lstm")

# Figure: forecast example (store with the highest bread sales)
s_store = int(np.argmax(SALES[FAM_IDX == 0, T_OBS-365:T_OBS].sum(1)))
fig, ax = plt.subplots(2, 1, figsize=(7, 4.6))
for fam in range(2):
    s = fam * len(store_ids) + s_store
    d_hist, d_fut = dates[TEST_O-42:TEST_O], dates[TEST_O:TEST_O+H]
    q = np.expm1(PREDS[MAIN][TEST_O][s])
    ax[fam].plot(d_hist, SALES[s, TEST_O-42:TEST_O], c="k", lw=1, label="history")
    ax[fam].plot(d_fut, SALES[s, TEST_O:TEST_O+H], c="k", marker="o", ms=3, lw=1, label="real (test)")
    ax[fam].plot(d_fut, q[:, 1], c="C0", label="LSTM median")
    ax[fam].fill_between(d_fut, q[:, 0], q[:, 2], color="C0", alpha=.25, label="10–90 % interval (recalibrated)")
    ax[fam].set_title(f"Store {store_ids[s_store]} – {FAMILIES[fam]}"); ax[fam].legend(fontsize=7, loc="upper left")
savefig("fig_forecast_example")

# %% [markdown]
# ## 12. Decision layer: newsvendor, multi-day rule for dairy and anomaly alerts

# %%
def quantile_at(q, tau):
    tau = float(np.clip(tau, 0.1, 0.9))
    if tau <= 0.5: return q[..., 0] + (tau - 0.1) / 0.4 * (q[..., 1] - q[..., 0])
    return q[..., 1] + (tau - 0.5) / 0.4 * (q[..., 2] - q[..., 1])

y_test, m_test = SALES[:, TEST_O:TEST_O+H], OPEN[:, TEST_O:TEST_O+H] > 0
NV = []
for name in ["LSTM ensemble", MAIN, "LightGBM ensemble", "LightGBM ensemble + conformal", "GRU ensemble + conformal"]:
    q = np.sort(PREDS[name][TEST_O], -1)
    for ratio in [2, 4]:
        tau = ratio / (ratio + 1)
        for fam in range(2):
            r = (FAM_IDX == fam)[:, None] & m_test
            cost = lambda Qty: (ratio * np.maximum(y_test - Qty, 0) + np.maximum(Qty - y_test, 0))[r].sum()
            c_med, c_nv = cost(np.expm1(q[..., 1])), cost(np.expm1(quantile_at(q, tau)))
            NV.append(dict(model=name, family=FAMILIES[fam], cost_ratio=f"{ratio}:1", tau_star=round(tau, 3),
                           cost_median=round(c_med), cost_newsvendor=round(c_nv), saving_pct=round(100*(c_med-c_nv)/c_med, 2)))
NV = pd.DataFrame(NV); NV.to_csv(f"{OUT_DIR}/newsvendor_final.csv", index=False); display(NV)

# %% [markdown]
# **Multi-day rule for dairy.** Bread leftovers are lost at the end of the day, so the daily newsvendor quantity is
# the order. Dairy keeps several days, so we simulate a perishable inventory (FIFO, shelf life `DAIRY_LIFE` days,
# no lead time) and compare: (P1) ordering the newsvendor quantity every day ignoring the stock on hand,
# (P2) order-up-to the median, (P3) order-up-to the τ* quantile, and (P4) order-up-to a quantile τ selected on the
# four validation windows. Cost = c_u · units short + c_o · units expired.

# %%
def simulate(levels, demand, open_, life, carry=True):
    n = demand.shape[0]; stock = np.zeros((n, life)); short = waste = sold = 0.0
    short_v, waste_v, sold_v = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(demand.shape[1]):
        on_hand = stock.sum(1)
        order = np.where(open_[:, t], np.maximum(levels[:, t] - on_hand, 0) if carry else levels[:, t], 0)
        stock[:, life - 1] += order
        rem = np.where(open_[:, t], demand[:, t], 0).astype(float)
        for a in range(life):
            take = np.minimum(stock[:, a], rem); stock[:, a] -= take; rem -= take; sold_v += take
        short_v += rem; waste_v += stock[:, 0]
        stock[:, :-1] = stock[:, 1:]; stock[:, -1] = 0
    return short_v, waste_v, sold_v

def policy_cost(preds, origins, fam, tau, ratio, life, carry):
    tot = dict(short=0, waste=0, sold=0, demand=0)
    rows = FAM_IDX == fam
    for o in origins:
        q = np.sort(preds[o][rows], -1)
        lv = np.expm1(quantile_at(q, tau)); d = SALES[rows, o:o+H]; op = OPEN[rows, o:o+H] > 0
        sh, wa, so = simulate(lv, d, op, life, carry)
        tot["short"] += sh.sum(); tot["waste"] += wa.sum(); tot["sold"] += so.sum(); tot["demand"] += d[op].sum()
    tot["cost"] = ratio * tot["short"] + tot["waste"]
    return tot

TAU_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2)
INV = []
for fam, life in [(1, DAIRY_LIFE), (1, 3), (0, 1)]:
    for ratio in [2, 4]:
        tau_star = ratio / (ratio + 1)
        val_costs = {t: policy_cost(PREDS[MAIN], VAL_O, fam, t, ratio, life, True)["cost"] for t in TAU_GRID}
        tau_val = min(val_costs, key=val_costs.get)
        pols = [("P1 newsvendor, ignores stock", tau_star, False), ("P2 order-up-to median", 0.5, True),
                ("P3 order-up-to tau*", tau_star, True), ("P4 order-up-to tau (validation)", tau_val, True)]
        if life == 1: pols = [pols[1], pols[2], pols[3]]
        for pname, tau, carry in pols:
            c = policy_cost(PREDS[MAIN], [TEST_O], fam, tau, ratio, life, carry)
            INV.append(dict(family=FAMILIES[fam], shelf_life=life, cost_ratio=f"{ratio}:1", policy=pname, tau=tau,
                            cost=round(c["cost"]), waste_pct=round(100 * c["waste"] / c["demand"], 2),
                            fill_rate_pct=round(100 * c["sold"] / c["demand"], 2)))
INV = pd.DataFrame(INV)
INV["saving_vs_first_pct"] = INV.groupby(["family", "shelf_life", "cost_ratio"]).cost.transform(lambda c: 100 * (c.iloc[0] - c) / c.iloc[0]).round(1)
INV.to_csv(f"{OUT_DIR}/inventory_policies.csv", index=False); display(INV)

d = INV[(INV.family == "DAIRY") & (INV.shelf_life == DAIRY_LIFE)]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for k, ratio in enumerate(["2:1", "4:1"]):
    dd = d[d.cost_ratio == ratio]
    ax[k].barh([p.split(" ", 1)[0] for p in dd.policy], dd.cost / 1000, color=["C7", "C0", "C1", "C2"])
    for j, (c, w_) in enumerate(zip(dd.cost / 1000, dd.waste_pct)): ax[k].text(c, j, f" waste {w_:.1f}%", va="center", fontsize=7)
    ax[k].invert_yaxis(); ax[k].set_xlabel("simulated cost (thousand units)"); ax[k].set_title(f"DAIRY, c_u:c_o = {ratio}")
    ax[k].set_xlim(0, dd.cost.max() / 1000 * 1.45)
savefig("fig_dairy_policies")

# %%
AL = []
for name in ["LSTM ensemble", MAIN, "LightGBM ensemble", "LightGBM ensemble + conformal"]:
    q = np.sort(PREDS[name][TEST_O], -1); y = Y[:, TEST_O:TEST_O+H]
    out = ((y < q[..., 0]) | (y > q[..., 2])) & m_test
    width = q[..., 2] - q[..., 0]; dist = np.maximum(q[..., 0] - y, y - q[..., 2])
    warn = out[:, 1:] & out[:, :-1]; crit = (dist > 0.5 * width) & m_test
    for fam in range(2):
        r = FAM_IDX == fam
        AL.append(dict(model=name, family=FAMILIES[fam], days_outside_pct=round(100 * out[r].sum() / m_test[r].sum(), 2),
                       warnings=int(warn[r].sum()), critical=int(crit[r].sum()),
                       stores_with_alert=int((warn[r].any(1) | crit[r].any(1)).sum())))
AL = pd.DataFrame(AL); AL.to_csv(f"{OUT_DIR}/anomaly_alerts_final.csv", index=False); display(AL)

# %% [markdown]
# ## 13. Explanations: integrated gradients, occlusion and retraining ablation

# %%
LSTM_TEST = load_model("lstm", BEST_CFG, RUNS["LSTM"][SEED]["state_test"])

def integrated_gradients(model, o, steps=32):
    model.eval()
    enc, dec, cats, _, _, _ = make_batch(*all_series(o))
    eb, db = enc.clone(), dec.clone()
    for g in GROUPS: eb, db = ablate_inputs(eb, db, g)
    ge_sum, gd_sum = torch.zeros_like(enc), torch.zeros_like(dec)
    with torch.backends.cudnn.flags(enabled=False):
        for a in np.linspace(1/steps, 1, steps):
            xe = (eb + float(a) * (enc - eb)).requires_grad_(True)
            xd = (db + float(a) * (dec - db)).requires_grad_(True)
            ge, gd = torch.autograd.grad(model(xe, xd, cats)[..., 1].sum(), [xe, xd])
            ge_sum += ge; gd_sum += gd
    ae, ad = ((enc - eb) * ge_sum / steps).cpu().numpy(), ((dec - db) * gd_sum / steps).cpu().numpy()
    per_day = {}
    for g, ch in GROUPS.items():
        if ch is None: per_day[g] = np.repeat(ae[:, :, 0].sum(1, keepdims=True) / H, H, 1)
        else: per_day[g] = ae[:, :, [1 + N_PAST + c for c in ch]].sum(2).sum(1, keepdims=True) / H + ad[:, :, ch].sum(2)
    return per_day                                         # {group: [S, H]} contribution to the scaled log median

def occlusion(model, o):
    base = nn_predict(model, o)[..., 1]
    return {g: np.abs(nn_predict(model, o, ablate=g)[..., 1] - base) for g in GROUPS}

IG_DAY = integrated_gradients(LSTM_TEST, TEST_O)
OCC = occlusion(LSTM_TEST, TEST_O)
expl = []
for fam in range(2):
    r = FAM_IDX == fam
    ig = {g: np.abs(v[r].sum(1)).mean() for g, v in IG_DAY.items()}
    oc = {g: v[r][m_test[r]].mean() for g, v in OCC.items()}
    for g in GROUPS:
        expl.append(dict(family=FAMILIES[fam], factor=g, IG_share_pct=100 * ig[g] / sum(ig.values()),
                         occlusion_share_pct=100 * oc[g] / sum(oc.values()), occlusion_mean_abs_log=oc[g]))
EXPL = pd.DataFrame(expl)

def retrain_ablation():
    rows = []
    for g in [None] + list(GROUPS):
        mv, ep, best, _ = train_nn("lstm", VAL_O[0] - 1, val_origins=VAL_O, seed=SEED, ablate=g, **BEST_CFG)
        P = {o: nn_predict(mv, o, ablate=g) for o in VAL_O}
        for fam in range(2):
            rows.append(dict(factor=g or "none", family=FAMILIES[fam], val_RMSLE=metrics(P, fam)["RMSLE"],
                             val_pinball=metrics(P, fam)["Pinball_log"]))
        print("ablation", g, "done")
    return pd.DataFrame(rows)
if RUN_ABLATION:
    ABL = cached("ablation", retrain_ablation)
    full = ABL[ABL.factor == "none"].set_index("family").val_RMSLE
    ABL["delta_RMSLE_pct"] = 100 * (ABL.val_RMSLE / ABL.family.map(full) - 1)
    ABL.to_csv(f"{OUT_DIR}/ablation_retrain.csv", index=False); display(ABL.round(4))
    EXPL = EXPL.merge(ABL[["factor", "family", "delta_RMSLE_pct"]], on=["factor", "family"], how="left")
EXPL.round(3).to_csv(f"{OUT_DIR}/explanations_validation.csv", index=False); display(EXPL.round(2))

fig, ax = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for fam in range(2):
    e = EXPL[EXPL.family == FAMILIES[fam]].set_index("factor")
    yy = np.arange(len(e))
    ax[fam].barh(yy - .2, e.IG_share_pct, .4, label="integrated gradients")
    ax[fam].barh(yy + .2, e.occlusion_share_pct, .4, label="occlusion")
    ax[fam].set_yticks(yy); ax[fam].set_yticklabels(e.index); ax[fam].set_title(FAMILIES[fam]); ax[fam].set_xlabel("share of attribution (%)")
ax[0].legend(fontsize=7); ax[0].invert_yaxis()
savefig("fig_explanations")

# %% [markdown]
# ## 14. Temporal Fusion Transformer (PyTorch Forecasting)
# Trained with targets up to Jul 14, 2017 and early-stopped on validation window 4, then forecasts the test window.

# %%
def run_tft():
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    pl.seed_everything(SEED)
    rows = []
    for s in range(S):
        t = np.arange(max(FIRST[s], 0), T_OBS)
        rows.append(pd.DataFrame({"series": str(s), "time_idx": t, "y": Y[s, t],
            "store_id": str(STORE_IDX[s]), "fam": FAMILIES[FAM_IDX[s]], "store_type": str(TYPE_IDX[s]),
            "store_cluster": str(CLUSTER_IDX[s]), "dow": DOW[t].astype(str), "payday": PAYDAY[t],
            "hol_nat": HOL[s, t, 0], "hol_reg": HOL[s, t, 1], "hol_loc": HOL[s, t, 2],
            "promo": PROMO[s, t], "closed": CLOSED[s, t], "quake": QUAKE[t],
            "dom_sin": KNOWN[s, t, 7], "dom_cos": KNOWN[s, t, 8], "m_sin": KNOWN[s, t, 9], "m_cos": KNOWN[s, t, 10],
            "trans": TRANS[s, t], "oil": OIL[t]}))
    df = pd.concat(rows, ignore_index=True)
    params = dict(time_idx="time_idx", target="y", group_ids=["series"], max_encoder_length=ENC_LEN, max_prediction_length=H,
        static_categoricals=["store_id", "fam", "store_type", "store_cluster"], time_varying_known_categoricals=["dow"],
        time_varying_known_reals=["time_idx", "payday", "hol_nat", "hol_reg", "hol_loc", "promo", "closed",
                                  "quake", "dom_sin", "dom_cos", "m_sin", "m_cos"],
        time_varying_unknown_reals=["y", "trans", "oil"],
        target_normalizer=GroupNormalizer(groups=["series"]), add_relative_time_idx=True, add_encoder_length=True)
    training = TimeSeriesDataSet(df[df.time_idx < VAL_O[3]], **params)
    val_ds = TimeSeriesDataSet.from_dataset(training, df[df.time_idx < VAL_O[3] + H], predict=True, stop_randomization=True)
    test_ds = TimeSeriesDataSet.from_dataset(training, df[df.time_idx < TEST_O + H], predict=True, stop_randomization=True)
    nw = 2 if IN_COLAB else 0
    tdl = training.to_dataloader(train=True, batch_size=256, num_workers=nw)
    vdl = val_ds.to_dataloader(train=False, batch_size=256, num_workers=nw)
    xdl = test_ds.to_dataloader(train=False, batch_size=256, num_workers=nw)
    tft = TemporalFusionTransformer.from_dataset(training, learning_rate=1e-3, hidden_size=32, attention_head_size=2,
        dropout=0.2, hidden_continuous_size=16, loss=QuantileLoss(QUANTILES), log_interval=0)
    trainer = pl.Trainer(max_epochs=1 if SMOKE else 15, accelerator="auto", gradient_clip_val=0.1, limit_train_batches=5 if SMOKE else 200,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=3)], enable_model_summary=False,
                         enable_progress_bar=False, logger=False, enable_checkpointing=False)
    t0 = time.time(); trainer.fit(tft, train_dataloaders=tdl, val_dataloaders=vdl)
    def tft_pred(dl):
        out = tft.predict(dl, mode="quantiles", return_index=True)
        q = out.output if hasattr(out, "output") else out[0]
        idx = out.index if hasattr(out, "index") else out[-1]
        q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        P = np.zeros((S, H, 3), np.float32)
        for k, sid in enumerate(idx["series"].astype(int).values): P[sid] = q[k]
        return np.sort(P, -1)
    res = dict(preds={VAL_O[3]: tft_pred(vdl), TEST_O: tft_pred(xdl)}, minutes=(time.time() - t0) / 60,
               epochs=trainer.current_epoch)
    try:
        raw = tft.predict(xdl, mode="raw", return_x=True)
        interp = tft.interpret_output(raw.output, reduction="sum")
        res["encoder_importance"] = pd.Series(interp["encoder_variables"].cpu().numpy(), index=tft.encoder_variables)
        res["decoder_importance"] = pd.Series(interp["decoder_variables"].cpu().numpy(), index=tft.decoder_variables)
    except Exception as e:
        print("TFT importance not available:", e)
    return res

TFT = None
if RUN_TFT:
    try:
        TFT = cached("tft", run_tft)
        PREDS["TFT"] = TFT["preds"]
        tft_rows = [dict(model="TFT", split="test", family=FAMILIES[f], **metrics({TEST_O: TFT["preds"][TEST_O]}, f)) for f in range(2)]
        tft_rows += [dict(model="TFT", split="validation (window 4)", family=FAMILIES[f], **metrics({VAL_O[3]: TFT["preds"][VAL_O[3]]}, f)) for f in range(2)]
        others = [dict(model=n, split="validation (window 4)", family=FAMILIES[f], **metrics({VAL_O[3]: PREDS[n][VAL_O[3]]}, f))
                  for n in ["LSTM ensemble", "LightGBM ensemble"] for f in range(2)]
        TFT_RES = pd.DataFrame(tft_rows + others); TFT_RES.to_csv(f"{OUT_DIR}/results_tft.csv", index=False); display(TFT_RES.round(4))
        print(f"TFT: {TFT['minutes']:.1f} min, {TFT['epochs']} epochs")
        for k in ("encoder_importance", "decoder_importance"):
            if k in TFT:
                vi = (100 * TFT[k] / TFT[k].sum()).sort_values(ascending=False).round(1)
                vi.to_csv(f"{OUT_DIR}/tft_{k}.csv"); display(vi.head(10))
    except Exception as e:
        print("TFT could not run:", repr(e))

# %% [markdown]
# ## 15. Forecast of Aug 16–31, 2017 and export for the Streamlit app

# %%
FUT = apply_conformal(FUTURE_LSTM, CONF["LSTM ensemble"])
app_rows = []
for s, (fam, st) in enumerate(series):
    f_i = FAMILIES.index(fam)
    for t in range(TEST_O - ENC_LEN, TEST_O):
        app_rows.append(dict(family=fam, store=st, date=dates[t], kind="history", sales=SALES[s, t], open=OPEN[s, t]))
    for kind, o, P in [("backtest", TEST_O, PREDS[MAIN][TEST_O]), ("backtest_lgb", TEST_O, PREDS["LightGBM ensemble + conformal"][TEST_O]),
                       ("future", FUT_O, FUT)]:
        q = np.expm1(np.sort(P[s], -1))
        for k in range(H):
            t = o + k
            app_rows.append(dict(family=fam, store=st, date=dates[t], kind=kind,
                                 sales=SALES[s, t] if kind != "future" else np.nan, open=OPEN[s, t] if kind != "future" else 1.0,
                                 q10=q[k, 0], q50=q[k, 1], q90=q[k, 2], promo=PROMO_RAW[s, t], payday=PAYDAY[t],
                                 holiday=float(HOL[s, t].max())))
APP = pd.DataFrame(app_rows); APP["date"] = APP.date.dt.date
APP.round(3).to_csv(f"{APP_DIR}/forecasts.csv", index=False)

IG_FUT = integrated_gradients(load_model("lstm", BEST_CFG, RUNS["LSTM"][SEED]["state_future"]), FUT_O)
att = []
for kind, igd, o in [("backtest", IG_DAY, TEST_O), ("future", IG_FUT, FUT_O)]:
    scale = make_batch(*all_series(o))[-1].cpu().numpy()           # attributions back to the log(1+x) scale
    for s, (fam, st) in enumerate(series):
        for g in GROUPS:
            for k in range(H):
                att.append(dict(family=fam, store=st, kind=kind, date=dates[o + k].date(), factor=g,
                                attribution=float(igd[g][s, k] * scale[s])))
pd.DataFrame(att).round(5).to_csv(f"{APP_DIR}/attributions.csv", index=False)
stores.to_csv(f"{APP_DIR}/stores.csv", index=False)
tft_test = [dict(model="TFT", split="test", family=FAMILIES[f], **metrics({TEST_O: PREDS["TFT"][TEST_O]}, f))
            for f in range(2)] if "TFT" in PREDS else []
pd.concat([RES[RES.split == "test"], pd.DataFrame(tft_test)]).round(4).to_csv(f"{APP_DIR}/metrics_test.csv", index=False)
json.dump(dict(main_model=MAIN, config=BEST_CFG, seeds=SEEDS, dairy_life=DAIRY_LIFE, test_origin=TEST_ORIGIN,
               future_origin=FUTURE_ORIGIN, conformal={k: {FAMILIES[f]: v for f, v in d.items()} for k, d in CONF.items()}),
          open(f"{APP_DIR}/meta.json", "w"), indent=2)
torch.save({"config": BEST_CFG, "card": CARD, "state_dict": RUNS["LSTM"][SEED]["state_future"]}, f"{OUT_DIR}/lstm_seed42_future.pt")
print("app data written to", APP_DIR)

# %%
# Package tables, figures and app data (without the cache) in results_final.zip and download it
import shutil
ZIP = "/content/results_final.zip" if IN_COLAB else f"{OUT_DIR}/../results_final.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, fnames in os.walk(OUT_DIR):
        if "cache" in dirpath: continue
        for fn in fnames:
            if fn.endswith(".zip"): continue
            full = os.path.join(dirpath, fn); zf.write(full, os.path.join("results", os.path.relpath(full, OUT_DIR)))
print("wrote", ZIP)
if IN_COLAB:
    shutil.copy(ZIP, OUT_DIR)                     # a copy also stays in Google Drive
    from google.colab import files
    files.download(ZIP)
