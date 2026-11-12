"""Builds the final IEEE-style report (paper/FreshCast_Final_Report.docx) from the tables and figures in results/.

    python paper/build_paper.py [results_dir]

Every number in the tables and most numbers in the text are read from the CSV files written by
src/final_experiments.py, so the report can be rebuilt after re-running the experiments.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = ROOT / "paper" / "FreshCast_Final_Report.docx"

GITHUB_URL = "https://github.com/johansoriano63-ux/bread-dairy-forecasting"
DRIVE_URL = "https://drive.google.com/DRIVE_LINK"
VIDEO_URL = "https://youtu.be/VIDEO_LINK"
AUTHORS = [("Stephany Trujillo", "stephany.trujillo@estudiantesunibague.edu.co"),
           ("Johan Soriano", "johan.soreano@estudiantesunibague.edu.co"),
           ("Cristian Cardona", "cristian.cardona@estudiantesunibague.edu.co"),
           ("Alexandra La Cruz", "alexandra.lacruz@unibague.edu.co")]
FAM = ["BREAD/BAKERY", "DAIRY"]

# ----------------------------------------------------------------------------- data
rd = lambda f: pd.read_csv(RES_DIR / f)
R = rd("results_final.csv")
SEEDR = rd("results_by_seed.csv")
TUNE = rd("tuning_lstm.csv")
NV = rd("newsvendor_final.csv")
INV = rd("inventory_policies.csv")
AL = rd("anomaly_alerts_final.csv")
EXPL = rd("explanations_validation.csv")
EDA = rd("eda_summary.csv")
CONF = rd("conformal_offsets.csv")
TFT = rd("results_tft.csv") if (RES_DIR / "results_tft.csv").exists() else None
TFT_ENC = pd.read_csv(RES_DIR / "tft_encoder_importance.csv", index_col=0).iloc[:, 0] if (RES_DIR / "tft_encoder_importance.csv").exists() else None
N_SEEDS = SEEDR.seed.nunique()
MAIN = "LSTM ensemble + conformal"


def m(model, fam, metric="RMSLE", split="test"):
    r = R[(R.model == model) & (R.family == FAM[fam]) & (R.split == split)]
    return float(r[metric].iloc[0])


def ms(model, fam, metric="RMSLE", split="test"):
    v = SEEDR[(SEEDR.model == model) & (SEEDR.family == FAM[fam]) & (SEEDR.split == split)][metric]
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def red(a, b):  # % reduction of a with respect to b
    return 100 * (b - a) / b


def cmp_word(a, b, tol=1.0):
    d = red(a, b)
    if abs(d) < tol: return "is on par with"
    return "outperforms" if d > 0 else "is outperformed by"


best_cfg = TUNE.loc[TUNE.val_pinball.idxmin()]
fmt = lambda x, k=4: f"{x:.{k}f}"

# ----------------------------------------------------------------------------- architecture figure
def architecture_figure(path):
    fig, ax = plt.subplots(figsize=(3.5, 4.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")
    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25", fc=fc, ec="#555", lw=.8))
        ax.text(x + w/2, y + h*0.66, title, ha="center", va="center", fontsize=6.6, weight="bold")
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=5.0, color="#333", linespacing=1.15)
    def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=.7, color="#444"))
    box(0.3, 11.0, 9.4, 1.2, "DATA: BREAD/BAKERY and DAIRY", "54 stores × 2 families = 108 daily series · promotions, holidays,\noil, transactions, store attributes", "#dde8f7")
    box(0.3, 9.3, 9.4, 1.2, "PREPROCESSING", "ln(1+x), complete calendar, closed-day mask, per-window scaling,\nwindows of 56 days of history + 16 days of horizon", "#dff0dc")
    box(0.3, 7.0, 4.4, 1.8, "ENCODER LSTM", "56-day history\n+ past covariates\n+ store/family embeddings", "#fdf0d5")
    box(5.3, 7.0, 4.4, 1.8, "DECODER LSTM", "known future covariates:\npromotions, holidays,\ncalendar (16 days)", "#fdf0d5")
    box(0.3, 5.3, 9.4, 1.2, "NON-CROSSING QUANTILE HEAD", "median + softplus offsets → τ = 0.1 · 0.5 · 0.9, pinball loss,\nseed ensemble + conformal recalibration", "#f8dcdc")
    for i, (t, s) in enumerate([("ORDER", "newsvendor τ*\n/ order-up-to"), ("ALERTS", "sales outside\nthe interval"), ("EXPLANATION", "integrated\ngradients")]):
        box(0.3 + i*3.2, 2.9, 2.9, 1.8, t, s, "#e7e1f3")
    box(0.3, 0.4, 9.4, 1.4, "WEB APPLICATION (Streamlit)", "forecast and interval · recommended order · factors · alerts", "#ececec")
    arrow(5, 11.0, 5, 10.55); arrow(2.5, 9.3, 2.5, 8.85); arrow(7.5, 9.3, 7.5, 8.85); arrow(4.75, 7.9, 5.25, 7.9)
    ax.text(5.0, 8.1, "state", ha="center", fontsize=4.6, color="#555")
    arrow(2.5, 7.0, 2.5, 6.55); arrow(7.5, 7.0, 7.5, 6.55)
    for i in range(3): arrow(1.75 + i*3.2, 5.3, 1.75 + i*3.2, 4.75); arrow(1.75 + i*3.2, 2.9, 1.75 + i*3.2, 1.85)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


architecture_figure(RES_DIR / "fig_architecture.png")

# ----------------------------------------------------------------------------- docx helpers
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
pf = st.paragraph_format; pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = 1.0

sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin, sec.left_margin, sec.right_margin = Inches(.75), Inches(1), Inches(.625), Inches(.625)
sec.header_distance = Inches(.4)
COL_W = Inches(3.5)


def set_columns(section, n):
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols"); section._sectPr.append(cols)
    cols.set(qn("w:num"), str(n)); cols.set(qn("w:space"), "360")


def para(text="", size=10, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, before=0, after=0, sc=False):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.first_line_indent = Inches(.14) if indent else None
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
    if text: add_runs(p, text, size, bold, italic, sc)
    return p


def add_runs(p, text, size=10, bold=False, italic=False, sc=False):
    """Text with *italic* segments marked by asterisks."""
    for k, part in enumerate(text.split("*")):
        if not part: continue
        r = p.add_run(part); r.font.size = Pt(size); r.bold = bold; r.italic = italic ^ (k % 2 == 1); r.font.small_caps = sc
    return p


SEC_N = [0]; SUB_N = [0]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


def heading(text, numbered=True):
    p = para(align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=8, after=4)
    p.paragraph_format.keep_with_next = True
    label = text
    if numbered:
        label = f"{ROMAN[SEC_N[0]]}. {text}"; SEC_N[0] += 1; SUB_N[0] = 0
    r = p.add_run(label); r.font.small_caps = True; r.font.size = Pt(10)


def subheading(text):
    p = para(align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, before=4, after=2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{chr(65 + SUB_N[0])}. {text}"); r.italic = True; SUB_N[0] += 1


EQ_N = [0]


def equation(text):
    EQ_N[0] += 1
    p = para(align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, before=3, after=3)
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(Inches(1.75), WD_TAB_ALIGNMENT.CENTER); ts.add_tab_stop(Inches(3.5), WD_TAB_ALIGNMENT.RIGHT)
    add_runs(p, f"\t{text}\t({EQ_N[0]})", 10, italic=False)
    return EQ_N[0]


FIG_N = [0]


def figure(path, caption, width=3.4):
    FIG_N[0] += 1
    p = para(align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    c = para(indent=False, before=2, after=6)
    add_runs(c, f"Fig. {FIG_N[0]}. {caption}", 8)
    return FIG_N[0]


TAB_N = [0]


def table(df, title, note=None, widths=None, bold_min=None):
    """df: DataFrame of strings. bold_min: list of (column, group_column) to bold the best (lowest) value per group."""
    TAB_N[0] += 1
    c = para(f"TABLE {ROMAN[TAB_N[0] - 1]}", 8, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=6)
    c.paragraph_format.keep_with_next = True
    c2 = para(align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=2); c2.paragraph_format.keep_with_next = True
    r = c2.add_run(title); r.font.size = Pt(8); r.font.small_caps = True
    t = doc.add_table(rows=1, cols=len(df.columns)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    bold_cells = set()
    for col, grp in (bold_min or []):
        vals = pd.to_numeric(df[col].str.replace("±.*", "", regex=True), errors="coerce")
        for _, idx in df.groupby(grp).groups.items():
            v = vals.loc[idx]
            if v.notna().any(): bold_cells.add((v.idxmin(), col))
    for j, h in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = ""
        rr = cell.paragraphs[0].add_run(str(h)); rr.bold = True; rr.font.size = Pt(7.5)
    for i, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            rr = cells[j].paragraphs[0].add_run(str(v)); rr.font.size = Pt(7.5)
            rr.bold = (i, df.columns[j]) in bold_cells
            if j > 0: cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    tbl = t._tbl; tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "8"); e.set(qn("w:color"), "000000"); borders.append(e)
    tblPr.append(borders)
    for cell in t.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders"); e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000"); b.append(e); tcPr.append(b)
    for row in t.rows:
        for j, cell in enumerate(row.cells):
            cell.paragraphs[0].paragraph_format.first_line_indent = None
            if widths: cell.width = Inches(widths[j])
            mar = OxmlElement("w:tcMar")
            for side in ("left", "right"):
                e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "45"); e.set(qn("w:type"), "dxa"); mar.append(e)
            cell._tc.get_or_add_tcPr().append(mar)
    if note:
        add_runs(para(indent=False, before=1, after=6), note, 7)
    else:
        para(after=4)
    return TAB_N[0]


def header_footer(section):
    hp = section.header.paragraphs[0]; hp.text = ""
    hp.paragraph_format.tab_stops.add_tab_stop(Inches(7.25), WD_TAB_ALIGNMENT.RIGHT)
    r = hp.add_run("INTELLIGENT SYSTEMS – UNIVERSIDAD DE IBAGUÉ – FINAL PROJECT\t"); r.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    rr = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "16"); rpr.append(sz)
    tx = OxmlElement("w:t"); tx.text = "1"; rr.append(rpr); rr.append(tx); fld.append(rr); hp._p.append(fld)


header_footer(sec)

# ----------------------------------------------------------------------------- numbers used in the text
L, G, LG = "LSTM ensemble + conformal", "GRU ensemble + conformal", "LightGBM ensemble + conformal"
rm = {f: {k: m(k, f) for k in ["Seasonal naive", "ETS", "LightGBM ensemble", "LSTM ensemble", "GRU ensemble"]} for f in range(2)}
cov = {f: {k: m(k, f, "Coverage_pct") for k in ["LSTM ensemble", L, "LightGBM ensemble", LG, "ETS", "ETS + conformal"]} for f in range(2)}
cmp_seed = SEEDR[SEEDR.split == "test"].pivot_table(index=["family", "seed"], columns="model", values="RMSLE")
wins = {f: int((cmp_seed.loc[FAM[f]]["LSTM"] < cmp_seed.loc[FAM[f]]["LightGBM"]).sum()) for f in range(2)}
nv = lambda model, fam, ratio: float(NV[(NV.model == model) & (NV.family == FAM[fam]) & (NV.cost_ratio == ratio)].saving_pct.iloc[0])
inv = lambda fam, life, ratio, pol: INV[(INV.family == FAM[fam]) & (INV.shelf_life == life) & (INV.cost_ratio == ratio) & INV.policy.str.startswith(pol)].iloc[0]
DL = int(INV[INV.family == "DAIRY"].shelf_life.max())
al = lambda model, fam, col: AL[(AL.model == model) & (AL.family == FAM[fam])][col].iloc[0]
ex = lambda fam: EXPL[EXPL.family == FAM[fam]].set_index("factor")
top_factor = {f: ex(f).IG_share_pct.idxmax() for f in range(2)}
abl_top = {f: ex(f).delta_RMSLE_pct.idxmax() for f in range(2)} if "delta_RMSLE_pct" in EXPL else None
corr = {f: float(np.corrcoef(ex(f).IG_share_pct, ex(f).occlusion_share_pct)[0, 1]) for f in range(2)}

# ----------------------------------------------------------------------------- title block (one column)
p = para(align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=10)
r = p.add_run("FreshCast: Probabilistic Forecasting and Ordering of Bakery and Dairy Products with a Recurrent Neural Network Agent")
r.font.size = Pt(22)
p = para(align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
r = p.add_run(", ".join(a for a, _ in AUTHORS[:-1]) + f", and {AUTHORS[-1][0]}"); r.font.size = Pt(11)
para("Faculty of Engineering, Universidad de Ibagué, Ibagué, Colombia", 9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
para(" · ".join(e for _, e in AUTHORS), 8, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=12)

new = doc.add_section(WD_SECTION.CONTINUOUS); set_columns(new, 2)

d_naive = red(rm[1]["LSTM ensemble"], rm[1]["Seasonal naive"]); b_naive = red(rm[0]["LSTM ensemble"], rm[0]["Seasonal naive"])
abstract = (
    "Bread and dairy are perishable products whose daily replenishment depends on forecasting demand well: what is "
    "not sold is lost and what is missing is a lost sale. We present FreshCast, an intelligent agent for the "
    "BREAD/BAKERY and DAIRY families of the 54 stores of the Ecuadorian chain Corporación Favorita (Kaggle Store "
    "Sales dataset, 108 daily series). Its core is an LSTM encoder–decoder with store and family embeddings and a "
    "non-crossing quantile output, trained as a single global model, whose 10–90 % intervals are recalibrated with "
    "conformal prediction. The quantiles are turned into a recommended order (newsvendor rule for bread, order-up-to "
    "rule with carried stock for dairy), into anomaly alerts and into factor-level explanations computed with "
    "integrated gradients, and everything is delivered in a Streamlit application. Under one protocol, with "
    f"{N_SEEDS} seeds, four rolling validation windows and a 16-day holdout test, the LSTM ensemble reduces RMSLE "
    f"by {b_naive:.0f} % (bread) and {d_naive:.0f} % (dairy) with respect to a seasonal naive model, reaching "
    f"{rm[0]['LSTM ensemble']:.3f} and {rm[1]['LSTM ensemble']:.3f} against {rm[0]['LightGBM ensemble']:.3f} and "
    f"{rm[1]['LightGBM ensemble']:.3f} for quantile LightGBM, and recalibration brings interval coverage to "
    f"{cov[0][L]:.0f} % and {cov[1][L]:.0f} % (target 80 %). The order rules lower the simulated cost of shortages "
    f"and waste, by up to {max(nv(L, 0, '2:1'), nv(L, 0, '4:1')):.0f} % for bread, and the carried-stock rule cuts "
    "dairy waste with respect to a daily newsvendor. The study also shows where the network is not better than a "
    "well-built gradient boosting model.")
p = para(indent=False, after=4); r = p.add_run("Abstract—"); r.bold = True; r.italic = True; r.font.size = Pt(9)
add_runs(p, abstract, 9, bold=True)
p = para(indent=False, after=4); r = p.add_run("Index Terms—"); r.bold = True; r.italic = True; r.font.size = Pt(9)
add_runs(p, "Conformal prediction, dairy, bakery, intelligent agents, LSTM, newsvendor problem, perishable inventory, "
            "probabilistic forecasting, quantile regression, retail, time series.", 9, bold=True)
p = para(indent=False, after=6)
add_runs(p, f"Code: {GITHUB_URL} · Dataset: {DRIVE_URL} · Video: {VIDEO_URL}", 8, italic=True)

# ----------------------------------------------------------------------------- I. Introduction
heading("Introduction")
para("In supermarkets, perishable products concentrate a large share of food waste and lost sales [1]. Bread is an "
     "extreme case: its shelf life is hours or a few days, it is restocked every day and what is not sold is thrown "
     "away. Dairy shares these traits with a longer life of a few days to a few weeks. In both families the daily "
     "forecast translates almost directly into the quantity that must be available, so an error in either direction "
     "has an immediate cost [1], [2].")
para("Recurrent and attention-based neural networks trained over many related series have reached state-of-the-art "
     "results in large-scale demand forecasting [3], [4], but their advantage over statistical methods and gradient "
     "boosting is not universal [5]–[7]. Moreover, a store manager does not need a point forecast but a decision: how "
     "much to order, when to worry, and why. This work addresses three questions for two perishable families of a Latin "
     "American chain: 1) can a recurrent network trained as a global model forecast their daily demand better than "
     "simple and gradient-boosting alternatives, measured by accuracy and by the quality of its intervals; 2) can its "
     "quantiles be turned into ordering decisions that respect the different shelf lives of bread and dairy; and 3) can "
     "its forecasts be explained and monitored in a way that store staff can use?")
para("The project was developed in four phases. A first proposal forecasted all 33 product families with gradient "
     "boosting; following the instructor's feedback, the second phase narrowed the scope to bread and dairy, made a "
     "recurrent network the core model and reviewed the literature; the third phase implemented the network and the "
     "baselines and reported first results; and this final phase completes the study with hyperparameter tuning, "
     f"several seeds, conformal recalibration, a multi-day ordering rule for dairy, the validation of the explanations, "
     "the Temporal Fusion Transformer (TFT) and the web application. The contributions are:")
for b in ["an LSTM encoder–decoder with embeddings and a non-crossing quantile head, compared under one protocol with "
          "five alternatives and with its run-to-run variability measured;",
          "a conformal recalibration step that makes the 10–90 % intervals usable for ordering and alerting;",
          "a decision layer that uses the newsvendor quantile for bread and an order-up-to rule with perishable "
          "carried stock for dairy;",
          "explanations by factor that are checked against occlusion and retraining ablations; and",
          "an open-source agent, from data to a Streamlit application, reproducible in Google Colab."]:
    p = para(indent=False); p.paragraph_format.left_indent = Inches(.15); add_runs(p, "• " + b)
para("Following the PEAS framework [8], the performance measure of the agent is the accuracy and probabilistic "
     "quality of the forecast (RMSLE, WAPE, pinball loss and coverage), the simulated cost of its orders and the rate "
     "of alerts; the environment is the 54 stores and their demand for bread and dairy; the actuators are the "
     "application screens that show forecasts, orders, explanations and alerts; and the sensors are the sales history "
     "and the context data. The environment is partially observable, stochastic, sequential and dynamic, and the "
     "system is a model-based learning agent: the network learns how demand evolves and the decision layer uses that "
     "model to act [8]. Section II reviews related work, Section III describes the data and methods, Section IV "
     "presents the results, Section V discusses them and Section VI concludes.")

# ----------------------------------------------------------------------------- II. Related work
heading("Related Work")
subheading("Demand Forecasting for Perishables")
para("Daily demand for fresh products shows strong weekly seasonality, sensitivity to holidays and promotions and a "
     "high cost of error in both directions. Taylor forecasted daily supermarket sales with exponentially weighted "
     "quantile regression, showing the value of estimating quantiles rather than only the mean [9]. Huber and "
     "Stuckenschmidt found that special calendar days concentrate the largest errors in a retail chain and that machine "
     "learning models that represent them explicitly beat traditional methods [2]. Fildes et al. list fresh products "
     "and promotions among the main open challenges of retail forecasting [1].")
subheading("Neural Networks for Time Series")
para("Recurrent networks keep an internal state that summarizes the history; LSTM [10] and GRU [11] cells use gates "
     "that mitigate vanishing gradients and capture long dependencies such as weekly seasonality. The "
     "sequence-to-sequence architecture [12] was adapted to multi-horizon forecasting by MQ-RNN, which decodes known "
     "future covariates and outputs several quantiles at once [13], and DeepAR trained one global recurrent network "
     "that outputs full distributions [3]. Hewamalage et al. showed that recurrent networks are competitive when they "
     "are trained as global models over many related series with proper per-series normalization [6], and entity "
     "embeddings let a network learn from identifiers such as the store [14]. Among more recent architectures, N-BEATS "
     "[15] and the TFT, which combines LSTM layers, attention and variable-selection networks [4], reached "
     "state-of-the-art accuracy, although a simple linear model beat several transformers on long horizons [16].")
subheading("Comparative Evidence in Retail")
para("The evidence is mixed. Machine learning models, including recurrent networks, were less accurate than "
     "statistical methods on short monthly series [17], but with abundant daily related series the picture changes: a "
     "global LSTM beat the baselines on e-commerce demand [18], and in the M5 competition on Walmart sales global "
     "machine learning methods beat the statistical references, with LightGBM [19] behind most top solutions [5], [20]. "
     "Kaggle forecasting competitions, including one with Corporación Favorita data, were won by combinations of "
     "gradient boosting and neural networks [21]. Neural networks must therefore be compared explicitly with boosting "
     "and with simple baselines in each domain.")
subheading("Probabilistic Forecasting, Calibration and Ordering")
para("For perishables, the classic decision problem is the newsvendor: the optimal stock is the quantile of demand "
     "given by the ratio between the cost of a shortage and the sum of the shortage and leftover costs [22]. Quantile "
     "regression [23] and its pinball loss let networks predict quantiles directly [4], [13], and the pinball loss is a "
     "proper scoring rule [24]. Huber et al. showed with bakery data that data-driven newsvendor approaches reduce "
     "ordering cost [25]. When stock lasts more than one period, perishable inventory theory recommends order-up-to "
     "policies that account for the age of the stock on hand [26]. Quantile models are often miscalibrated out of "
     "sample; conformalized quantile regression corrects their intervals with a held-out calibration set and gives "
     "finite-sample coverage guarantees under exchangeability [27], [28].")
subheading("Interpretability")
para("Integrated gradients attribute a prediction to each input by integrating gradients along a path from a "
     "baseline input and satisfy desirable axioms [29]; the TFT offers built-in variable-selection weights [4]. "
     "Explainability for time series is less mature than in other domains [30], so attributions should be validated, "
     "for example with occlusion or retraining ablations. The second delivery of this project identified four gaps "
     "that guide this work: few studies focus on specific perishable families, many report only point errors without "
     "the resulting decision, the advantage of networks over boosting is disputed, and there is little evidence and "
     "few user-oriented explanations with Latin American data.")

# ----------------------------------------------------------------------------- III. Methods
heading("Methods")
subheading("Dataset")
para("We use the public Store Sales – Time Series Forecasting dataset of Kaggle [31]. It contains daily sales from "
     "January 1, 2013 to August 15, 2017 for 54 stores and 33 product families, together with the number of items on "
     "promotion, national, regional and local holidays, the daily oil price (Ecuador's economy depends on it), "
     "daily transactions per store and store attributes (city, state, type and cluster). A test file adds the "
     "promotions of August 16–31, 2017, whose sales are not public. Filtering BREAD/BAKERY and DAIRY yields 108 daily "
     "series of 1,688 days. Table I lists the variables and how the model uses them. The files are shared in Google "
     f"Drive ({DRIVE_URL}) and the code in GitHub ({GITHUB_URL}).")
vars_df = pd.DataFrame([
    ["sales", "Target, ln(1+x); past only"], ["onpromotion", "Items on promotion; known in advance"],
    ["holidays_events", "National, regional (by state), local (by city); transfers applied"],
    ["Calendar", "Day of week, day of month, month, payday (15th and last day)"],
    ["Earthquake", "Indicator of the 4 weeks after April 16, 2016"],
    ["Closed day", "Store sold nothing in any family (e.g., Dec. 25); masked"],
    ["oil, transactions", "Past covariates only (not known in advance)"],
    ["store, type, cluster, family", "Learned embeddings"]], columns=["Variable", "Use in the model"])
t_vars = table(vars_df, "Variables Used by the Model", widths=[1.2, 2.3])
para("Every series is reindexed to the complete calendar. Days on which a store sold nothing in any family are "
     "flagged as closed and excluded from the loss and the metrics, as are the days before the first sale of each "
     "series, because 16 series belong to stores that opened after 2013. Transferred holidays are moved to the day "
     "actually observed and work days are ignored, and missing oil prices are interpolated linearly.")
para(f"Exploration (Table {ROMAN[t_vars]} and Fig. 1) shows that weekly seasonality dominates both families: sales "
     f"peak on Sunday ({EDA.dow_Sun[0]:.2f} and {EDA.dow_Sun[1]:.2f} times the series mean) and reach their minimum "
     "on Thursday, which supports an encoder of eight full weeks. Dairy reacts much more than bread to promotions and "
     "national holidays, and the payday effect is small (2–3 %); promotion ratios are associations rather than "
     "causal effects, because promotions are not assigned at random.")
eda_df = pd.DataFrame([["Weekend vs. weekday", *EDA.weekend_vs_weekday], ["Sunday (highest day)", *EDA.dow_Sun],
                       ["Thursday (lowest day)", *EDA.dow_Thu], ["Payday vs. other days", *EDA.payday_vs_other],
                       ["National holiday vs. other", *EDA.national_holiday_vs_other], ["Promotion vs. none", *EDA.promo_vs_no_promo]],
                      columns=["Pattern", "Bread", "Dairy"])
eda_df[["Bread", "Dairy"]] = eda_df[["Bread", "Dairy"]].map(lambda v: f"{v:.2f}")
t_eda = table(eda_df, "Relative Sales by Condition (Open Days, 2013–2017)", widths=[1.9, .8, .8],
              note="Relative sales: sales divided by the mean of the series over open days.")
f_dow = figure(RES_DIR / "fig_eda_dow.png", "Relative sales by day of week for BREAD/BAKERY and DAIRY (sales divided by the series mean, open days, 2013–2017).")

subheading("Agent Architecture")
f_arch = figure(RES_DIR / "fig_architecture.png", "Architecture of FreshCast: the LSTM encoder–decoder produces "
                "recalibrated quantiles that feed the order recommendation, the anomaly alerts and the explanations "
                "shown in the web application.", 3.1)
para(f"Fig. {f_arch} shows the agent. The data pipeline builds windows of 56 days of history and 16 days of horizon "
     "(the horizon of the Kaggle competition). The neural core forecasts three quantiles per day; a seed ensemble and a "
     "conformal step calibrate them; and three components act on the quantiles: the order rule, the alert rule and "
     "the explanation, which the application presents to the user.")

subheading("LSTM Encoder–Decoder with Quantile Output")
para("The encoder, a two-layer LSTM [10], reads 56 days of scaled log-sales, transactions, oil price and the known "
     "covariates. Its final state initializes the decoder, another two-layer LSTM that receives the known covariates "
     "of the next 16 days (promotions, holidays, calendar, closed-day and earthquake indicators). Store, store type, "
     "cluster and family enter through learned embeddings [14] concatenated to every time step, so one global model "
     "learns the patterns shared by both families and their differences. Each window is divided by the mean log-sales "
     "of its encoder, because bread and dairy in the same store have very different levels [6]. For each day a linear "
     "layer outputs the median and two offsets that pass through a softplus function:")
equation("ŷ0.1 = ŷ0.5 − softplus(g1(ht)),   ŷ0.9 = ŷ0.5 + softplus(g2(ht))")
para("where ŷ0.5 = f(ht) is the median, ht the decoder state and f, g1, g2 linear maps, so the quantiles never cross. "
     "The network is trained with the pinball loss [23], averaged over quantiles, days and series and ignoring closed days:")
e_pin = equation("Lτ(y, ŷτ) = max{ τ(y − ŷτ), (τ − 1)(y − ŷτ) }")
para("Training uses Adam [32], dropout [33], gradient clipping and early stopping on the validation windows. The "
     "hidden size, dropout and learning rate were tuned on the validation windows over the five configurations of "
     f"Table {ROMAN[TAB_N[0]]}; the selected configuration was then trained with {N_SEEDS} seeds and the seed forecasts "
     "were averaged (seed ensemble). A GRU variant [11] with the same structure was trained in the same way.")
tune_df = TUNE.copy()
tune_df = pd.DataFrame({"Hidden": tune_df.hidden.astype(int), "Dropout": tune_df.dropout, "lr": tune_df.lr.map(lambda v: f"{v:.0e}"),
                        "Best epoch": tune_df.best_epoch.astype(int), "Val. pinball": tune_df.val_pinball.map(lambda v: f"{v:.5f}"),
                        "RMSLE bread": tune_df.val_RMSLE_Bread.map(fmt), "RMSLE dairy": tune_df.val_RMSLE_Dairy.map(fmt)}).astype(str)
tune_df["g"] = 0
t_tune = table(tune_df.drop(columns="g"), "Hyperparameter Tuning of the LSTM (Validation Windows, Seed 42)",
               note=("Fixed: 2 layers, embeddings 8/3/4/2 (store/type/cluster/family), Adam, gradient clipping 1.0, batch "
                     "256 windows, 300 batches per epoch, at most 40 epochs, patience 6. Selected: hidden "
                     f"{int(best_cfg.hidden)}, dropout {best_cfg.dropout}, learning rate {best_cfg.lr:.0e}."),
               widths=[.45, .5, .45, .5, .6, .5, .5])

subheading("Comparison Models")
para("Five models are compared with the LSTM under the same protocol. The seasonal naive model repeats the sales of the "
     "same weekday of the previous week, with quantiles from its empirical residuals over the last year. ETS [34] uses "
     "additive weekly seasonality and is refitted for each series at every origin on the last 364 days, with a normal "
     "80 % interval from its residuals. LightGBM [19] is a global direct multi-horizon model with one row per series, "
     "origin and horizon day, trained with the quantile objective (one model per quantile) on origins every four days "
     "since 2015; its features (lags, rolling means, same-weekday means, promotions, calendar and store attributes) "
     f"use only the history before the origin, and it is also run with {N_SEEDS} seeds and ensembled. The GRU variant "
     "shares the LSTM architecture. The TFT [4] is implemented with PyTorch Forecasting with hidden size 32, two "
     "attention heads and the same covariates; because of its cost it is trained once, with targets up to July 14, "
     "early-stopped on the fourth validation window and then used to forecast the test window.")

subheading("Evaluation Protocol and Metrics")
split_df = pd.DataFrame([["Training", "First sale – May 27, 2017 (July 30 for test models)"],
                         ["Validation 1–4", "May 28 – July 30, 2017 (four 16-day windows)"],
                         ["Test (holdout)", "July 31 – August 15, 2017"],
                         ["Forecast (app)", "August 16 – 31, 2017 (no labels)"]], columns=["Split", "Dates"])
t_split = table(split_df, "Data Split", widths=[1.1, 2.4])
para(f"Table {ROMAN[t_split - 1]} shows the split. The local baselines are refitted at every origin. The global models "
     "are trained with targets up to May 27, 2017, early-stopped (or tuned) on the four validation windows and each "
     "window is forecast from the real history up to its first day (rolling origin [35]). For the test window they are "
     "retrained from scratch with targets up to July 30, 2017 for the number of epochs or trees selected in "
     "validation, and the test window is used only once. For the application the LSTM is retrained once more with "
     "targets up to August 15 to forecast August 16–31. The median is evaluated with RMSLE, the official metric of the "
     "competition [31], WAPE and MAE; the intervals with the pinball loss on the log scale and the empirical coverage "
     "of the 10–90 % interval, whose target is 80 % [20], [24]. All metrics are reported by family on open days:")
equation("RMSLE = √[ (1/n) Σi ( ln(1 + ŷi) − ln(1 + yi) )² ]")

subheading("Conformal Recalibration")
para("Quantile models trained on past data are usually miscalibrated on new periods. Following conformalized quantile "
     "regression [27], [28], we compute on the four validation windows the scores s_lo = ŷ0.1 − y and s_hi = y − ŷ0.9 "
     "on the log scale, take their empirical 90 % quantiles (with the finite-sample correction (1 + 1/n)) per family, "
     "and widen (or narrow) each bound by that amount. The correction is estimated only on validation data and applied "
     "unchanged to the test and future forecasts. Because each tail is corrected separately, the recalibrated 10 % and "
     "90 % quantiles remain usable by the order rule.")

subheading("Decision Layer: Orders for Bread and Dairy")
para("Following the newsvendor model [22], the recommended availability is the critical quantile of the forecast demand")
e_tau = equation("τ* = cu / (cu + co)")
para("where cu is the cost of a missing unit and co the cost of a leftover one. Since the dataset has no costs, the user "
     "sets their ratio in the application; the quantile is interpolated linearly between the three estimated quantiles "
     "on the log scale and limited to [0.1, 0.9]. For bread, leftovers are lost at the end of the day, so this quantity "
     "is the daily order. Dairy keeps several days, so ordering the newsvendor quantity every day would accumulate stock. "
     f"For dairy we therefore simulate a perishable inventory with first-in-first-out sales, a shelf life of {DL} days "
     "(and 3 days as a sensitivity case) and no lead time [26], and compare four rules on the test window: P1, the "
     "daily newsvendor that ignores the stock on hand; P2, order-up-to the median; P3, order-up-to the τ* quantile, "
     "which orders only the difference between the target and the usable stock; and P4, order-up-to a quantile "
     "selected on the validation windows from a grid between 0.1 and 0.9. Cost is cu times the units short plus co "
     "times the units that expire.")
subheading("Anomaly Alerts and Explanations")
para("When real sales arrive, a warning is raised if they fall outside the 10–90 % interval on two consecutive days, "
     "and a critical alert if they exceed the interval by more than half its width on one day. Explanations use "
     "integrated gradients [29] on the median forecast with 32 integration steps. The baseline is a neutral week: "
     "day-of-week inputs set to 1/7, no promotion, holiday or payday, and the sales history replaced by its mean level. "
     "Attributions are grouped into five factors (promotion, day of week, payday, holiday and recent history). To check "
     "them we compare their shares with 1) occlusion, the change of the median forecast when only one factor is set "
     "to its baseline value, and 2) retraining ablation, the change of validation RMSLE when the network is retrained "
     "without that factor.")
subheading("Application and Implementation")
para("The application, built with Streamlit [36], lets the user choose a store, a family and a period (the August "
     "16–31 forecast or a backtest against the real sales of the test window). It shows the recent sales, the 16-day "
     "forecast with its interval, a control for the cost ratio with the resulting daily order (and the carried stock "
     "for dairy), the factors behind each day's forecast with a short sentence, the series with active alerts and the "
     "accuracy of every model. It reads only the files exported by the experiments, so it runs without a GPU. The "
     "pipeline uses pandas and NumPy, the networks and integrated gradients are written in PyTorch [37], the baselines "
     "use statsmodels [38] and LightGBM [19], and the TFT uses PyTorch Forecasting. All experiments run in one Google "
     f"Colab notebook with a T4 GPU and fixed seeds; the code is available at {GITHUB_URL} and a promotional video at "
     f"{VIDEO_URL}.")

# ----------------------------------------------------------------------------- IV. Results
heading("Results")
subheading("Training and Tuning")
f_train = figure(RES_DIR / "fig_training_lstm.png", "Pinball loss of the LSTM (seed 42) by epoch on the training data and the validation windows; the dashed line marks the selected epoch.")
ep = int(pd.read_csv(RES_DIR / "training_curve_lstm.csv").epoch.max())
para(f"The configuration with hidden size {int(best_cfg.hidden)}, dropout {best_cfg.dropout} and learning rate "
     f"{best_cfg.lr:.0e} obtained the lowest validation pinball loss (Table {ROMAN[t_tune - 1]}), although the "
     "differences between configurations are small. Fig. "
     f"{f_train} shows that the validation loss flattens after a few epochs and that there is no growing gap between "
     "the curves that would indicate overfitting; early stopping restores the best epoch.")

subheading("Forecast Accuracy")
rows = []
for f in range(2):
    for name, label in [("Seasonal naive", "Seasonal naive"), ("ETS", "ETS")]:
        rows.append([label, ["Bread", "Dairy"][f], fmt(m(name, f)), f"{m(name, f, 'WAPE_pct'):.2f}", fmt(m(name, f, 'Pinball_log')), f"{m(name, f, 'Coverage_pct'):.1f}"])
    if TFT is not None:
        tr = TFT[(TFT.model == "TFT") & (TFT.split == "test") & (TFT.family == FAM[f])].iloc[0]
        rows.append(["TFT", ["Bread", "Dairy"][f], fmt(tr.RMSLE), f"{tr.WAPE_pct:.2f}", fmt(tr.Pinball_log), f"{tr.Coverage_pct:.1f}"])
    for base in ["LightGBM", "GRU", "LSTM"]:
        mu, sd = ms(base, f); wm, ws = ms(base, f, "WAPE_pct"); pm, _ = ms(base, f, "Pinball_log"); cm, _ = ms(base, f, "Coverage_pct")
        rows.append([f"{base} ({N_SEEDS} seeds)", ["Bread", "Dairy"][f], f"{mu:.4f}±{sd:.4f}", f"{wm:.2f}", fmt(pm), f"{cm:.1f}"])
        e = f"{base} ensemble + conformal"
        rows.append([f"{base} ens. + conf.", ["Bread", "Dairy"][f], fmt(m(e, f)), f"{m(e, f, 'WAPE_pct'):.2f}", fmt(m(e, f, 'Pinball_log')), f"{m(e, f, 'Coverage_pct'):.1f}"])
test_df = pd.DataFrame(rows, columns=["Model", "Family", "RMSLE", "WAPE (%)", "Pinball", "Cov. (%)"])
t_test = table(test_df, "Test Results (July 31 – August 15, 2017)",
               note=(f"Seeds: mean±standard deviation over {N_SEEDS} seeds (WAPE, pinball and coverage: means). Ens. + conf.: "
                     "average of the seed forecasts with conformal recalibration (the median, hence RMSLE and WAPE, is "
                     "not changed by recalibration). Pinball loss on the ln(1+x) scale. Cov.: open days inside the "
                     "10–90 % interval (target 80 %). Best value per family in bold."),
               widths=[1.0, .45, .75, .45, .45, .4], bold_min=[("RMSLE", "Family"), ("WAPE (%)", "Family"), ("Pinball", "Family")])
vrows = []
for f in range(2):
    for name, label in [("Seasonal naive", "Seasonal naive"), ("ETS", "ETS"), ("LightGBM ensemble", "LightGBM ens."),
                        ("GRU ensemble", "GRU ens."), ("LSTM ensemble", "LSTM ens.")]:
        vrows.append([label, ["Bread", "Dairy"][f], fmt(m(name, f, split="validation")), f"{m(name, f, 'WAPE_pct', 'validation'):.2f}",
                      fmt(m(name, f, 'Pinball_log', 'validation')), f"{m(name, f, 'Coverage_pct', 'validation'):.1f}"])
val_df = pd.DataFrame(vrows, columns=["Model", "Family", "RMSLE", "WAPE (%)", "Pinball", "Cov. (%)"])
t_val = table(val_df, "Validation Results (Four Windows, May 28 – July 30, 2017)",
              note="Raw (not recalibrated) intervals. Best value per family in bold.",
              widths=[1.0, .45, .55, .5, .5, .5], bold_min=[("RMSLE", "Family"), ("WAPE (%)", "Family"), ("Pinball", "Family")])
lgb_m = {f: ms("LightGBM", f)[0] for f in range(2)}; lstm_m = {f: ms("LSTM", f)[0] for f in range(2)}
para(f"Tables {ROMAN[t_test - 1]} and {ROMAN[t_val - 1]} present the results. First, all learned models clearly beat "
     f"the statistical baselines: on the test window the LSTM ensemble reduces RMSLE by {b_naive:.0f} % (bread) and "
     f"{d_naive:.0f} % (dairy) with respect to the seasonal naive model and by "
     f"{red(rm[0]['LSTM ensemble'], rm[0]['ETS']):.0f} % and {red(rm[1]['LSTM ensemble'], rm[1]['ETS']):.0f} % with "
     f"respect to ETS. Second, for dairy the LSTM ensemble {cmp_word(rm[1]['LSTM ensemble'], rm[1]['LightGBM ensemble'])} "
     f"the LightGBM ensemble (RMSLE {rm[1]['LSTM ensemble']:.4f} versus {rm[1]['LightGBM ensemble']:.4f}), and the "
     f"single LSTM beats the single LightGBM in {wins[1]} of {N_SEEDS} seeds. Third, for bread the LSTM ensemble "
     f"{cmp_word(rm[0]['LSTM ensemble'], rm[0]['LightGBM ensemble'])} LightGBM ({rm[0]['LSTM ensemble']:.4f} versus "
     f"{rm[0]['LightGBM ensemble']:.4f}; {wins[0]} of {N_SEEDS} seeds in favor of the LSTM). Fourth, the run-to-run "
     f"standard deviation of the LSTM test RMSLE ({ms('LSTM', 0)[1]:.4f} for bread and {ms('LSTM', 1)[1]:.4f} for "
     "dairy) is of the same order as some of the differences between models, which is why the paired comparison by "
     f"seed (Fig. {FIG_N[0] + 1}) and the ensembles are more reliable than a single run. In validation, LightGBM "
     f"obtains RMSLE {m('LightGBM ensemble', 0, split='validation'):.4f} and {m('LightGBM ensemble', 1, split='validation'):.4f} "
     f"against {m('LSTM ensemble', 0, split='validation'):.4f} and {m('LSTM ensemble', 1, split='validation'):.4f} for "
     "the LSTM, so conclusions drawn from the single 16-day test window must be read with care.")
f_seeds = figure(RES_DIR / "fig_seeds_rmsle.png", f"Test RMSLE of LightGBM, LSTM and GRU over {N_SEEDS} seeds (box) and of their seed ensembles (star).")
if TFT is not None:
    tb = TFT[(TFT.model == "TFT") & (TFT.split == "test")].set_index("family").RMSLE
    v4 = TFT[TFT.split == "validation (window 4)"].pivot_table(index="model", columns="family", values="RMSLE")
    para(f"The TFT reaches test RMSLE {tb[FAM[0]]:.4f} (bread) and {tb[FAM[1]]:.4f} (dairy). On validation window 4, "
         f"its RMSLE is {v4.loc['TFT', FAM[0]]:.4f} and {v4.loc['TFT', FAM[1]]:.4f}, against "
         f"{v4.loc['LSTM ensemble', FAM[0]]:.4f} and {v4.loc['LSTM ensemble', FAM[1]]:.4f} for the LSTM ensemble. "
         "Its training budget was smaller and it was trained once, so this comparison is indicative; its "
         "variable-selection weights are discussed with the explanations.")
f_fc = figure(RES_DIR / "fig_forecast_example.png", "Test forecast of the recalibrated LSTM ensemble for the store with the highest bread sales (top: BREAD/BAKERY; bottom: DAIRY): history, real sales, median and 10–90 % interval.")
para(f"Fig. {f_fc} shows the test forecast for the store with the highest bread sales: the median follows the weekly "
     "cycle and most real values fall inside the interval.")

subheading("Interval Calibration")
crow = []
for f in range(2):
    for base, lab in [("ETS", "ETS"), ("LightGBM ensemble", "LightGBM"), ("GRU ensemble", "GRU"), ("LSTM ensemble", "LSTM")]:
        crow.append([lab, ["Bread", "Dairy"][f], f"{m(base, f, 'Coverage_pct', 'validation'):.1f}",
                     f"{m(base, f, 'Coverage_pct'):.1f}", f"{m(base + ' + conformal', f, 'Coverage_pct'):.1f}",
                     fmt(m(base, f, 'Pinball_log')), fmt(m(base + ' + conformal', f, 'Pinball_log'))])
cal_df = pd.DataFrame(crow, columns=["Model", "Family", "Val. cov.", "Test cov.", "Test cov. conf.", "Pinball", "Pinball conf."])
t_cal = table(cal_df, "Coverage (%) and Pinball Loss Before and After Conformal Recalibration",
              note="Val. cov.: raw coverage on the validation windows used to estimate the correction. Target coverage 80 %.",
              widths=[.6, .45, .5, .5, .6, .45, .5])
f_cov = figure(RES_DIR / "fig_coverage_conformal.png", "Coverage of the 10–90 % interval on the test window before (light) and after (dark) conformal recalibration; the dashed line is the 80 % target.")
para(f"Table {ROMAN[t_cal - 1]} and Fig. {f_cov} show that the raw intervals are miscalibrated in opposite directions: "
     f"the statistical baselines are too wide, while the learned models are too narrow on the test window (LSTM "
     f"ensemble {cov[0]['LSTM ensemble']:.1f} % for bread and {cov[1]['LSTM ensemble']:.1f} % for dairy; LightGBM "
     f"{cov[0]['LightGBM ensemble']:.1f} % and {cov[1]['LightGBM ensemble']:.1f} %). After recalibration the LSTM "
     f"reaches {cov[0][L]:.1f} % and {cov[1][L]:.1f} % and LightGBM {cov[0][LG]:.1f} % and {cov[1][LG]:.1f} %, and the "
     f"pinball loss of the LSTM changes from {m('LSTM ensemble', 0, 'Pinball_log'):.4f} to {m(L, 0, 'Pinball_log'):.4f} "
     f"(bread) and from {m('LSTM ensemble', 1, 'Pinball_log'):.4f} to {m(L, 1, 'Pinball_log'):.4f} (dairy). The "
     "correction comes from validation windows that are farther from the end of training than the test window, so it "
     "can over- or under-correct; coverage guarantees hold only if errors are exchangeable across periods [28].")

subheading("Ordering Decisions")
nvrows = []
for model, lab in [("LSTM ensemble", "LSTM (raw)"), (L, "LSTM + conf."), ("LightGBM ensemble", "LightGBM (raw)"), (LG, "LightGBM + conf.")]:
    nvrows.append([lab, f"{nv(model, 0, '2:1'):+.1f}", f"{nv(model, 0, '4:1'):+.1f}", f"{nv(model, 1, '2:1'):+.1f}", f"{nv(model, 1, '4:1'):+.1f}"])
nv_df = pd.DataFrame(nvrows, columns=["Model", "Bread 2:1", "Bread 4:1", "Dairy 2:1", "Dairy 4:1"])
t_nv = table(nv_df, "Cost Saving (%) of the Daily Newsvendor Order vs. Ordering the Median (Test)",
             note="Ratio cu:co = 2:1 (τ* = 0.67) and 4:1 (τ* = 0.80). Positive: the newsvendor order is cheaper. Single-day cost, leftovers lost.",
             widths=[1.1, .6, .6, .6, .6])
invrows = []
for life in sorted(INV[INV.family == "DAIRY"].shelf_life.unique(), reverse=True):
    for ratio in ["2:1", "4:1"]:
        for pol in ["P1", "P2", "P3", "P4"]:
            r_ = inv(1, life, ratio, pol)
            invrows.append([f"{life} d", ratio, pol, f"{r_.tau:.2f}", f"{r_.cost / 1000:,.1f}", f"{r_.waste_pct:.1f}", f"{r_.fill_rate_pct:.1f}"])
inv_df = pd.DataFrame(invrows, columns=["Life", "cu:co", "Rule", "τ", "Cost (k)", "Waste (%)", "Fill (%)"])
inv_df["grp"] = inv_df.Life + inv_df["cu:co"]
t_inv = table(inv_df.drop(columns="grp"), "Dairy Ordering Rules on the Test Window (Perishable Inventory Simulation)",
              note=("P1: daily newsvendor ignoring stock; P2: order-up-to median; P3: order-up-to τ*; P4: order-up-to τ "
                    "selected on validation. Cost in thousands of units (cu × short + co × expired). Waste: expired "
                    "units as % of demand. Fill: share of demand served. LSTM ensemble + conformal."),
              widths=[.4, .45, .4, .4, .6, .6, .55])
f_inv = figure(RES_DIR / "fig_dairy_policies.png", f"Simulated cost of the four dairy ordering rules on the test window (shelf life {DL} days) and the resulting waste.")
p1, p3, p4 = inv(1, DL, "4:1", "P1"), inv(1, DL, "4:1", "P3"), inv(1, DL, "4:1", "P4")
b3, b4 = inv(0, 1, "4:1", "P3"), inv(0, 1, "4:1", "P4")
para(f"For bread (Table {ROMAN[t_nv - 1]}), ordering the τ* quantile of the recalibrated LSTM instead of the median "
     f"changes the simulated cost by {nv(L, 0, '2:1'):+.1f} % at a 2:1 ratio and {nv(L, 0, '4:1'):+.1f} % at 4:1, "
     f"against {nv('LSTM ensemble', 0, '4:1'):+.1f} % with the raw intervals, which shows that the quality of the "
     "upper quantile matters for the decision. In the inventory simulation with one-day life, the τ selected on "
     f"validation for bread at 4:1 is {b4.tau:.2f} (cost {b4.cost / 1000:,.1f} k versus {b3.cost / 1000:,.1f} k for τ* = "
     f"{b3.tau:.2f}). For dairy (Table {ROMAN[t_inv - 1]} and Fig. {f_inv}), the daily newsvendor that ignores the "
     f"stock on hand wastes {p1.waste_pct:.1f} % of demand at 4:1 with a {DL}-day life, while the order-up-to rule "
     f"with τ* wastes {p3.waste_pct:.1f} % with a fill rate of {p3.fill_rate_pct:.1f} %, a cost change of "
     f"{p3.saving_vs_first_pct:+.1f} %; the rule tuned on validation (τ = {p4.tau:.2f}) changes the cost by "
     f"{p4.saving_vs_first_pct:+.1f} %. Accounting for carried stock is therefore what makes the quantile forecast "
     "useful for a product that lasts several days.")

subheading("Anomaly Alerts")
alrows = []
for model, lab in [("LSTM ensemble", "LSTM (raw)"), (L, "LSTM + conf."), ("LightGBM ensemble", "LightGBM (raw)"), (LG, "LightGBM + conf.")]:
    for f in range(2):
        alrows.append([lab, ["Bread", "Dairy"][f], f"{al(model, f, 'days_outside_pct'):.1f}", int(al(model, f, "warnings")),
                       int(al(model, f, "critical")), f"{int(al(model, f, 'stores_with_alert'))}/54"])
al_df = pd.DataFrame(alrows, columns=["Model", "Family", "Outside (%)", "Warn.", "Crit.", "Stores"]).astype(str)
t_al = table(al_df, "Anomaly Alerts on the Test Window",
             note="Outside: open days with real sales outside the 10–90 % interval (about 20 % expected with a calibrated interval). Stores: stores with at least one alert in 16 days.",
             widths=[1.0, .45, .55, .45, .45, .5])
para(f"Table {ROMAN[t_al - 1]} shows the alert rule. With calibrated intervals about 20 % of open days are expected "
     f"outside the interval; after recalibration the LSTM leaves {al(L, 0, 'days_outside_pct'):.1f} % (bread) and "
     f"{al(L, 1, 'days_outside_pct'):.1f} % (dairy) of days outside, and {int(al(L, 0, 'stores_with_alert'))} and "
     f"{int(al(L, 1, 'stores_with_alert'))} of the 54 stores receive at least one alert in 16 days, against "
     f"{int(al('LSTM ensemble', 0, 'stores_with_alert'))} and {int(al('LSTM ensemble', 1, 'stores_with_alert'))} with "
     "the raw intervals. The alert rate is therefore controlled by the calibration of the interval, and a store "
     "manager can trade sensitivity for fewer alerts by widening it.")

subheading("Explanations")
exrows = []
for f in range(2):
    e = ex(f)
    for g in ["Promotion", "Day of week", "Payday", "Holiday", "Recent history"]:
        exrows.append([g, ["Bread", "Dairy"][f], f"{e.loc[g, 'IG_share_pct']:.1f}", f"{e.loc[g, 'occlusion_share_pct']:.1f}",
                       f"{e.loc[g, 'delta_RMSLE_pct']:+.1f}" if "delta_RMSLE_pct" in e else "–"])
ex_df = pd.DataFrame(exrows, columns=["Factor", "Family", "IG (%)", "Occl. (%)", "ΔRMSLE (%)"])
t_ex = table(ex_df, "Validation of the Explanations (LSTM, Seed 42)",
             note="IG: share of the integrated-gradient attribution (test window). Occl.: share of the change of the median when the factor is set to its baseline. ΔRMSLE: change of validation RMSLE when the network is retrained without the factor.",
             widths=[1.0, .5, .6, .6, .7])
f_ex = figure(RES_DIR / "fig_explanations.png", "Share of attribution by factor of the LSTM median forecast on the test window: integrated gradients versus occlusion.")
txt = (f"Table {ROMAN[t_ex - 1]} and Fig. {f_ex} compare the three views. Integrated gradients assign the largest share "
       f"to {top_factor[0].lower()} for bread and to {top_factor[1].lower()} for dairy, and their shares agree closely "
       f"with occlusion (correlation {corr[0]:.2f} and {corr[1]:.2f}), which means the attributions are faithful to the "
       "model. ")
if abl_top:
    txt += (f"Retraining ablation tests something different, whether the information is needed: removing "
            f"{abl_top[0].lower()} hurts bread the most ({ex(0).loc[abl_top[0], 'delta_RMSLE_pct']:+.1f} % RMSLE) and "
            f"removing {abl_top[1].lower()} hurts dairy the most ({ex(1).loc[abl_top[1], 'delta_RMSLE_pct']:+.1f} %), "
            "while factors whose removal barely changes the error can be replaced by other inputs that carry similar "
            "information (for example, recent history already contains the weekly pattern). ")
if TFT_ENC is not None:
    top3 = ", ".join(TFT_ENC.sort_values(ascending=False).index[:3])
    txt += f"The TFT variable-selection weights rank {top3} first among the encoder variables. "
txt += ("Integrated gradients depend on the chosen baseline, so in the application the explanation is presented as "
        "the effect of each factor relative to a neutral week, not as a causal effect.")
para(txt)

subheading("Application")
app_png = RES_DIR / "fig_app.png"
if app_png.exists():
    f_app = figure(app_png, "FreshCast application: 16-day forecast with its interval and availability target for a store and family, recommended orders and factor explanations.")
    para(f"Fig. {f_app} shows the application. A store manager selects the store and family, reads the expected sales "
         "and their range, sets how costly a shortage is relative to a leftover and obtains the order for each day; for "
         "dairy the order discounts the stock carried from previous days. The Why? tab explains each day's forecast "
         "in one sentence, and the alerts tab lists the series whose sales left the expected range.")

# ----------------------------------------------------------------------------- V. Discussion
heading("Discussion")
para("Three lessons stand out. First, the recurrent network is a strong model for these perishable families, clearly "
     "better than statistical baselines, but its advantage over a well-engineered gradient boosting model depends on "
     "the family and the period, and the variation between training seeds is of the same order as some differences "
     "between models; measuring that variability and ensembling seeds is necessary before claiming that one model is "
     "better, as anticipated by the M5 findings [5]. Second, for a decision-making agent the calibration of the "
     "intervals matters as much as the accuracy of the median: a simple conformal step estimated on validation data "
     "changes the order cost and the alert rate more than switching between the learned models. Third, the decision "
     "rule must match the product: the daily newsvendor is adequate for bread, but for dairy it accumulates stock and "
     "waste, and an order-up-to rule that accounts for the carried stock is required.")
para("The study has limitations. The dataset has no costs, stock levels or shelf lives, so the decision results are "
     "simulations with assumed ratios and lives, and sales are censored by availability, so the true demand on "
     "stock-out days is unknown. The holdout test is one 16-day window of 2017; conclusions about small differences "
     "between models would need more test windows. The TFT received a smaller training budget than the recurrent "
     "networks. The explanations describe the model, not the causal effect of promotions or holidays, which are not "
     "assigned at random.")

# ----------------------------------------------------------------------------- VI. Conclusions
heading("Conclusions")
para("We built and evaluated FreshCast, an intelligent agent that forecasts the daily demand for bread and dairy of 54 "
     "supermarkets as recalibrated quantiles, turns them into orders that respect the shelf life of each family, "
     "raises alerts when sales leave the expected range and explains each forecast by factor, all delivered in a web "
     f"application. The LSTM encoder–decoder, trained as one global model with {N_SEEDS} seeds, reduces RMSLE with "
     f"respect to the seasonal naive model by {b_naive:.0f} % for bread and {d_naive:.0f} % for dairy, "
     f"{cmp_word(rm[1]['LSTM ensemble'], rm[1]['LightGBM ensemble'])} quantile LightGBM for dairy and "
     f"{cmp_word(rm[0]['LSTM ensemble'], rm[0]['LightGBM ensemble'])} it for bread; conformal recalibration brings its "
     f"coverage to {cov[0][L]:.0f} % and {cov[1][L]:.0f} %; and the order rules reduce the simulated cost of shortages "
     "and waste, with the carried-stock rule being essential for dairy. Future work includes evaluating on more test "
     "windows, using stock and cost data from a real store, censored-demand corrections, a joint multi-day "
     "optimization of the dairy order and a field study of how store staff use the explanations.")

heading("Acknowledgment", numbered=False)
para("The authors thank Dr. Alexandra La Cruz for her guidance during the four phases of the project, and Corporación "
     "Favorita and Kaggle for making the data public.")

# ----------------------------------------------------------------------------- references
heading("References", numbered=False)
REFS = [
    'R. Fildes, S. Ma, and S. Kolassa, "Retail forecasting: Research and practice," *Int. J. Forecasting*, vol. 38, no. 4, pp. 1283–1318, 2022.',
    'J. Huber and H. Stuckenschmidt, "Daily retail demand forecasting using machine learning with emphasis on calendric special days," *Int. J. Forecasting*, vol. 36, no. 4, pp. 1420–1438, 2020.',
    'D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, "DeepAR: Probabilistic forecasting with autoregressive recurrent networks," *Int. J. Forecasting*, vol. 36, no. 3, pp. 1181–1191, 2020.',
    'B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, "Temporal fusion transformers for interpretable multi-horizon time series forecasting," *Int. J. Forecasting*, vol. 37, no. 4, pp. 1748–1764, 2021.',
    'S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions," *Int. J. Forecasting*, vol. 38, no. 4, pp. 1346–1364, 2022.',
    'H. Hewamalage, C. Bergmeir, and K. Bandara, "Recurrent neural networks for time series forecasting: Current status and future directions," *Int. J. Forecasting*, vol. 37, no. 1, pp. 388–427, 2021.',
    'K. Benidis et al., "Deep learning for time series forecasting: Tutorial and literature survey," *ACM Comput. Surv.*, vol. 55, no. 6, Art. no. 121, 2023.',
    'S. J. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2021.',
    'J. W. Taylor, "Forecasting daily supermarket sales using exponentially weighted quantile regression," *Eur. J. Oper. Res.*, vol. 178, no. 1, pp. 154–167, 2007.',
    'S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.',
    'K. Cho et al., "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2014, pp. 1724–1734.',
    'I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning with neural networks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 27, 2014, pp. 3104–3112.',
    'R. Wen, K. Torkkola, B. Narayanaswamy, and D. Madeka, "A multi-horizon quantile recurrent forecaster," in *NeurIPS Time Series Workshop*, 2017, arXiv:1711.11053.',
    'C. Guo and F. Berkhahn, "Entity embeddings of categorical variables," 2016, arXiv:1604.06737.',
    'B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.',
    'A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are transformers effective for time series forecasting?," in *Proc. AAAI Conf. Artif. Intell.*, vol. 37, no. 9, 2023, pp. 11121–11128.',
    'S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "Statistical and machine learning forecasting methods: Concerns and ways forward," *PLoS ONE*, vol. 13, no. 3, Art. no. e0194889, 2018.',
    'K. Bandara et al., "Sales demand forecast in e-commerce using a long short-term memory neural network methodology," in *Proc. Int. Conf. Neural Inf. Process. (ICONIP)*, LNCS, vol. 11955, 2019, pp. 462–474.',
    'G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.',
    'S. Makridakis et al., "The M5 uncertainty competition: Results, findings and conclusions," *Int. J. Forecasting*, vol. 38, no. 4, pp. 1365–1385, 2022.',
    'C. S. Bojer and J. P. Meldgaard, "Kaggle forecasting competitions: An overlooked learning opportunity," *Int. J. Forecasting*, vol. 37, no. 2, pp. 587–603, 2021.',
    'Y. Qin, R. Wang, A. J. Vakharia, Y. Chen, and M. M. H. Seref, "The newsvendor problem: Review and directions for future research," *Eur. J. Oper. Res.*, vol. 213, no. 2, pp. 361–374, 2011.',
    'R. Koenker and G. Bassett, "Regression quantiles," *Econometrica*, vol. 46, no. 1, pp. 33–50, 1978.',
    'T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *J. Amer. Statist. Assoc.*, vol. 102, no. 477, pp. 359–378, 2007.',
    'J. Huber, S. Müller, M. Fleischmann, and H. Stuckenschmidt, "A data-driven newsvendor problem: From data to decision," *Eur. J. Oper. Res.*, vol. 278, no. 3, pp. 904–915, 2019.',
    'S. Nahmias, "Perishable inventory theory: A review," *Oper. Res.*, vol. 30, no. 4, pp. 680–708, 1982.',
    'Y. Romano, E. Patterson, and E. Candès, "Conformalized quantile regression," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 3543–3553.',
    'A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Found. Trends Mach. Learn.*, vol. 16, no. 4, pp. 494–591, 2023.',
    'M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 3319–3328.',
    'T. Rojat, R. Puget, D. Filliat, J. Del Ser, R. Gelin, and N. Díaz-Rodríguez, "Explainable artificial intelligence (XAI) on time series data: A survey," 2021, arXiv:2104.00950.',
    'Kaggle, "Store Sales – Time Series Forecasting," Kaggle Competitions. [Online]. Available: https://www.kaggle.com/c/store-sales-time-series-forecasting (accessed: Nov. 2026).',
    'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.',
    'N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014.',
    'R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.',
    'L. J. Tashman, "Out-of-sample tests of forecasting accuracy: An analysis and review," *Int. J. Forecasting*, vol. 16, no. 4, pp. 437–450, 2000.',
    'Streamlit Inc., "Streamlit: A faster way to build and share data apps," 2026. [Online]. Available: https://streamlit.io',
    'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.',
    'S. Seabold and J. Perktold, "Statsmodels: Econometric and statistical modeling with Python," in *Proc. 9th Python in Science Conf. (SciPy)*, 2010, pp. 92–96.',
]
for i, ref in enumerate(REFS, 1):
    p = para(indent=False); p.paragraph_format.left_indent = Inches(.3); p.paragraph_format.first_line_indent = Inches(-.3)
    add_runs(p, f"[{i}]\t{ref}", 8)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(.3))

doc.save(OUT)
print("wrote", OUT, "| figures", FIG_N[0], "| tables", TAB_N[0], "| equations", EQ_N[0], "| references", len(REFS))
