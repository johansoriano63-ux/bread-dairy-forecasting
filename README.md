# FreshCast: Probabilistic Forecasting and Ordering Agent for Bakery and Dairy

Final project of the **Intelligent Systems** course, Universidad de Ibagué (Colombia), 2026.
Stephany Trujillo · Johan Soriano · Cristian Cardona · Instructor: Dr. Alexandra La Cruz

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johansoriano63-ux/bread-dairy-forecasting/blob/main/notebooks/Final_Experiments_Bread_Dairy.ipynb)

FreshCast is an intelligent agent for two perishable product families of the Ecuadorian supermarket chain
Corporación Favorita: **BREAD/BAKERY** and **DAIRY** (54 stores, 108 daily series, Kaggle
[Store Sales – Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting)).
Its core is an **LSTM encoder–decoder** with store and family embeddings and a non-crossing quantile head,
trained as one global model. The agent:

1. forecasts the next 16 days of sales as a **10–90 % interval** (quantiles 0.1, 0.5, 0.9), recalibrated with conformal prediction;
2. turns the forecast into a **recommended order** (newsvendor critical quantile for bread, order-up-to rule with carried stock for dairy);
3. raises **anomaly alerts** when real sales leave the interval;
4. **explains** each forecast by factor (promotion, day of week, payday, holiday, recent history) with integrated gradients.

| Resource | Link |
|---|---|
| Report (IEEE paper, PDF) | [`paper/FreshCast_Final_Report.pdf`](paper/) |
| Experiments notebook (Colab) | [`notebooks/Final_Experiments_Bread_Dairy.ipynb`](notebooks/Final_Experiments_Bread_Dairy.ipynb) · [open in Colab](https://colab.research.google.com/github/johansoriano63-ux/bread-dairy-forecasting/blob/main/notebooks/Final_Experiments_Bread_Dairy.ipynb) |
| How to run it (Spanish guide) | [`docs/GUIA_COLAB.md`](docs/GUIA_COLAB.md) |
| Dataset (Google Drive) | **DRIVE_LINK** |
| Promotional video (3 min) | **VIDEO_LINK** |
| Web application | `streamlit run app/app.py` (or **APP_LINK** if deployed) |

## Repository structure

```
├── src/final_experiments.py     # every experiment of the report, as a script with # %% cells
├── src/make_notebook.py         # builds the Colab notebook from the script
├── notebooks/
│   ├── Final_Experiments_Bread_Dairy.ipynb   # final experiments (run this one)
│   └── Phase3_Experiments_Bread_Dairy.ipynb  # phase 3 notebook (history)
├── app/app.py                   # Streamlit application (reads app/data, no GPU needed)
├── app/data/                    # forecasts, attributions and metrics exported by the experiments
├── results/                     # tables (.csv) and figures (.png) used in the report
├── paper/                       # final report (.docx and .pdf) and its generator build_paper.py
├── docs/                        # Colab guide, video script and previous deliveries
└── data/                        # put the Kaggle CSV files here (not versioned)
```

## Reproducing the results

**Google Colab (recommended, ~1.5 h on a T4 GPU)**

1. Open `notebooks/Final_Experiments_Bread_Dairy.ipynb` in Colab and select `Runtime → Change runtime type → T4 GPU`.
2. Upload `store-sales-time-series-forecasting.zip` (or the CSV files) to `/content` — it is extracted automatically
   (or set `USE_DRIVE = True` to copy them from the shared Drive folder, or `USE_KAGGLE_API = True`).
   Results and a resume cache are saved in Google Drive (`MyDrive/freshcast_results`).
3. `Runtime → Run all`. At the end `results_final.zip` is downloaded: copy its tables and figures to `results/`
   and the content of `results/app_data/` to `app/data/`. Step-by-step guide: [`docs/GUIA_COLAB.md`](docs/GUIA_COLAB.md).

**Locally**

```bash
pip install -r requirements.txt
python src/final_experiments.py          # data in ./data, results in ./results (slow without a GPU)
```

Environment variables: `N_SEEDS` (default 5), `RUN_TUNING`, `RUN_ABLATION`, `RUN_TFT` (1/0) and `SMOKE=1`
for a quick end-to-end check with tiny training budgets.

## Running the application

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Choose a store and a family, the period (forecast for Aug 16–31, 2017 or backtest against real sales of
Jul 31 – Aug 15, 2017), the cost of a shortage relative to a leftover and, for dairy, the shelf life.
The app shows the forecast interval, the recommended daily order, the factors behind each day's forecast,
the stores with active alerts and the accuracy of every model.

## Method in brief

* **Data**: 108 series (54 stores × 2 families), Jan 1, 2013 – Aug 15, 2017; sales in ln(1+x); closed days and
  days before each store opened are masked; national/regional/local holidays, promotions, paydays (15th and last day),
  oil price, transactions and the April 2016 earthquake as covariates.
* **Model**: 2-layer LSTM encoder (56 days) and decoder (16 days of known covariates), embeddings of store, type,
  cluster and family, per-window scaling, median + two softplus offsets so quantiles never cross, pinball loss.
* **Protocol**: 4 rolling validation windows (May 28 – Jul 30, 2017) for early stopping and tuning, and a 16-day
  holdout test (Jul 31 – Aug 15, 2017) used once; 5 seeds per model; seed ensembles.
* **Baselines**: seasonal naive, ETS, quantile LightGBM (global, direct multi-horizon), GRU and Temporal Fusion Transformer.
* **Metrics**: RMSLE (official Kaggle metric), WAPE, MAE, pinball loss and coverage of the 10–90 % interval.

## License and data

Code under the MIT license. The data belong to Corporación Favorita and are distributed by Kaggle under the
competition rules; they are not included in this repository.
