# Deploying NeuraVol

The website is a Streamlit app (`app.py`). It only reads small files in this folder, so it runs free on
Streamlit Community Cloud. The daily data refresh is a separate job (`daily_update.py`).

## 1. Run it on your laptop first

```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

## 2. Put the project on GitHub

1. Create a new repository on github.com (e.g. `neuravol-forecast`).
2. In this folder:
   ```
   git init
   git add .
   git commit -m "NeuraVol v1"
   git branch -M main
   git remote add origin https://github.com/<your-username>/neuravol-forecast.git
   git push -u origin main
   ```
   `.gitignore` keeps the 1.5 GB of raw bhavcopies out of the repo; the app does not need them.

## 3. Publish the website (free)

1. Go to share.streamlit.io and sign in with GitHub.
2. "Create app" -> pick the repository, branch `main`, main file `app.py`.
3. Deploy. You get a public link like `https://neuravol.streamlit.app` for your CV.

## 4. Daily updates

**Option A: GitHub Actions (automatic).** `.github/workflows/daily_update.yml` runs `daily_update.py` at 19:45 IST
on weekdays and commits the new surface and forecast; Streamlit Cloud redeploys automatically.
Enable it under the repo's Actions tab. You can also trigger it by hand ("Run workflow").
Note: NSE sometimes blocks requests from cloud servers. If the job logs show `403 blocked`, use option B.

**Option B: your laptop (always works).** Schedule this with Windows Task Scheduler at ~19:45 on weekdays:
```
cd "C:\Users\darsh\OneDrive - Vidyalankar Institute of Technology\NEURAVOLAFORECAST"
python daily_update.py
git add data/surfaces data/forecasts data/latest_chain.csv
git commit -m "daily update"
git push
```
`daily_update.py` needs the pipeline packages: `pip install -r requirements-pipeline.txt`.

## 5. Retraining

Re-run `notebooks/NeuraVol_v2_Colab.ipynb` (monthly is plenty) with the latest `data/surfaces/` files, copy the new
`neuravol_v2.pt` into `models/`, then `python predict.py` and push.

## What each file is for

| File | Purpose |
|---|---|
| `app.py`, `dashboard_data.py`, `.streamlit/config.toml` | the website |
| `models/neuravol_v2.pt` | trained model (CPU inference) |
| `predict.py`, `src/model.py`, `src/features.py` | forecast from the latest day |
| `daily_update.py` | fetch new bhavcopy -> new surface -> new forecast |
| `data/surfaces/`, `data/forecasts/`, `data/track_record/`, `data/latest_chain.csv` | data the app reads |
| `reports/backtest/`, `reports/nn_v2/nn_v2_results.json` | results shown on the Track record page |
