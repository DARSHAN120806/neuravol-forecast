# Automating the daily NSE data update

`daily_update.py` downloads the day's F&O bhavcopy from NSE, builds the new volatility surface, runs the model and
logs the forecast. These files run it automatically every weekday at 19:45 IST (NSE publishes around 18:00-19:00).

| Option | Cost | Reliability | Files |
|---|---|---|---|
| **A. Your Windows laptop** | free | works as long as the laptop is on at 19:45; NSE accepts home internet | `setup_windows_task.bat`, `run_daily_update.bat` |
| **B. Linux VPS + cron** | about ₹400-900/month (e.g. a small Mumbai-region server) | always on; NSE sometimes blocks data-centre IPs, so test first | `vps_daily.sh` |
| **C. VPS + n8n** | same as B | same as B, plus a visual dashboard and easy alerts (Telegram, email) | `n8n_workflow.json`, `vps_daily.sh` |
| D. GitHub Actions | free | easiest, but NSE often blocks GitHub's servers | `.github/workflows/daily_update.yml` |

## A. Windows laptop (recommended to start)
1. `pip install -r requirements-pipeline.txt` (includes the CPU version of PyTorch for forecasts).
2. Double-click `scheduler\setup_windows_task.bat` once.
3. Logs go to `logs\daily_update.log`. If the folder is a git clone of your GitHub repo, new data is pushed automatically
   and the online dashboard updates.

## B. Linux VPS with cron
```
git clone https://github.com/DARSHAN120806/neuravol-forecast.git && cd neuravol-forecast
python3 -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt
.venv/bin/python daily_update.py            # test once by hand; check it downloads
crontab -e                                  # add the two lines below
CRON_TZ=Asia/Kolkata
45 19 * * 1-5 /home/ubuntu/neuravol-forecast/scheduler/vps_daily.sh
```
For `git push` from the server, add a GitHub deploy key with write access (repo Settings > Deploy keys) or a
fine-grained personal access token limited to this repository.

## C. n8n on the VPS
1. Install n8n on the same server (Docker or npm) and do steps B1-B3 first.
2. In n8n: Workflows > Import from file > `n8n_workflow.json`. Adjust the path in "Run daily update" if your clone is
   not in `/home/ubuntu/neuravol-forecast`.
3. The Execute Command node only exists in self-hosted n8n, and recent versions disable it by default; enable it via
   the `NODES_EXCLUDE` setting if it is missing.
4. Optional: add a Telegram or email node after it to get a message with the day's forecast.

## Live bid/ask collector (cleaner data)

`collector/collect_chain.py` saves the full NIFTY option chain with bid and ask prices every 15 minutes during market
hours, using the free **Upstox Analytics Token** (read-only, valid 1 year, no daily login):

1. Open https://account.upstox.com/developer/apps, create an app, go to the **Analytics** tab, click **Generate Token**.
2. Save the token in `secrets/upstox_token.txt` (git-ignored; never commit it) and test:
   `python collector/collect_chain.py --once`
3. Schedule it:
   - Windows: double-click `scheduler/setup_collector_task.bat` once (starts 09:10 on weekdays).
   - VPS cron: `10 9 * * 1-5 cd /home/ubuntu/neuravol-forecast && .venv/bin/python collector/collect_chain.py --loop --every 15 >> logs/collector.log 2>&1`
     (with `CRON_TZ=Asia/Kolkata`; copy the token file to the server).

Every evening the daily update also runs `collector/build_mid_surface.py`, which turns the 15:25 snapshot into a
surface built from mid prices in `data/surfaces_mid/`. After 2-3 months of these, retrain the model on them.
