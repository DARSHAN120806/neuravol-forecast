@echo off
REM NeuraVol daily update for Windows: fetch today's NSE bhavcopy, rebuild surface, forecast, push to GitHub.
cd /d "%~dp0.."
if not exist logs mkdir logs
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
echo ===== %date% %time% ===== >> logs\daily_update.log
python daily_update.py >> logs\daily_update.log 2>&1
if exist data\live python collector\build_mid_surface.py >> logs\daily_update.log 2>&1
if exist .git (
  git add data/surfaces data/surfaces_mid data/forecasts data/latest_chain.csv >> logs\daily_update.log 2>&1
  git commit -m "daily update %date%" >> logs\daily_update.log 2>&1
  git push >> logs\daily_update.log 2>&1
)
