@echo off
REM NeuraVol live collector: snapshots the NIFTY option chain (bid/ask) every 15 minutes until 15:30 IST.
cd /d "%~dp0.."
if not exist logs mkdir logs
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
python collector\collect_chain.py --loop --every 15 >> logs\collector.log 2>&1
