@echo off
REM One-time test of the Upstox live collector: takes one snapshot of the NIFTY option chain.
cd /d "%~dp0.."
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
python collector\collect_chain.py --once
echo.
echo If you see "saved ... rows" above, the collector works.
pause
