@echo off
REM Run once: starts the live collector at 09:10 every weekday (it stops itself after market close).
schtasks /create /tn "NeuraVol live collector" /tr "\"%~dp0run_collector.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:10 /f
echo.
echo Done. Check it in Task Scheduler under "NeuraVol live collector". The laptop must be on during market hours.
pause
