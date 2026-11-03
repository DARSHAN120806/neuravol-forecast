@echo off
REM Run once: registers a Windows scheduled task that runs the daily update at 19:45 on weekdays.
schtasks /create /tn "NeuraVol daily update" /tr "\"%~dp0run_daily_update.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 19:45 /f
echo.
echo Done. Check it in Task Scheduler under "NeuraVol daily update". The laptop must be on (not asleep) at 19:45.
pause
