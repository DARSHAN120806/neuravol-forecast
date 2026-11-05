@echo off
rem Smoke test of the v3 training code (QUICK mode, a few minutes). Output: logs\v3_quick.log
cd /d "%~dp0\.."
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist logs mkdir logs
set NEURAVOL_DATA=data\surfaces
set NEURAVOL_OUT=reports\nn_v3_quick
set MPLBACKEND=Agg
echo started %date% %time% > logs\v3_quick.log
python -u -c "src=open('notebooks/neuravol_v3.py', encoding='utf-8').read().replace('QUICK = False', 'QUICK = True', 1); exec(compile(src, 'v3', 'exec'))" >> logs\v3_quick.log 2>&1
echo finished %date% %time% exit %errorlevel% >> logs\v3_quick.log
