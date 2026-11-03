@echo off
cd /d "%~dp0"
echo Starting NeuraVol dashboard...
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
