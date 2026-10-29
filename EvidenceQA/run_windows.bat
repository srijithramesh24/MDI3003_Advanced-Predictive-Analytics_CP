@echo off
cd /d "%~dp0"
echo EvidenceQA setup and launch
py -3.12 --version
if errorlevel 1 (
  echo Python 3.12 not found. Install it from https://www.python.org/downloads/release/python-31210/
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
pause
