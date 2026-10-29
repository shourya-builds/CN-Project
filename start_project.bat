@echo off
setlocal
if not exist ".venv\Scripts\python.exe" (
  echo Creating project environment...
  py -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
streamlit run app.py
