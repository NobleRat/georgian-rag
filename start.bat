@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Creating Python environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Python 3.11+ is required: https://www.python.org/downloads/
    pause
    exit /b 1
  )
  echo [2/2] Installing packages...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created .env - open it in Notepad and set OPENAI_API_KEY.
)

echo Starting... the browser will open at http://127.0.0.1:8765
".venv\Scripts\python.exe" -m app.server
pause
