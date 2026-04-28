@echo off
REM Run tests using the venv's Python directly (avoids conda/base PATH clash).
pushd "%~dp0..\"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Create one with:
  echo     python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
  popd
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest %*
popd
