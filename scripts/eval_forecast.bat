@echo off
REM Stand up sub-component (a) — forecasting models — on toy + optional real data.
pushd "%~dp0..\"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m backend.forecast.run_eval %*
popd
