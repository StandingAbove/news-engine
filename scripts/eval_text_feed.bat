@echo off
REM Stand up sub-component (b) — text feed — on toy headlines.
pushd "%~dp0..\"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m backend.news.toy_eval %*
popd
