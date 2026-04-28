@echo off
rem Windows launcher for News-Engine.
rem Usage:  scripts\run.bat           (local only)
rem         scripts\run.bat --share   (+ Cloudflare tunnel public URL)

setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"

set SHARE=0
if "%1"=="--share" set SHARE=1

if not exist .venv (
  echo Creating virtualenv at .venv
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

if not exist .env (
  echo .env missing - copying .env.example. Edit it to add API keys.
  copy .env.example .env >nul
)

if not exist data mkdir data

set HOST=0.0.0.0
set PORT=8000
echo Starting server on http://%HOST%:%PORT%

if "%SHARE%"=="1" (
  where cloudflared >nul 2>&1
  if errorlevel 1 (
    echo cloudflared is not installed.
    echo   winget install --id Cloudflare.cloudflared
    echo   or see https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation
    echo Running locally only.
    uvicorn backend.main:app --host %HOST% --port %PORT%
  ) else (
    start "cloudflared" cmd /c "timeout /t 3 >nul && cloudflared tunnel --url http://localhost:%PORT% --no-autoupdate"
    uvicorn backend.main:app --host %HOST% --port %PORT%
  )
) else (
  uvicorn backend.main:app --host %HOST% --port %PORT%
)

endlocal
