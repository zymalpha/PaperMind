@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo [PaperMind] Checking local environment...
where python >nul 2>nul || (echo [ERROR] Python was not found. Install Python 3.11+ and retry.& exit /b 1)
where node >nul 2>nul || (echo [ERROR] Node.js was not found. Install Node.js 18+ and retry.& exit /b 1)
where npm >nul 2>nul || (echo [ERROR] npm was not found. Check your Node.js installation.& exit /b 1)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (echo [ERROR] Python 3.11 or newer is required.& exit /b 1)
node -e "process.exit(Number(Number(process.versions.node.split('.')[0]) < 18))"
if errorlevel 1 (echo [ERROR] Node.js 18 or newer is required.& exit /b 1)
python -c "import fastapi, uvicorn"
if errorlevel 1 (echo [ERROR] Python dependencies are missing. Run: python -m pip install -r requirements.txt& exit /b 1)
if not exist ".env" echo [WARN] .env is missing. Copy .env.example to .env and configure the backend key.
if not exist "web\node_modules" (
  echo [PaperMind] Installing web dependencies for first run...
  pushd web
  if exist package-lock.json (call npm ci) else (call npm install)
  if errorlevel 1 (popd & echo [ERROR] npm dependency installation failed.& exit /b 1)
  popd
)
echo [PaperMind] Checking required ports...
powershell -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>nul
if errorlevel 1 (echo [ERROR] Port 8000 is already in use. Stop the conflicting service before starting PaperMind.& exit /b 1)
powershell -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort 5173 -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>nul
if errorlevel 1 (echo [ERROR] Port 5173 is already in use. Stop the conflicting service before starting PaperMind.& exit /b 1)
echo [PaperMind] Starting FastAPI on http://127.0.0.1:8000 ...
start "PaperMind API" /D "%~dp0" cmd /k "set PYTHONPATH=%~dp0src&& python -m uvicorn server.main:app --host 127.0.0.1 --port 8000"
echo [PaperMind] Waiting for API health check...
set API_READY=
for /L %%i in (1,1,30) do (
  powershell -NoProfile -Command "try { $r=Invoke-RestMethod http://127.0.0.1:8000/api/health -TimeoutSec 1; if ($r.status -eq 'ok' -and $r.model) { exit 0 } } catch {} ; exit 1" >nul 2>nul
  if not errorlevel 1 (set API_READY=1& goto :api_ready)
  timeout /t 1 /nobreak >nul
)
:api_ready
if not defined API_READY (
  echo [ERROR] FastAPI did not become healthy. Check the PaperMind API window.
  exit /b 1
)
echo [PaperMind] Starting React on http://127.0.0.1:5173 ...
start "PaperMind Web" /D "%~dp0web" cmd /k "npm run dev"
echo [PaperMind] Waiting for web frontend...
set WEB_READY=
for /L %%i in (1,1,20) do (
  powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173 -TimeoutSec 1; if ($r.StatusCode -eq 200 -and $r.Content.Contains('PaperMind')) { exit 0 } } catch {} ; exit 1" >nul 2>nul
  if not errorlevel 1 (set WEB_READY=1& goto :web_ready)
  timeout /t 1 /nobreak >nul
)
:web_ready
if not defined WEB_READY (
  echo [ERROR] React frontend did not become available. Check the PaperMind Web window.
  exit /b 1
)
start "" http://127.0.0.1:5173
echo.
echo [PaperMind] Ready. Browser: http://127.0.0.1:5173  API: http://127.0.0.1:8000
echo [PaperMind] Use stop_web.bat to stop both services.
endlocal
