@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo [PaperMind] Checking local environment...
where python >nul 2>nul || (echo [ERROR] Python was not found. Install Python 3.11+ and retry.& exit /b 1)
where node >nul 2>nul || (echo [ERROR] Node.js was not found. Install Node.js 18+ and retry.& exit /b 1)
where npm >nul 2>nul || (echo [ERROR] npm was not found. Check your Node.js installation.& exit /b 1)
if not exist ".env" echo [WARN] .env is missing. Copy .env.example to .env and configure the backend key.
if not exist "web\node_modules" (
  echo [PaperMind] Installing web dependencies for first run...
  pushd web
  call npm install
  if errorlevel 1 (popd & echo [ERROR] npm install failed.& exit /b 1)
  popd
)
echo [PaperMind] Starting FastAPI on http://127.0.0.1:8000 ...
start "PaperMind API" /D "%~dp0" cmd /k "set PYTHONPATH=%~dp0src&& python -m uvicorn server.main:app --host 127.0.0.1 --port 8000"
echo [PaperMind] Waiting for API health check...
set API_READY=
for /L %%i in (1,1,30) do (
  powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {} ; exit 1" >nul 2>nul
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
  powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173 -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {} ; exit 1" >nul 2>nul
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
