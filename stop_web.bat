@echo off
echo [PaperMind] Stopping local web services...
taskkill /FI "WINDOWTITLE eq PaperMind API*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq PaperMind Web*" /T /F >nul 2>nul
echo [PaperMind] Services stopped. (Existing unrelated Python/Node processes were not targeted.)
