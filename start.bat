@echo off
REM ── Arogya AI: start backend (:8000) + frontend (:5173) in one shot ──
cd /d "%~dp0"

echo Starting Arogya AI...
start "Arogya Backend"  cmd /k "cd /d "%~dp0backend" && venv\Scripts\python -m uvicorn app.main:app --port 8001"
start "Arogya Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo   Backend : http://localhost:8001/api/health
echo   Frontend: http://localhost:5173/
echo.
echo Two terminal windows opened. Close them to stop the servers.
