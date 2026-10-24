@echo off
echo ========================================================
echo  Review Intelligence Platform - Full Vite Dev Launcher
echo ========================================================
echo.

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo [1/3] Checking Node.js and dependencies...
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js / npm is not installed or not in PATH.
    echo To run the Vite dev server, please install Node.js 18+.
    echo Alternatively, run run_demo.bat to launch without Node.js.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo node_modules not found. Running npm install...
    call npm install
)

echo [2/3] Starting FastAPI Backend on port 8000...
if exist ".venv\Scripts\activate.bat" (
    start "FastAPI Backend (Port 8000)" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn server.app:app --host 127.0.0.1 --port 8000"
) else (
    start "FastAPI Backend (Port 8000)" cmd /k "python -m uvicorn server.app:app --host 127.0.0.1 --port 8000"
)

echo Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak >nul

echo [3/3] Starting Vite Frontend on port 5173...
start http://localhost:5173
call npm run dev

pause
