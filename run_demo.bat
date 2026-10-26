@echo off
setlocal enabledelayedexpansion
title ComplaintLens - Launcher

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: 1. Check Python installation and version (Python 3.10-3.12 required)
where python >nul 2>&1
if errorlevel 1 (
    echo ComplaintLens needs Python 3.10-3.12 (found none^)
    exit /b 1
)

set "PY_VER=unknown"
for /f %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"

python -c "import sys; v=sys.version_info; sys.exit(0 if (v.major==3 and 10<=v.minor<=12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ComplaintLens needs Python 3.10-3.12 (found !PY_VER!^)
    exit /b 1
)

:: 2. Check Node.js installation (Node >=18 required)
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo ComplaintLens needs Node.js ^>=18 to run the frontend UI.
    echo Please install Node.js from https://nodejs.org/ and try again.
    exit /b 1
)

echo ======================================================================
echo   ComplaintLens - Setup and Live Launcher
echo ======================================================================
echo.

:: 3. Copy .env.example to .env if .env does not exist
if not exist ".env" (
    if exist ".env.example" (
        echo [Setup] Initializing default configuration: copying .env.example to .env ...
        copy ".env.example" ".env" >nul
    )
)

:: 4. Python Virtual Environment (.venv)
if not exist ".venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [Setup] Installing Python dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] Failed to install all requirements. Continuing...
    )
) else (
    echo [1/3] Using existing virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

:: 5. Node / npm packages
echo [2/3] Checking frontend dependencies...
if not exist "node_modules" (
    if exist "package-lock.json" (
        echo [Setup] Installing frontend dependencies via npm ci...
        call npm ci
    ) else (
        echo [Setup] Installing frontend dependencies via npm install...
        call npm install
    )
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies.
        exit /b 1
    )
)

:: 6. Launch Services
echo [3/3] Launching services...
echo   - Backend API: http://127.0.0.1:8000
echo   - Frontend UI: http://localhost:5173
echo.

:: Launch FastAPI backend in background window
start "ComplaintLens API" cmd /c "cd /d "%PROJECT_DIR%" && call .venv\Scripts\activate.bat && python -m uvicorn server.app:app --host 127.0.0.1 --port 8000"

:: Launch Vite frontend
start "ComplaintLens UI" cmd /c "cd /d "%PROJECT_DIR%" && npm run dev"
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5173"

echo ComplaintLens launched successfully. Press any key to close this launcher window.
pause >nul
