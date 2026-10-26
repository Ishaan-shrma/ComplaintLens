@echo off
setlocal enabledelayedexpansion
title ComplaintLens - Launcher

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ======================================================================
echo   ComplaintLens - Setup and Live Launcher
echo ======================================================================
echo.

:: 1. Copy .env.example to .env if .env does not exist
if not exist ".env" (
    if exist ".env.example" (
        echo [Setup] Initializing default configuration: copying .env.example to .env ...
        copy ".env.example" ".env" >nul
    )
)

:: 2. Check Python installation
echo [1/4] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from python.org.
    echo (Note: Python 3.14 lacks official pre-built PyTorch binary wheels).
    pause
    exit /b 1
)

:: 3. Python Virtual Environment (.venv)
if not exist ".venv\Scripts\activate.bat" (
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [Setup] Installing Python dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to install all requirements. Continuing...
    )
) else (
    echo [2/4] Using existing virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

:: 4. Node / npm packages
echo [3/4] Checking Node.js and frontend dependencies...
if not exist "node_modules" (
    call npm --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [Setup] Installing frontend node_modules...
        call npm install
    ) else (
        echo [Notice] npm not detected in PATH; skipping npm install. Pre-built dist/ will be served if present.
    )
)

:: 5. Launch Servers
echo [4/4] Launching services...
echo   - Backend API: http://127.0.0.1:8000
echo   - Frontend UI: http://localhost:5173 (or http://127.0.0.1:8000)
echo.

:: Launch FastAPI backend in background window
start "Review Intelligence API" cmd /c "cd /d "%PROJECT_DIR%" && call .venv\Scripts\activate.bat && python -m uvicorn server.app:app --host 127.0.0.1 --port 8000"

:: Launch Vite frontend if npm is available
call npm --version >nul 2>&1
if %errorlevel% equ 0 (
    start "Review Intelligence UI" cmd /c "cd /d "%PROJECT_DIR%" && npm run dev"
    start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5173"
) else (
    start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"
)

echo Prototype launched successfully. Press any key to close this launcher window.
pause >nul
