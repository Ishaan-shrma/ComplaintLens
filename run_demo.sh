#!/usr/bin/env bash
# ==============================================================================
# Review Intelligence Prototype - Setup & Live Launcher (macOS / Linux)
# ==============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "  Review Intelligence Prototype - Setup and Live Launcher"
echo "======================================================================"
echo ""

# 1. Copy .env.example to .env if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "[Setup] Copying .env.example to .env ..."
    cp .env.example .env
fi

# 2. Check Python installation
echo "[1/4] Checking Python environment..."
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10, 3.11, or 3.12 (Python 3.14 lacks official PyTorch wheels)."
    exit 1
fi

"$PYTHON_BIN" --version

# 3. Virtual environment
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
    source .venv/bin/activate
    echo "[Setup] Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt || echo "[Warning] Pip install completed with warnings."
else
    echo "[2/4] Using existing virtual environment (.venv)..."
    source .venv/bin/activate
fi

# 4. Node / npm packages
echo "[3/4] Checking Node.js frontend dependencies..."
if [ ! -d "node_modules" ]; then
    if command -v npm &> /dev/null; then
        echo "[Setup] Installing npm packages..."
        npm install
    else
        echo "[Notice] npm not found; skipping. Pre-built dist/ will be served if present."
    fi
fi

# 5. Launch Backend & Frontend
echo "[4/4] Starting services..."
echo "  - Backend API: http://127.0.0.1:8000"
echo "  - Frontend UI: http://localhost:5173"
echo ""

# Trap to kill background processes on script exit (Ctrl+C)
cleanup() {
    echo ""
    echo "Shutting down Review Intelligence Prototype services..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start backend
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Start frontend if npm is available
if command -v npm &> /dev/null; then
    npm run dev &
    FRONTEND_PID=$!
fi

sleep 2
echo "Services running. Press Ctrl+C to terminate."
wait
