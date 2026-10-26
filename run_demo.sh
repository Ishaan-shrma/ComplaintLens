#!/usr/bin/env bash
# ==============================================================================
# ComplaintLens - Setup & Live Launcher (macOS / Linux)
# ==============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Check Python installation and version (Python 3.10-3.12 required)
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ComplaintLens needs Python 3.10-3.12 (found none)"
    exit 1
fi

PY_VER=$("$PYTHON_BIN" -c "import sys; print(sys.version.split()[0])" 2>/dev/null || echo "unknown")
if ! "$PYTHON_BIN" -c "import sys; v=sys.version_info; sys.exit(0 if (v.major == 3 and 10 <= v.minor <= 12) else 1)" 2>/dev/null; then
    echo "ComplaintLens needs Python 3.10-3.12 (found $PY_VER)"
    exit 1
fi

# 2. Check Node.js installation (Node >=18 required)
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed or not in PATH."
    echo "ComplaintLens needs Node.js >=18 to run the frontend UI."
    echo "Please install Node.js from https://nodejs.org/ and try again."
    exit 1
fi

echo "======================================================================"
echo "  ComplaintLens - Setup and Live Launcher"
echo "======================================================================"
echo ""

# 3. Copy .env.example to .env if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "[Setup] Copying .env.example to .env ..."
    cp .env.example .env
fi

# 4. Virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
    source .venv/bin/activate
    echo "[Setup] Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt || echo "[Warning] Pip install completed with warnings."
else
    echo "[1/3] Using existing virtual environment (.venv)..."
    source .venv/bin/activate
fi

# 5. Node / npm packages
echo "[2/3] Checking frontend dependencies..."
if [ ! -d "node_modules" ]; then
    if [ -f "package-lock.json" ]; then
        echo "[Setup] Installing frontend dependencies via npm ci..."
        npm ci
    else
        echo "[Setup] Installing frontend dependencies via npm install..."
        npm install
    fi
fi

# 6. Launch Backend & Frontend
echo "[3/3] Starting services..."
echo "  - Backend API: http://127.0.0.1:8000"
echo "  - Frontend UI: http://localhost:5173"
echo ""

# Trap to kill background processes on script exit (Ctrl+C)
cleanup() {
    echo ""
    echo "Shutting down ComplaintLens services..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start backend
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Start frontend
npm run dev &
FRONTEND_PID=$!

sleep 2
echo "Services running. Press Ctrl+C to terminate."
wait
