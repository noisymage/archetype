#!/bin/bash

# Archetype - Character Consistency Validator
# Startup script for development (macOS/Linux)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "🚀 Starting Archetype..."
echo ""

# Backend setup
echo "📦 Setting up Python backend..."
cd "$BACKEND_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "   Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv and install dependencies
echo "   Activating virtual environment..."
source .venv/bin/activate

echo "   Installing/updating dependencies..."
pip install -q -r requirements.txt

# Start backend in background
echo "   Starting FastAPI server on http://localhost:8000..."
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Frontend setup
echo ""
echo "📦 Setting up React frontend..."
cd "$FRONTEND_DIR"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "   Installing npm dependencies..."
    npm install
fi

# Start frontend
echo "   Starting Vite dev server on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Archetype is running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "   Goodbye!"
}

trap cleanup EXIT

# Wait for either process to exit
wait
