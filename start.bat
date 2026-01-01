@echo off
REM Archetype - Character Consistency Validator
REM Startup script for development (Windows)

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set BACKEND_DIR=%SCRIPT_DIR%backend
set FRONTEND_DIR=%SCRIPT_DIR%frontend

echo 🚀 Starting Archetype...
echo.

REM Backend setup
echo 📦 Setting up Python backend...
cd /d "%BACKEND_DIR%"

REM Create venv if it doesn't exist
if not exist ".venv" (
    echo    Creating Python virtual environment...
    python -m venv .venv
)

REM Activate venv
echo    Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo    Installing/updating dependencies...
pip install -q -r requirements.txt

REM Start backend in new window
echo    Starting FastAPI server on http://localhost:8000...
start "Archetype Backend" cmd /c "cd /d %BACKEND_DIR% && .venv\Scripts\activate.bat && uvicorn main:app --reload --port 8000"

REM Frontend setup
echo.
echo 📦 Setting up React frontend...
cd /d "%FRONTEND_DIR%"

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo    Installing npm dependencies...
    call npm install
)

REM Start frontend in new window
echo    Starting Vite dev server on http://localhost:5173...
start "Archetype Frontend" cmd /c "cd /d %FRONTEND_DIR% && npm run dev"

echo.
echo ✅ Archetype is running!
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:8000
echo.
echo Close the terminal windows to stop the services.

endlocal
