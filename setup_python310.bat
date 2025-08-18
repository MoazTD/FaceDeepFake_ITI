@echo off
echo ============================================================
echo 🐍 PYTHON 3.10 SETUP FOR FACE SWAP APPLICATION  
echo ============================================================
echo.

REM Check if Python 3.10 is available
echo 🔍 Checking for Python 3.10...
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3.10 is not available
    echo.
    echo 📥 Please install Python 3.10:
    echo    1. Visit: https://www.python.org/downloads/release/python-3108/
    echo    2. Download and install Python 3.10.8
    echo    3. Run this script again
    echo.
    echo 🌐 Opening download page...
    start https://www.python.org/downloads/release/python-3108/
    goto :end
)

echo ✅ Python 3.10 is available!
echo.
echo 🔧 Creating virtual environment with Python 3.10...
py -3.10 -m venv faceswap_env_py310
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    goto :end
)

echo ✅ Virtual environment created successfully!
echo.
echo 🎉 Setup complete! Use activate_python310.bat to start.

:end
pause