@echo off
echo ============================================================
echo 🔧 ACTIVATING CONDA RVC-ENV ENVIRONMENT
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda not found in PATH!
    echo 🔧 Please make sure Anaconda/Miniconda is installed and added to PATH.
    pause
    exit /b 1
)

REM Check if the rvc-env environment exists
conda env list | findstr /C:"rvc-env" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda environment 'rvc-env' not found!
    echo 🔧 Please create the environment first with: conda create -n rvc-env python=3.10
    pause
    exit /b 1
)

REM Activate the conda environment
call conda activate rvc-env
if %errorlevel% neq 0 (
    echo ❌ Failed to activate rvc-env environment!
    pause
    exit /b 1
)

echo ✅ Environment activated! Python version:
python --version
echo.

echo ============================================================
echo 🚀 RUNNING MAIN.PY SCRIPT
echo ============================================================
echo.

REM Run the Python script
python "C:\Users\ROG\Downloads\New folder (4)\main.py"

REM Check if the script ran successfully
if %errorlevel% neq 0 (
    echo ❌ Script execution failed!
    echo Error code: %errorlevel%
) else (
    echo ✅ Script completed successfully!
)

echo.
echo Press any key to exit...
pause >nul