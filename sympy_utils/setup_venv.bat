@echo off
REM Create and install sympy into local virtual environment (Windows cmd)
REM Usage: In cmd, cd into the sympy_utils directory and run:
REM   call setup_venv.bat

set VENV_DIR=%~dp0venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [INFO] Creating virtual environment: %VENV_DIR%
  py -3 -m venv "%VENV_DIR%"
) else (
  echo [INFO] Virtual environment already exists: %VENV_DIR%
)

echo [INFO] Activating virtual environment and upgrading pip...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip

if exist "%~dp0requirements.txt" (
  echo [INFO] Installing dependencies from requirements.txt...
  python -m pip install -r "%~dp0requirements.txt"
) else (
  echo [INFO] requirements.txt not found, installing sympy directly...
  python -m pip install sympy
)

echo.
echo [DONE] Installation complete.
echo Usage:
echo   call "%VENV_DIR%\Scripts\activate.bat"
echo To exit:
echo   deactivate
