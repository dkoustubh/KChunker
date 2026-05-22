@echo off
title KChunker Dashboard Launcher (Windows)
cd /d "%~dp0"
cls
echo ====================================================
echo             KCHUNKER GUI AUTO-LAUNCHER             
echo ====================================================
echo.
echo Drag-and-drop a file here (or type the file address),
echo then press Enter to launch the dashboard and auto-ingest:
echo.
set /p filepath="File path (press Enter to skip): "

:: Remove quotes if they exist around the dragged file path
if defined filepath (
    set filepath=%filepath:"=%
)

if "%filepath%"=="" (
    echo.
    echo Launching GUI dashboard...
    echo.
    set PYTHONPATH=.
    .venv\Scripts\python gui.py
) else (
    echo.
    echo Starting GUI dashboard with auto-ingestion for:
    echo   %filepath%
    echo.
    set PYTHONPATH=.
    .venv\Scripts\python gui.py --file "%filepath%"
)
pause
