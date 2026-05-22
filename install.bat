@echo off
title KChunker Installer (Windows)
cd /d "%~dp0"
cls
echo ====================================================
echo             KCHUNKER INSTALLER (Windows)           
echo ====================================================
echo.

:: Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing uv package manager...
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
) else (
    echo uv package manager is already installed.
)

echo Synchronizing project dependencies...
uv sync

echo.
echo ====================================================
echo Installation complete! You can now run KChunker.
echo ====================================================
pause
