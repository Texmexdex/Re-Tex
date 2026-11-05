@echo off
rem Re-Tex installer (Windows 10/11, NVIDIA GPU). Double-click it. Safe to run again: it skips what's done.
rem Options:  install.bat -Mode Full | -Mode Core   (skip the question)
cd /d "%~dp0"
title Re-Tex installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install.ps1" %*
echo.
pause
