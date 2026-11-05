@echo off
rem Double-click to start Re-Tex (it opens in your browser). Close this window to quit.
rem First time on this PC? Run install.bat first.
cd /d "%~dp0"
title Re-Tex
if not exist ".venv-phase2\Scripts\python.exe" (
  echo Re-Tex isn't installed on this PC yet: double-click install.bat first.
  pause
  exit /b 1
)
".venv-phase2\Scripts\python.exe" -m mashlab %*
if errorlevel 1 pause
