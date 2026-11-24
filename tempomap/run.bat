@echo off
rem Map a song by hand: double-click, type a few words of its title. Or: run.bat fear inoculum
rem (In Re-Tex: right-click a song - "Tempo map".) Reads the library; writes only to maps\ and cache\ here.
setlocal
cd /d "%~dp0"
set PY=..\.venv-phase2\Scripts\python.exe
if not exist "%PY%" (
  echo Re-Tex's Python isn't at %PY% - TempoMap uses it.
  pause
  exit /b 1
)
"%PY%" -m tempomap %*
if errorlevel 1 pause
