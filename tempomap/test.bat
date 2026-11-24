@echo off
rem Makes four test songs with a known tempo map (in a temp folder), maps them, and checks the result.
setlocal
cd /d "%~dp0"
"..\.venv-phase2\Scripts\python.exe" tests\run_tests.py
pause
