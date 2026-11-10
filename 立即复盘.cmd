@echo off
rem Double-click: run the optimal-combination review now (trading days after 15:10), show progress, open the page.
rem Extra arguments go to run_review.ps1, for example:  -Date 2026-10-14
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_review.ps1" -Show %*
echo.
pause
