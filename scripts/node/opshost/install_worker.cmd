@echo off
rem  One more click: installs the elevated worker so FUTURE privileged actions
rem  need no prompt.  Right-click -> Run as administrator.
title autoforge elevated worker
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\china\autoforge_node\install_worker.ps1"
echo.
echo Done. Closing in 25s.
timeout /t 25 >nul
