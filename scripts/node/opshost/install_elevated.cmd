@echo off
rem  Right-click -> "Run as administrator".  Runs install_elevated.ps1 elevated.
rem  All output is teed to C:\Users\china\elevated_install.log so "did it work"
rem  is answerable afterwards from a non-elevated session.
title autoforge elevated setup
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\china\autoforge_node\install_elevated.ps1"
echo.
echo Done. Closing in 20s.
timeout /t 20 >nul
