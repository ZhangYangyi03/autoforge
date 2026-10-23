@echo off
rem ===========================================================================
rem  autoforge-launcher.cmd -- start the WorkBuddy proxy, then autoforge.
rem
rem  Double-click this from the desktop. It:
rem    1. checks whether wbproxy is already listening on 8777
rem    2. starts it in a HIDDEN window if not (no taskbar button, no focus)
rem    3. waits until it answers /healthz
rem    4. drops you into the autoforge REPL
rem
rem  Nothing is installed and no PATH is modified.
rem
rem  To stop the proxy afterwards:
rem     taskkill /F /IM python.exe /FI "WINDOWTITLE eq wbproxy*"
rem  or close it from the Task Manager -- it runs as a plain python.exe.
rem
rem  The two paths below are the only machine-specific parts. Edit them if the
rem  proxy or the checkout moves; nothing else here is location-dependent.
rem ===========================================================================

setlocal
set "PROXY=%~dp0..\wbproxy.py"
set "PROJ=%~dp0..\..\.."
set "PORT=8777"
set "URL=http://127.0.0.1:%PORT%/healthz"

echo.
echo === autoforge launcher =========================================
echo.

rem --- is the proxy already up? -------------------------------------------
netstat -ano | findstr /R /C:"TCP.*:%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ok] wbproxy already listening on %PORT%
    goto :ready
)

rem --- start it, fully hidden ----------------------------------------------
if not exist "%PROXY%" (
    echo [!] proxy not found: %PROXY%
    echo     autoforge will start but every model call will fail.
    goto :ready
)

echo [..] starting wbproxy ...
rem `-WindowStyle Hidden` gives a process with no window at all
rem (MainWindowHandle 0), unlike `start /MIN` which leaves a taskbar button.
rem
rem Output is redirected to a log file: a hidden process that fails on startup
rem would otherwise be unreadable, and "it just doesn't work" is the worst
rem failure mode. Overwritten each start, so it cannot grow without bound.
powershell -NoProfile -Command "Start-Process python -ArgumentList '-X','utf8','%PROXY%' -WindowStyle Hidden -RedirectStandardOutput '%TEMP%\wbproxy.log' -RedirectStandardError '%TEMP%\wbproxy.err'"

rem --- wait for it to answer (up to 40s) ----------------------------------
set /a TRIES=0
:wait
set /a TRIES+=1
if %TRIES% GTR 20 goto :timeout
powershell -NoProfile -Command "try{(Invoke-WebRequest -Uri '%URL%' -TimeoutSec 3 -UseBasicParsing).StatusCode}catch{exit 1}" >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto :wait
)
echo [ok] wbproxy is up on %PORT%
goto :ready

:timeout
echo [!] wbproxy did not answer within 40s.
echo     Check %TEMP%\wbproxy.err, or run it by hand:
echo         python -X utf8 "%PROXY%"
echo.
echo     Continuing into autoforge anyway...
echo.

:ready
rem --- autoforge -----------------------------------------------------------
if not exist "%PROJ%\pyproject.toml" (
    echo [!] autoforge not found: %PROJ%
    pause
    exit /b 1
)

cd /d "%PROJ%"
echo.
echo === entering autoforge ==========================================
echo   model is set in %USERPROFILE%\.autoforge\config.json
echo   change it there, or list what the proxy offers:
echo       python "%PROXY%" models --free
echo.
python -X utf8 -m autoforge.cli chat

echo.
echo === autoforge exited ============================================
pause
endlocal
