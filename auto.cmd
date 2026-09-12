@echo off
rem autoforge - one-command launcher for Windows.
rem
rem Straight from a fresh clone, no install step:
rem
rem     git clone https://github.com/<you>/autoforge && cd autoforge && auto
rem
rem (PowerShell needs the explicit path: .\auto)
rem
rem Want `auto` on your PATH everywhere instead? Then:  pip install -e .
setlocal
cd /d "%~dp0"

set "PY="
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo auto: no python on PATH - install Python 3.10+ first 1>&2
    exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo auto: need Python 3.10+ 1>&2
    %PY% --version 1>&2
    exit /b 1
)

rem requests is the single runtime dependency; fetch it once if it is missing.
%PY% -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo auto: installing the one dependency ^(requests^)... 1>&2
    %PY% -m pip install --quiet requests
)

%PY% -m autoforge.cli %*
exit /b %ERRORLEVEL%
