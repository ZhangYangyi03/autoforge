@echo off
REM One command after clone: fetch the engine, install the substrate, run the demo.
REM
REM   run_demo.cmd
REM
REM `pip install -e .` pulls autoforge from git, so no sibling checkout is needed.
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo run_demo: python not found on PATH. Install Python 3.10+ and retry.
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo run_demo: need Python 3.10 or newer.
  python -V
  exit /b 1
)

where git >nul 2>&1
if errorlevel 1 (
  echo run_demo: git is required -- the engine is installed from a git URL.
  exit /b 1
)

echo run_demo: installing the substrate and its engine dependency...
python -m pip install --quiet -e .
if errorlevel 1 (
  echo run_demo: install failed, see output above.
  exit /b 1
)

echo run_demo: running the end-to-end evolution demo
python examples\demo_evolution.py
exit /b %errorlevel%
