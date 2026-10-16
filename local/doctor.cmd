@echo off
rem Audit the installed autoforge build against its install manifest.
rem
rem Deliberately outside the venv and outside the autoforge package. A stock
rem `pip install autoforge-agent==0.4.2` deletes everything under both of
rem them, so any checker that lives there is uninstalled by the exact event it
rem exists to report -- and says nothing at the moment it matters most.
rem
rem   doctor              report, exit 1 if the install has drifted
rem   doctor --repair     quarantine pip debris, reinstall from the local wheel
rem   doctor --as-json    machine-readable result
setlocal

set DOCTOR=%LOCALAPPDATA%\autoforge\install_doctor.py
if not exist "%DOCTOR%" set DOCTOR=%USERPROFILE%\.autoforge\install_doctor.py
if not exist "%DOCTOR%" (
  echo no install_doctor.py found under:
  echo   %LOCALAPPDATA%\autoforge
  echo   %USERPROFILE%\.autoforge
  echo copy it from autoforge-interrupt-fix\install_doctor.py and the manifest
  echo next to it, then re-run.
  exit /b 1
)

if exist "C:\autoforge\venv\Scripts\python.exe" (
  "C:\autoforge\venv\Scripts\python.exe" "%DOCTOR%" %*
) else (
  python "%DOCTOR%" %*
)
exit /b %ERRORLEVEL%
