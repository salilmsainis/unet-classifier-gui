@echo off
setlocal
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
cd /d "%APP_DIR%" || goto :directory_error

if exist "%VENV_PYTHON%" goto :check_environment

set "BOOTSTRAP_PYTHON="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.12"
)
if defined BOOTSTRAP_PYTHON goto :offer_create

where python >nul 2>nul
if errorlevel 1 goto :python_missing
python -c "import sys; raise SystemExit(sys.version_info < (3, 12))" >nul 2>nul
if errorlevel 1 goto :python_missing
set "BOOTSTRAP_PYTHON=python"

:offer_create
echo A local Python environment is needed in:
echo   %VENV_DIR%
echo.
set "ANSWER="
set /p "ANSWER=Create it and install the application now? [Y/n] "
if /i "%ANSWER%"=="n" goto :cancelled
if /i "%ANSWER%"=="no" goto :cancelled
%BOOTSTRAP_PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_error
call :install_project
if errorlevel 1 goto :install_error
goto :launch

:check_environment
"%VENV_PYTHON%" -c "import sys; raise SystemExit(sys.version_info < (3, 12))" >nul 2>nul
if errorlevel 1 goto :old_environment
"%VENV_PYTHON%" "%APP_DIR%train.py" --check >nul 2>nul
if not errorlevel 1 goto :launch

echo The environment is incomplete or out of date.
set "ANSWER="
set /p "ANSWER=Install or repair the application now? [Y/n] "
if /i "%ANSWER%"=="n" goto :cancelled
if /i "%ANSWER%"=="no" goto :cancelled
call :install_project
if errorlevel 1 goto :install_error

:launch
"%VENV_PYTHON%" "%APP_DIR%train.py" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" exit /b 0
echo.
echo U-Net classifier exited with code %RESULT%. See the error above and README.md.
pause
exit /b %RESULT%

:install_project
echo.
echo Installing the U-Net classifier and its Python dependencies...
"%VENV_PYTHON%" -m pip install -e "%APP_DIR%"
exit /b %ERRORLEVEL%

:python_missing
echo Python 3.12 or newer was not found. Install Python 3.12, then run this launcher again.
goto :failure

:old_environment
echo The existing .venv uses Python older than 3.12. Remove .venv and run this launcher again.
goto :failure

:directory_error
echo The application folder could not be opened.
goto :failure

:venv_error
echo Could not create the Python environment.
goto :failure

:install_error
echo Installation failed. See the error above and README.md.
goto :failure

:cancelled
echo Installation cancelled.

:failure
echo.
pause
exit /b 1
