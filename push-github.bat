@echo off
cd /d "%~dp0"

git add .

git diff --cached --quiet
if %errorlevel%==0 (
    echo No changes to push.
    pause
    exit /b
)

git commit -m "Update project"
git push origin main

pause
