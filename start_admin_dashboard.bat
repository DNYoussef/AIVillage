@echo off
echo Starting AIVillage Admin Dashboard...
echo Dashboard will be available at: http://localhost:3006
echo Press Ctrl+C to stop the server
echo --------------------------------------------------

cd /d "%~dp0"
python scripts\start_admin_dashboard.py

pause
