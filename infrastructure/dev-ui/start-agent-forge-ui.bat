@echo off
echo ========================================
echo  Agent Forge Developer UI Startup
echo ========================================
echo.
echo Starting comprehensive Agent Forge Developer Interface...
echo - API Controllers for real phase execution
echo - Model chat interface for testing trained models
echo - Real-time progress monitoring with WebSocket
echo - System metrics dashboard
echo.

REM Set environment variables for Agent Forge
set PYTHONPATH=%PYTHONPATH%;%CD%\core
set AGENT_FORGE_DEV=true

REM Change to the integration directory
cd /d "%~dp0"

echo [INFO] Starting Agent Forge Developer UI Integration System...
python integration-setup.py

echo.
echo ========================================
echo  Agent Forge Developer UI Started!
echo ========================================
echo.
echo Access the developer interface at:
echo   🌐 Main Developer UI: http://localhost:8080
echo.
echo Individual API services:
echo   🤖 Agent Forge Controller: http://localhost:8083
echo   💬 Model Chat Interface: http://localhost:8084
echo   📡 WebSocket Manager: http://localhost:8085
echo.
echo Press Ctrl+C to stop all services
echo ========================================

pause
