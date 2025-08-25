#!/usr/bin/env python3
"""
Simple Gateway Server - Minimal Dependencies
Serves the admin interface and provides basic routing.
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Forge Gateway")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if ui directory exists
ui_path = Path(__file__).parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path)), name="ui")


@app.get("/")
async def root():
    """Root endpoint - redirect to admin interface."""
    return {
        "service": "Agent Forge Gateway",
        "version": "1.0.0",
        "admin_interface": "/admin_interface.html",
        "api_endpoints": {"agent_forge": "http://localhost:8083", "websocket": "ws://localhost:8085/ws"},
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/admin_interface.html")
async def admin_interface():
    """Serve the admin interface HTML."""
    html_file = Path(__file__).parent / "admin_interface.html"

    if html_file.exists():
        return FileResponse(str(html_file))
    else:
        # Return a simple HTML interface if file doesn't exist
        return HTMLResponse(
            content="""
<!DOCTYPE html>
<html>
<head>
    <title>Agent Forge Admin Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; }
        .status { padding: 10px; margin: 10px 0; background: #e8f4f8; border-radius: 4px; }
        button {
            background: #007bff; color: white; border: none;
            padding: 10px 20px; border-radius: 4px; cursor: pointer;
            font-size: 16px; margin: 5px;
        }
        button:hover { background: #0056b3; }
        .progress {
            width: 100%; height: 30px; background: #e0e0e0;
            border-radius: 4px; overflow: hidden; margin: 10px 0;
        }
        .progress-bar {
            height: 100%; background: #4CAF50; width: 0%;
            transition: width 0.3s; display: flex; align-items: center;
            justify-content: center; color: white;
        }
        .models { margin-top: 20px; }
        .model-card {
            border: 1px solid #ddd; padding: 10px; margin: 10px 0;
            border-radius: 4px; background: #f9f9f9;
        }
        #log {
            background: #1e1e1e; color: #0f0; padding: 10px;
            border-radius: 4px; height: 200px; overflow-y: auto;
            font-family: monospace; font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Agent Forge Control Center</h1>

        <div class="status">
            <h2>System Status</h2>
            <p>API: <span id="api-status">Checking...</span></p>
            <p>WebSocket: <span id="ws-status">Disconnected</span></p>
        </div>

        <div class="controls">
            <h2>Cognate Model Creation</h2>
            <button onclick="startCognate()">🧠 Create 3 Cognate Models (25M params)</button>
            <div class="progress">
                <div class="progress-bar" id="progress-bar">0%</div>
            </div>
            <div id="status-message">Ready to create models</div>
        </div>

        <div class="models">
            <h2>Created Models</h2>
            <div id="models-list">No models created yet</div>
        </div>

        <div class="log-section">
            <h2>Activity Log</h2>
            <div id="log"></div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8083';
        const WS_URL = 'ws://localhost:8085/ws';
        let ws = null;

        function log(message) {
            const logEl = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            logEl.innerHTML = `[${time}] ${message}<br>` + logEl.innerHTML;
        }

        async function checkAPIStatus() {
            try {
                const response = await fetch(API_URL + '/health');
                if (response.ok) {
                    document.getElementById('api-status').innerHTML = '✅ Online';
                } else {
                    document.getElementById('api-status').innerHTML = '❌ Offline';
                }
            } catch (error) {
                document.getElementById('api-status').innerHTML = '❌ Offline';
            }
        }

        function connectWebSocket() {
            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                document.getElementById('ws-status').innerHTML = '✅ Connected';
                log('WebSocket connected');
                ws.send(JSON.stringify({ type: 'subscribe', channel: 'phases' }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'phase_update') {
                    updateProgress(data.progress * 100, data.message);
                }
            };

            ws.onclose = () => {
                document.getElementById('ws-status').innerHTML = '❌ Disconnected';
                log('WebSocket disconnected');
                setTimeout(connectWebSocket, 3000);
            };
        }

        function updateProgress(percent, message) {
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-bar').innerHTML = Math.round(percent) + '%';
            document.getElementById('status-message').innerHTML = message;
        }

        async function startCognate() {
            log('Starting Cognate model creation...');
            updateProgress(0, 'Initializing...');

            try {
                const response = await fetch(API_URL + '/phases/cognate/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const data = await response.json();
                log('Cognate creation started: ' + data.message);

                // Poll for updates
                pollPhaseStatus();

            } catch (error) {
                log('Error: ' + error.message);
                updateProgress(0, 'Error starting Cognate phase');
            }
        }

        async function pollPhaseStatus() {
            try {
                const response = await fetch(API_URL + '/phases/status');
                const data = await response.json();

                const cognatePhase = data.phases.find(p => p.phase_name === 'Cognate');
                if (cognatePhase) {
                    updateProgress(cognatePhase.progress * 100, cognatePhase.message);

                    if (cognatePhase.status === 'completed') {
                        log('✅ Cognate models created successfully!');
                        loadModels();
                    } else if (cognatePhase.status === 'running') {
                        setTimeout(pollPhaseStatus, 1000);
                    }
                }
            } catch (error) {
                log('Error polling status: ' + error.message);
            }
        }

        async function loadModels() {
            try {
                const response = await fetch(API_URL + '/models');
                const data = await response.json();

                const modelsEl = document.getElementById('models-list');
                if (data.models.length > 0) {
                    modelsEl.innerHTML = data.models.map(model => `
                        <div class="model-card">
                            <strong>${model.model_name}</strong><br>
                            Parameters: ${model.parameter_count.toLocaleString()}<br>
                            Focus: ${model.focus}<br>
                            Created: ${new Date(model.created_at).toLocaleString()}
                        </div>
                    `).join('');
                }
            } catch (error) {
                log('Error loading models: ' + error.message);
            }
        }

        // Initialize
        checkAPIStatus();
        connectWebSocket();
        loadModels();
        setInterval(checkAPIStatus, 5000);
    </script>
</body>
</html>
        """
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Gateway Server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
