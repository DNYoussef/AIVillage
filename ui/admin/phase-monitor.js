/**
 * Phase Monitor JavaScript - Real-time Agent Forge Phase Control & Monitoring
 *
 * Features:
 * - WebSocket real-time phase updates
 * - Phase execution control (START buttons)
 * - Progress bar animations and status management
 * - System metrics integration (CPU/GPU/Memory)
 * - Error handling and recovery
 * - Training metrics visualization
 */

class PhaseMonitor {
    constructor(options = {}) {
        this.config = {
            apiBaseUrl: options.apiBaseUrl || 'http://localhost:8083',
            wsUrl: options.wsUrl || 'ws://localhost:8085/ws',
            updateInterval: options.updateInterval || 2000,
            reconnectAttempts: options.reconnectAttempts || 5,
            reconnectDelay: options.reconnectDelay || 3000,
            ...options
        };

        // State management
        this.phases = [
            'Cognate', 'EvoMerge', 'Quiet-STaR', 'BitNet',
            'Forge Training', 'Tool Baking', 'ADAS', 'Final Compression'
        ];
        this.phaseStatus = {};
        this.systemMetrics = {};
        this.activeTasks = [];
        this.trainedModels = [];

        // WebSocket connection management
        this.ws = null;
        this.wsConnected = false;
        this.reconnectCount = 0;
        this.clientId = null;

        // DOM elements cache
        this.elements = {};

        this.init();
    }

    /**
     * Initialize the phase monitor
     */
    async init() {
        try {
            console.log('🚀 Initializing Phase Monitor...');

            // Cache DOM elements
            this.cacheElements();

            // Setup event listeners
            this.setupEventListeners();

            // Initialize WebSocket connection
            await this.connectWebSocket();

            // Initial data load
            await this.loadInitialData();

            // Start periodic updates (fallback if WebSocket fails)
            this.startPeriodicUpdates();

            console.log('✅ Phase Monitor initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize Phase Monitor:', error);
            this.handleError('Initialization failed', error);
        }
    }

    /**
     * Cache frequently accessed DOM elements
     */
    cacheElements() {
        // Phase containers
        this.elements.phaseContainer = document.getElementById('phase-container');
        this.elements.systemMetrics = document.getElementById('system-metrics');
        this.elements.connectionStatus = document.getElementById('connection-status');

        // Global controls
        this.elements.refreshBtn = document.getElementById('refresh-phases');
        this.elements.stopAllBtn = document.getElementById('stop-all-phases');

        // Create elements if they don't exist
        if (!this.elements.phaseContainer) {
            console.warn('Phase container not found, creating default structure');
            this.createDefaultStructure();
        }
    }

    /**
     * Create default DOM structure if not present
     */
    createDefaultStructure() {
        const container = document.createElement('div');
        container.id = 'phase-monitor-container';
        container.innerHTML = `
            <div class="phase-monitor-header">
                <h2>Agent Forge Phase Monitor</h2>
                <div class="controls">
                    <button id="refresh-phases" class="btn btn-secondary">🔄 Refresh</button>
                    <button id="stop-all-phases" class="btn btn-danger">⏹️ Stop All</button>
                    <div id="connection-status" class="connection-status">Connecting...</div>
                </div>
            </div>
            <div id="system-metrics" class="system-metrics"></div>
            <div id="phase-container" class="phase-container"></div>
        `;

        document.body.appendChild(container);
        this.cacheElements();
    }

    /**
     * Setup event listeners for UI interactions
     */
    setupEventListeners() {
        // Global refresh button
        if (this.elements.refreshBtn) {
            this.elements.refreshBtn.addEventListener('click', () => {
                this.refreshAllData();
            });
        }

        // Stop all phases button
        if (this.elements.stopAllBtn) {
            this.elements.stopAllBtn.addEventListener('click', () => {
                this.stopAllPhases();
            });
        }

        // Window events
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });

        // Visibility change (pause updates when tab not visible)
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                this.refreshAllData();
            }
        });
    }

    /**
     * Connect to WebSocket for real-time updates
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                console.log(`🔌 Connecting to WebSocket: ${this.config.wsUrl}`);

                this.ws = new WebSocket(this.config.wsUrl);

                this.ws.onopen = () => {
                    console.log('✅ WebSocket connected');
                    this.wsConnected = true;
                    this.reconnectCount = 0;
                    this.updateConnectionStatus('connected');

                    // Subscribe to relevant channels
                    this.subscribeToChannels();
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    this.handleWebSocketMessage(JSON.parse(event.data));
                };

                this.ws.onclose = () => {
                    console.log('🔌 WebSocket disconnected');
                    this.wsConnected = false;
                    this.updateConnectionStatus('disconnected');
                    this.attemptReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    this.updateConnectionStatus('error');
                    reject(error);
                };

                // Timeout for connection
                setTimeout(() => {
                    if (!this.wsConnected) {
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 10000);

            } catch (error) {
                console.error('❌ WebSocket connection failed:', error);
                reject(error);
            }
        });
    }

    /**
     * Subscribe to WebSocket channels for real-time updates
     */
    subscribeToChannels() {
        const channels = [
            'agent_forge_phases',
            'system_metrics',
            'training_metrics',
            'model_updates'
        ];

        channels.forEach(channel => {
            this.sendWebSocketMessage({
                type: 'subscribe',
                channel: channel
            });
        });
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(data) {
        try {
            switch (data.type) {
                case 'connection_established':
                    this.clientId = data.client_id;
                    console.log(`📡 WebSocket client ID: ${this.clientId}`);
                    break;

                case 'subscription_confirmed':
                    console.log(`✅ Subscribed to channel: ${data.channel}`);
                    break;

                case 'phase_update':
                    this.handlePhaseUpdate(data);
                    break;

                case 'system_metrics':
                    this.handleSystemMetrics(data.metrics);
                    break;

                case 'training_metrics':
                    this.handleTrainingMetrics(data);
                    break;

                case 'model_update':
                    this.handleModelUpdate(data);
                    break;

                case 'pong':
                    // Handle ping/pong for connection health
                    break;

                default:
                    console.log('📨 Unknown WebSocket message:', data);
            }
        } catch (error) {
            console.error('❌ Error handling WebSocket message:', error);
        }
    }

    /**
     * Handle real-time phase updates
     */
    handlePhaseUpdate(data) {
        const { phase_name, status, progress, message, artifacts } = data;

        console.log(`📊 Phase update - ${phase_name}: ${status} (${Math.round(progress * 100)}%)`);

        // Update phase status
        this.phaseStatus[phase_name] = {
            status,
            progress,
            message,
            artifacts,
            lastUpdate: new Date()
        };

        // Update UI
        this.updatePhaseUI(phase_name, this.phaseStatus[phase_name]);
    }

    /**
     * Handle system metrics updates
     */
    handleSystemMetrics(metrics) {
        this.systemMetrics = {
            ...metrics,
            lastUpdate: new Date()
        };

        this.updateSystemMetricsUI();
    }

    /**
     * Handle training metrics updates
     */
    handleTrainingMetrics(data) {
        const { phase_name, metrics } = data;
        console.log(`📈 Training metrics for ${phase_name}:`, metrics);

        // Update training metrics display
        this.updateTrainingMetricsUI(phase_name, metrics);
    }

    /**
     * Handle model updates
     */
    handleModelUpdate(data) {
        const { model_id, event_type, data: modelData } = data;
        console.log(`🤖 Model update - ${model_id}: ${event_type}`);

        // Update models list or specific model info
        this.updateModelUI(model_id, event_type, modelData);
    }

    /**
     * Send message through WebSocket
     */
    sendWebSocketMessage(message) {
        if (this.wsConnected && this.ws) {
            try {
                this.ws.send(JSON.stringify(message));
            } catch (error) {
                console.error('❌ Failed to send WebSocket message:', error);
            }
        }
    }

    /**
     * Attempt to reconnect WebSocket
     */
    async attemptReconnect() {
        if (this.reconnectCount >= this.config.reconnectAttempts) {
            console.error('❌ Max reconnection attempts reached');
            this.updateConnectionStatus('failed');
            return;
        }

        this.reconnectCount++;
        console.log(`🔄 Attempting WebSocket reconnection (${this.reconnectCount}/${this.config.reconnectAttempts})`);

        setTimeout(async () => {
            try {
                await this.connectWebSocket();
            } catch (error) {
                console.error('❌ Reconnection failed:', error);
            }
        }, this.config.reconnectDelay);
    }

    /**
     * Load initial data from APIs
     */
    async loadInitialData() {
        try {
            console.log('📊 Loading initial data...');

            // Load phase statuses
            await this.loadPhaseStatuses();

            // Load system metrics
            await this.loadSystemMetrics();

            // Load trained models
            await this.loadTrainedModels();

            // Render initial UI
            this.renderPhaseUI();
            this.updateSystemMetricsUI();

            console.log('✅ Initial data loaded');
        } catch (error) {
            console.error('❌ Failed to load initial data:', error);
        }
    }

    /**
     * Load phase statuses from API
     */
    async loadPhaseStatuses() {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/phases/status`);
            const data = await response.json();

            // Update phase status from API response
            data.phases.forEach(phase => {
                this.phaseStatus[phase.phase_name] = {
                    status: phase.status,
                    progress: phase.progress,
                    message: phase.message,
                    startTime: phase.start_time,
                    duration: phase.duration_seconds,
                    artifacts: phase.artifacts
                };
            });

            // Update system metrics if included
            if (data.system_metrics) {
                this.systemMetrics = data.system_metrics;
            }

        } catch (error) {
            console.error('❌ Failed to load phase statuses:', error);
            throw error;
        }
    }

    /**
     * Load system metrics
     */
    async loadSystemMetrics() {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/system/metrics`);
            this.systemMetrics = await response.json();
        } catch (error) {
            console.error('❌ Failed to load system metrics:', error);
        }
    }

    /**
     * Load trained models
     */
    async loadTrainedModels() {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/models`);
            const data = await response.json();
            this.trainedModels = data.models || [];
        } catch (error) {
            console.error('❌ Failed to load trained models:', error);
        }
    }

    /**
     * Start a specific phase
     */
    async startPhase(phaseName) {
        try {
            console.log(`🚀 Starting phase: ${phaseName}`);

            // Update UI to show starting state
            this.updatePhaseStatus(phaseName, 'starting', 0, 'Initializing...');

            let endpoint;
            let requestBody = {};

            // Map phase names to API endpoints
            switch (phaseName) {
                case 'Cognate':
                    endpoint = '/phases/cognate/start';
                    break;
                case 'EvoMerge':
                    endpoint = '/phases/evomerge/start';
                    requestBody = { config: {} };
                    break;
                default:
                    // For other phases, we'll use a generic approach
                    console.warn(`⚠️ Phase ${phaseName} not implemented yet`);
                    this.updatePhaseStatus(phaseName, 'error', 0, 'Phase not implemented');
                    return;
            }

            // Make API call to start phase
            const response = await fetch(`${this.config.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log(`✅ Phase ${phaseName} started:`, result);

            // Update UI with successful start
            this.updatePhaseStatus(phaseName, 'running', 0.1, result.message || 'Phase started');

            // Store task ID for monitoring
            if (result.task_id) {
                this.activeTasks.push({
                    taskId: result.task_id,
                    phaseName: phaseName,
                    startTime: new Date()
                });
            }

        } catch (error) {
            console.error(`❌ Failed to start phase ${phaseName}:`, error);
            this.updatePhaseStatus(phaseName, 'error', 0, `Error: ${error.message}`);
        }
    }

    /**
     * Stop a specific phase
     */
    async stopPhase(phaseName) {
        try {
            console.log(`⏹️ Stopping phase: ${phaseName}`);

            // Find active task for this phase
            const activeTask = this.activeTasks.find(task => task.phaseName === phaseName);
            if (!activeTask) {
                console.warn(`No active task found for phase: ${phaseName}`);
                return;
            }

            // Make API call to cancel task
            const response = await fetch(`${this.config.apiBaseUrl}/tasks/${activeTask.taskId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log(`✅ Phase ${phaseName} stopped:`, result);

            // Update UI
            this.updatePhaseStatus(phaseName, 'ready', 0, 'Ready');

            // Remove from active tasks
            this.activeTasks = this.activeTasks.filter(task => task.taskId !== activeTask.taskId);

        } catch (error) {
            console.error(`❌ Failed to stop phase ${phaseName}:`, error);
        }
    }

    /**
     * Stop all running phases
     */
    async stopAllPhases() {
        console.log('⏹️ Stopping all phases...');

        const runningPhases = Object.keys(this.phaseStatus).filter(
            name => this.phaseStatus[name].status === 'running'
        );

        for (const phaseName of runningPhases) {
            await this.stopPhase(phaseName);
        }
    }

    /**
     * Update phase status in memory and UI
     */
    updatePhaseStatus(phaseName, status, progress, message, artifacts = null) {
        this.phaseStatus[phaseName] = {
            status,
            progress,
            message,
            artifacts,
            lastUpdate: new Date()
        };

        this.updatePhaseUI(phaseName, this.phaseStatus[phaseName]);
    }

    /**
     * Render the complete phase monitoring UI
     */
    renderPhaseUI() {
        if (!this.elements.phaseContainer) return;

        const html = this.phases.map(phaseName => {
            const status = this.phaseStatus[phaseName] || { status: 'ready', progress: 0, message: 'Ready' };
            return this.createPhaseHTML(phaseName, status);
        }).join('');

        this.elements.phaseContainer.innerHTML = html;

        // Add event listeners for phase controls
        this.phases.forEach(phaseName => {
            const startBtn = document.getElementById(`start-${phaseName.replace(/\s+/g, '-').toLowerCase()}`);
            const stopBtn = document.getElementById(`stop-${phaseName.replace(/\s+/g, '-').toLowerCase()}`);

            if (startBtn) {
                startBtn.addEventListener('click', () => this.startPhase(phaseName));
            }

            if (stopBtn) {
                stopBtn.addEventListener('click', () => this.stopPhase(phaseName));
            }
        });
    }

    /**
     * Create HTML for individual phase
     */
    createPhaseHTML(phaseName, status) {
        const phaseId = phaseName.replace(/\s+/g, '-').toLowerCase();
        const statusClass = this.getStatusClass(status.status);
        const progressPercent = Math.round(status.progress * 100);

        return `
            <div class="phase-card ${statusClass}" id="phase-${phaseId}">
                <div class="phase-header">
                    <h3 class="phase-title">${phaseName}</h3>
                    <div class="phase-status">
                        <span class="status-badge status-${status.status}">${status.status.toUpperCase()}</span>
                    </div>
                </div>

                <div class="phase-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressPercent}%"></div>
                        <span class="progress-text">${progressPercent}%</span>
                    </div>
                </div>

                <div class="phase-message">
                    <span class="message-text">${status.message || 'Ready'}</span>
                </div>

                <div class="phase-controls">
                    <button id="start-${phaseId}" class="btn btn-primary phase-start-btn"
                            ${status.status === 'running' ? 'disabled' : ''}>
                        ${status.status === 'running' ? '⏳ Running' : '▶️ Start'}
                    </button>
                    <button id="stop-${phaseId}" class="btn btn-secondary phase-stop-btn"
                            ${status.status !== 'running' ? 'disabled' : ''}>
                        ⏹️ Stop
                    </button>
                </div>

                ${status.artifacts ? this.createArtifactsHTML(status.artifacts) : ''}
            </div>
        `;
    }

    /**
     * Create HTML for phase artifacts/results
     */
    createArtifactsHTML(artifacts) {
        if (!artifacts || Object.keys(artifacts).length === 0) return '';

        return `
            <div class="phase-artifacts">
                <h4>Results</h4>
                <ul class="artifacts-list">
                    ${Object.entries(artifacts).map(([key, value]) =>
                        `<li><strong>${key}:</strong> ${JSON.stringify(value)}</li>`
                    ).join('')}
                </ul>
            </div>
        `;
    }

    /**
     * Update specific phase UI
     */
    updatePhaseUI(phaseName, status) {
        const phaseId = phaseName.replace(/\s+/g, '-').toLowerCase();
        const phaseElement = document.getElementById(`phase-${phaseId}`);

        if (!phaseElement) {
            // If phase element doesn't exist, re-render entire UI
            this.renderPhaseUI();
            return;
        }

        // Update status class
        phaseElement.className = `phase-card ${this.getStatusClass(status.status)}`;

        // Update status badge
        const statusBadge = phaseElement.querySelector('.status-badge');
        if (statusBadge) {
            statusBadge.textContent = status.status.toUpperCase();
            statusBadge.className = `status-badge status-${status.status}`;
        }

        // Update progress bar
        const progressFill = phaseElement.querySelector('.progress-fill');
        const progressText = phaseElement.querySelector('.progress-text');
        if (progressFill && progressText) {
            const progressPercent = Math.round(status.progress * 100);
            progressFill.style.width = `${progressPercent}%`;
            progressText.textContent = `${progressPercent}%`;
        }

        // Update message
        const messageText = phaseElement.querySelector('.message-text');
        if (messageText) {
            messageText.textContent = status.message || 'Ready';
        }

        // Update buttons
        const startBtn = phaseElement.querySelector('.phase-start-btn');
        const stopBtn = phaseElement.querySelector('.phase-stop-btn');

        if (startBtn) {
            startBtn.disabled = status.status === 'running';
            startBtn.textContent = status.status === 'running' ? '⏳ Running' : '▶️ Start';
        }

        if (stopBtn) {
            stopBtn.disabled = status.status !== 'running';
        }

        // Update artifacts if present
        const existingArtifacts = phaseElement.querySelector('.phase-artifacts');
        if (existingArtifacts) {
            existingArtifacts.remove();
        }

        if (status.artifacts) {
            phaseElement.insertAdjacentHTML('beforeend', this.createArtifactsHTML(status.artifacts));
        }
    }

    /**
     * Update system metrics UI
     */
    updateSystemMetricsUI() {
        if (!this.elements.systemMetrics || !this.systemMetrics) return;

        const metrics = this.systemMetrics;
        const html = `
            <div class="metrics-container">
                <h3>System Resources</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">CPU Usage</div>
                        <div class="metric-value">${metrics.cpu?.usage_percent?.toFixed(1) || 0}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${metrics.cpu?.usage_percent || 0}%"></div>
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value">${metrics.memory?.usage_percent?.toFixed(1) || 0}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${metrics.memory?.usage_percent || 0}%"></div>
                        </div>
                    </div>

                    ${metrics.gpu ? `
                        <div class="metric-card">
                            <div class="metric-label">GPU Memory</div>
                            <div class="metric-value">${(metrics.gpu.gpu_memory_allocated || 0).toFixed(1)} GB</div>
                            <div class="metric-detail">${metrics.gpu.gpu_name || 'Unknown GPU'}</div>
                        </div>
                    ` : ''}
                </div>

                <div class="metrics-details">
                    <span class="detail-item">CPU Cores: ${metrics.cpu?.count || 'N/A'}</span>
                    <span class="detail-item">Total Memory: ${(metrics.memory?.total_gb || 0).toFixed(1)} GB</span>
                    ${metrics.gpu ? `<span class="detail-item">GPU Count: ${metrics.gpu.gpu_count || 0}</span>` : ''}
                    <span class="detail-item">Last Update: ${new Date().toLocaleTimeString()}</span>
                </div>
            </div>
        `;

        this.elements.systemMetrics.innerHTML = html;
    }

    /**
     * Update training metrics UI
     */
    updateTrainingMetricsUI(phaseName, metrics) {
        // Find phase element and add/update training metrics
        const phaseId = phaseName.replace(/\s+/g, '-').toLowerCase();
        const phaseElement = document.getElementById(`phase-${phaseId}`);

        if (!phaseElement) return;

        // Remove existing training metrics
        const existingMetrics = phaseElement.querySelector('.training-metrics');
        if (existingMetrics) {
            existingMetrics.remove();
        }

        // Add new training metrics
        const metricsHTML = `
            <div class="training-metrics">
                <h4>Training Metrics</h4>
                <div class="metrics-list">
                    ${Object.entries(metrics).map(([key, value]) =>
                        `<div class="metric-item">
                            <span class="metric-name">${key}:</span>
                            <span class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                        </div>`
                    ).join('')}
                </div>
            </div>
        `;

        phaseElement.insertAdjacentHTML('beforeend', metricsHTML);
    }

    /**
     * Update model UI
     */
    updateModelUI(modelId, eventType, modelData) {
        // This could update a models section if present
        console.log(`Model UI update: ${modelId} - ${eventType}`, modelData);
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(status) {
        if (!this.elements.connectionStatus) return;

        const statusMap = {
            connected: { text: '🟢 Connected', class: 'connected' },
            connecting: { text: '🟡 Connecting...', class: 'connecting' },
            disconnected: { text: '🔴 Disconnected', class: 'disconnected' },
            error: { text: '❌ Connection Error', class: 'error' },
            failed: { text: '💀 Connection Failed', class: 'failed' }
        };

        const statusInfo = statusMap[status] || statusMap.disconnected;
        this.elements.connectionStatus.textContent = statusInfo.text;
        this.elements.connectionStatus.className = `connection-status ${statusInfo.class}`;
    }

    /**
     * Get CSS class for phase status
     */
    getStatusClass(status) {
        const classMap = {
            ready: 'status-ready',
            starting: 'status-starting',
            running: 'status-running',
            completed: 'status-completed',
            error: 'status-error',
            cancelled: 'status-cancelled'
        };
        return classMap[status] || 'status-unknown';
    }

    /**
     * Start periodic updates (fallback if WebSocket fails)
     */
    startPeriodicUpdates() {
        setInterval(async () => {
            if (!this.wsConnected) {
                try {
                    await this.loadPhaseStatuses();
                    await this.loadSystemMetrics();
                    this.updateSystemMetricsUI();
                } catch (error) {
                    console.error('❌ Periodic update failed:', error);
                }
            }
        }, this.config.updateInterval);
    }

    /**
     * Refresh all data
     */
    async refreshAllData() {
        console.log('🔄 Refreshing all data...');
        try {
            await this.loadInitialData();
            console.log('✅ Data refreshed');
        } catch (error) {
            console.error('❌ Failed to refresh data:', error);
        }
    }

    /**
     * Handle errors
     */
    handleError(message, error) {
        console.error(`❌ ${message}:`, error);

        // Show error notification if available
        if (typeof showNotification === 'function') {
            showNotification(message, 'error');
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        console.log('🧹 Cleaning up Phase Monitor...');

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.wsConnected = false;
    }

    /**
     * Public API for external control
     */
    getPhaseStatus(phaseName = null) {
        if (phaseName) {
            return this.phaseStatus[phaseName] || null;
        }
        return this.phaseStatus;
    }

    getSystemMetrics() {
        return this.systemMetrics;
    }

    getTrainedModels() {
        return this.trainedModels;
    }

    isPhaseRunning(phaseName) {
        const status = this.phaseStatus[phaseName];
        return status && status.status === 'running';
    }

    async forceRefresh() {
        await this.refreshAllData();
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're in the right context (dev UI)
    if (document.body.classList.contains('dev-ui') ||
        document.getElementById('phase-container') ||
        window.initPhaseMonitor) {

        console.log('🚀 Auto-initializing Phase Monitor...');
        window.phaseMonitor = new PhaseMonitor();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PhaseMonitor;
} else if (typeof window !== 'undefined') {
    window.PhaseMonitor = PhaseMonitor;
}

/**
 * CSS Classes Reference (to be included in your CSS):
 *
 * .phase-card { ... }
 * .phase-header { ... }
 * .phase-title { ... }
 * .status-badge { ... }
 * .status-ready { color: #28a745; }
 * .status-running { color: #007bff; }
 * .status-completed { color: #28a745; }
 * .status-error { color: #dc3545; }
 * .progress-bar { ... }
 * .progress-fill { ... }
 * .progress-text { ... }
 * .phase-controls { ... }
 * .btn { ... }
 * .btn-primary { ... }
 * .btn-secondary { ... }
 * .metrics-container { ... }
 * .metrics-grid { ... }
 * .metric-card { ... }
 * .metric-bar { ... }
 * .metric-fill { ... }
 * .connection-status { ... }
 * .connected { color: #28a745; }
 * .disconnected { color: #dc3545; }
 * .training-metrics { ... }
 * .phase-artifacts { ... }
 */
