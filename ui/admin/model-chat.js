/**
 * Model Chat Interface
 *
 * Interactive chat interface for testing trained models after each Agent Forge phase:
 * - Dynamic model buttons that appear when models are trained
 * - Real-time chat with trained models via WebSocket
 * - Model comparison for side-by-side testing
 * - Automatic model registration from training phases
 * - Session management and chat history
 */

class ModelChatInterface {
    constructor() {
        this.apiBaseUrl = '/api/model-chat';
        this.wsConnections = new Map(); // WebSocket connections per session
        this.activeSessions = new Map(); // Active chat sessions
        this.availableModels = [];
        this.comparisonMode = false;
        this.selectedModels = new Set();

        this.init();
    }

    async init() {
        this.createChatInterface();
        await this.loadAvailableModels();
        this.setupEventListeners();
        this.startModelPolling(); // Check for new models periodically

        console.log('Model Chat Interface initialized');
    }

    createChatInterface() {
        const container = document.getElementById('model-chat-container');
        if (!container) return;

        container.innerHTML = `
            <div class="model-chat-interface">
                <!-- Header Controls -->
                <div class="chat-header">
                    <h3>Agent Forge Model Chat</h3>
                    <div class="header-controls">
                        <button id="refresh-models" class="btn btn-sm btn-primary">
                            <i class="fas fa-sync"></i> Refresh Models
                        </button>
                        <button id="toggle-comparison" class="btn btn-sm btn-secondary">
                            <i class="fas fa-columns"></i> Compare Models
                        </button>
                        <button id="clear-all-sessions" class="btn btn-sm btn-danger">
                            <i class="fas fa-trash"></i> Clear All
                        </button>
                    </div>
                </div>

                <!-- Model Selection -->
                <div class="models-panel">
                    <h4>Available Models</h4>
                    <div id="model-buttons" class="model-buttons">
                        <!-- Dynamic model buttons will be inserted here -->
                    </div>
                    <div id="model-loading" class="loading-indicator" style="display: none;">
                        <i class="fas fa-spinner fa-spin"></i> Loading models...
                    </div>
                </div>

                <!-- Chat Interface -->
                <div id="chat-interface" class="chat-interface" style="display: none;">
                    <!-- Single model chat -->
                    <div id="single-chat" class="single-chat">
                        <div class="chat-session">
                            <div class="chat-header-info">
                                <span id="current-model-name">No Model Selected</span>
                                <span id="session-info"></span>
                            </div>
                            <div id="chat-messages" class="chat-messages"></div>
                            <div class="chat-input-container">
                                <div class="input-group">
                                    <textarea id="chat-input" class="form-control"
                                             placeholder="Type your message..."
                                             rows="2"></textarea>
                                    <div class="input-group-append">
                                        <button id="send-message" class="btn btn-primary">
                                            <i class="fas fa-paper-plane"></i>
                                        </button>
                                    </div>
                                </div>
                                <div class="chat-options">
                                    <label>Max Tokens:
                                        <input id="max-tokens" type="number" value="256" min="1" max="1024">
                                    </label>
                                    <label>Temperature:
                                        <input id="temperature" type="range" value="0.7" min="0" max="2" step="0.1">
                                        <span id="temp-value">0.7</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Comparison mode -->
                    <div id="comparison-chat" class="comparison-chat" style="display: none;">
                        <div class="comparison-header">
                            <h4>Model Comparison</h4>
                            <div class="selected-models" id="selected-models"></div>
                        </div>
                        <div class="comparison-input">
                            <div class="input-group">
                                <textarea id="comparison-input" class="form-control"
                                         placeholder="Enter prompt to compare across selected models..."
                                         rows="3"></textarea>
                                <div class="input-group-append">
                                    <button id="compare-send" class="btn btn-success">
                                        <i class="fas fa-balance-scale"></i> Compare
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div id="comparison-results" class="comparison-results"></div>
                    </div>
                </div>

                <!-- Status Panel -->
                <div class="status-panel">
                    <div id="connection-status" class="status-item">
                        <i class="fas fa-wifi text-success"></i> Connected
                    </div>
                    <div id="model-status" class="status-item">
                        <i class="fas fa-robot text-muted"></i> <span id="loaded-models-count">0</span> models loaded
                    </div>
                    <div id="session-status" class="status-item">
                        <i class="fas fa-comments text-info"></i> <span id="active-sessions-count">0</span> active sessions
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Header controls
        document.getElementById('refresh-models')?.addEventListener('click', () => this.loadAvailableModels());
        document.getElementById('toggle-comparison')?.addEventListener('click', () => this.toggleComparisonMode());
        document.getElementById('clear-all-sessions')?.addEventListener('click', () => this.clearAllSessions());

        // Chat input
        const chatInput = document.getElementById('chat-input');
        const sendButton = document.getElementById('send-message');

        chatInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        sendButton?.addEventListener('click', () => this.sendMessage());

        // Temperature slider
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        tempSlider?.addEventListener('input', (e) => {
            tempValue.textContent = e.target.value;
        });

        // Comparison mode
        document.getElementById('compare-send')?.addEventListener('click', () => this.compareModels());

        // Window events
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    async loadAvailableModels() {
        const loadingIndicator = document.getElementById('model-loading');
        const modelButtons = document.getElementById('model-buttons');

        if (loadingIndicator) loadingIndicator.style.display = 'block';

        try {
            const response = await fetch(`${this.apiBaseUrl}/models`);
            const data = await response.json();

            this.availableModels = data.models;
            this.renderModelButtons();
            this.updateStatus();

        } catch (error) {
            console.error('Failed to load models:', error);
            this.showError('Failed to load available models');
        } finally {
            if (loadingIndicator) loadingIndicator.style.display = 'none';
        }
    }

    renderModelButtons() {
        const container = document.getElementById('model-buttons');
        if (!container) return;

        if (this.availableModels.length === 0) {
            container.innerHTML = `
                <div class="no-models">
                    <i class="fas fa-robot text-muted"></i>
                    <p>No trained models available yet.</p>
                    <small>Models will appear here after each Agent Forge training phase completes.</small>
                </div>
            `;
            return;
        }

        container.innerHTML = this.availableModels.map(model => {
            const isLoaded = model.status === 'loaded';
            const isSelected = this.selectedModels.has(model.model_id);

            return `
                <div class="model-card ${isSelected ? 'selected' : ''}" data-model-id="${model.model_id}">
                    <div class="model-header">
                        <h5>${model.model_name || model.model_id}</h5>
                        <div class="model-status">
                            <span class="badge ${isLoaded ? 'badge-success' : 'badge-secondary'}">
                                ${isLoaded ? 'Loaded' : 'Available'}
                            </span>
                        </div>
                    </div>
                    <div class="model-info">
                        <small><strong>Phase:</strong> ${model.phase_name || 'Unknown'}</small><br>
                        <small><strong>Parameters:</strong> ${(model.parameter_count || 0).toLocaleString()}</small><br>
                        <small><strong>Registered:</strong> ${new Date(model.registered_at).toLocaleString()}</small>
                    </div>
                    <div class="model-actions">
                        ${this.comparisonMode ? `
                            <button class="btn btn-sm ${isSelected ? 'btn-warning' : 'btn-outline-primary'} select-model">
                                <i class="fas ${isSelected ? 'fa-minus' : 'fa-plus'}"></i>
                                ${isSelected ? 'Remove' : 'Select'}
                            </button>
                        ` : `
                            <button class="btn btn-sm btn-primary chat-model">
                                <i class="fas fa-comment"></i> Chat
                            </button>
                        `}
                        <button class="btn btn-sm ${isLoaded ? 'btn-danger' : 'btn-secondary'} load-model">
                            <i class="fas ${isLoaded ? 'fa-eject' : 'fa-download'}"></i>
                            ${isLoaded ? 'Unload' : 'Load'}
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        // Add event listeners to model buttons
        container.querySelectorAll('.chat-model').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelId = e.target.closest('.model-card').dataset.modelId;
                this.startChatSession(modelId);
            });
        });

        container.querySelectorAll('.select-model').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelId = e.target.closest('.model-card').dataset.modelId;
                this.toggleModelSelection(modelId);
            });
        });

        container.querySelectorAll('.load-model').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelId = e.target.closest('.model-card').dataset.modelId;
                const model = this.availableModels.find(m => m.model_id === modelId);

                if (model.status === 'loaded') {
                    this.unloadModel(modelId);
                } else {
                    this.loadModel(modelId);
                }
            });
        });
    }

    toggleModelSelection(modelId) {
        if (this.selectedModels.has(modelId)) {
            this.selectedModels.delete(modelId);
        } else {
            this.selectedModels.add(modelId);
        }

        this.renderModelButtons();
        this.updateSelectedModelsDisplay();
    }

    updateSelectedModelsDisplay() {
        const container = document.getElementById('selected-models');
        if (!container) return;

        const selectedModelNames = Array.from(this.selectedModels).map(id => {
            const model = this.availableModels.find(m => m.model_id === id);
            return model ? model.model_name || model.model_id : id;
        });

        container.innerHTML = selectedModelNames.length > 0
            ? `Selected: ${selectedModelNames.join(', ')}`
            : 'No models selected for comparison';
    }

    async loadModel(modelId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/models/${modelId}/load`, {
                method: 'POST'
            });

            if (response.ok) {
                await this.loadAvailableModels(); // Refresh model status
                this.showSuccess(`Model ${modelId} loaded successfully`);
            } else {
                const error = await response.json();
                this.showError(`Failed to load model: ${error.detail}`);
            }
        } catch (error) {
            console.error('Load model error:', error);
            this.showError('Failed to load model');
        }
    }

    async unloadModel(modelId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/models/${modelId}/unload`, {
                method: 'DELETE'
            });

            if (response.ok) {
                await this.loadAvailableModels(); // Refresh model status
                this.showSuccess(`Model ${modelId} unloaded successfully`);
            } else {
                const error = await response.json();
                this.showError(`Failed to unload model: ${error.detail}`);
            }
        } catch (error) {
            console.error('Unload model error:', error);
            this.showError('Failed to unload model');
        }
    }

    toggleComparisonMode() {
        this.comparisonMode = !this.comparisonMode;

        const singleChat = document.getElementById('single-chat');
        const comparisonChat = document.getElementById('comparison-chat');
        const toggleBtn = document.getElementById('toggle-comparison');

        if (this.comparisonMode) {
            singleChat.style.display = 'none';
            comparisonChat.style.display = 'block';
            toggleBtn.innerHTML = '<i class="fas fa-comment"></i> Single Chat';
            toggleBtn.className = 'btn btn-sm btn-primary';
        } else {
            singleChat.style.display = 'block';
            comparisonChat.style.display = 'none';
            toggleBtn.innerHTML = '<i class="fas fa-columns"></i> Compare Models';
            toggleBtn.className = 'btn btn-sm btn-secondary';
            this.selectedModels.clear();
        }

        this.renderModelButtons();
        this.updateSelectedModelsDisplay();
    }

    async startChatSession(modelId) {
        const model = this.availableModels.find(m => m.model_id === modelId);
        if (!model) {
            this.showError('Model not found');
            return;
        }

        // Ensure model is loaded
        if (model.status !== 'loaded') {
            await this.loadModel(modelId);
        }

        const sessionId = this.generateSessionId();

        // Update UI
        document.getElementById('chat-interface').style.display = 'block';
        document.getElementById('current-model-name').textContent = model.model_name || model.model_id;
        document.getElementById('session-info').textContent = `Session: ${sessionId.slice(0, 8)}...`;

        // Clear previous messages
        document.getElementById('chat-messages').innerHTML = '';

        // Setup WebSocket connection
        this.setupWebSocketConnection(sessionId, modelId);

        // Store active session
        this.activeSessions.set(sessionId, {
            sessionId,
            modelId,
            model: model,
            messages: []
        });

        this.updateStatus();
    }

    setupWebSocketConnection(sessionId, modelId) {
        const wsUrl = `ws://${window.location.host}/api/model-chat/ws/${sessionId}`;

        try {
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log(`WebSocket connected for session: ${sessionId}`);
                this.updateConnectionStatus(true);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data, sessionId);
            };

            ws.onclose = () => {
                console.log(`WebSocket disconnected for session: ${sessionId}`);
                this.updateConnectionStatus(false);
                this.wsConnections.delete(sessionId);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showError('WebSocket connection failed');
                this.updateConnectionStatus(false);
            };

            this.wsConnections.set(sessionId, ws);

        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.showError('Failed to establish real-time connection');
        }
    }

    handleWebSocketMessage(data, sessionId) {
        if (data.type === 'chat_response') {
            this.displayChatResponse(data.data, sessionId);
        } else if (data.type === 'error') {
            this.showError(`Chat error: ${data.error}`);
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();

        if (!message) return;

        const activeSession = Array.from(this.activeSessions.values())[0];
        if (!activeSession) {
            this.showError('No active chat session');
            return;
        }

        // Clear input
        input.value = '';

        // Display user message
        this.displayUserMessage(message, activeSession.sessionId);

        // Send via WebSocket if available, otherwise use REST API
        const ws = this.wsConnections.get(activeSession.sessionId);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                model_id: activeSession.modelId,
                message: message
            }));
        } else {
            // Fallback to REST API
            await this.sendMessageRest(activeSession, message);
        }
    }

    async sendMessageRest(session, message) {
        try {
            const maxTokens = parseInt(document.getElementById('max-tokens').value) || 256;
            const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;

            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_id: session.modelId,
                    message: message,
                    session_id: session.sessionId,
                    max_tokens: maxTokens,
                    temperature: temperature
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.displayChatResponse(data, session.sessionId);
            } else {
                const error = await response.json();
                this.showError(`Chat failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Send message error:', error);
            this.showError('Failed to send message');
        }
    }

    displayUserMessage(message, sessionId) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(message)}</div>
                <div class="message-meta">
                    <i class="fas fa-user"></i>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                </div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store in session
        const session = this.activeSessions.get(sessionId);
        if (session) {
            session.messages.push({
                role: 'user',
                content: message,
                timestamp: new Date()
            });
        }
    }

    displayChatResponse(responseData, sessionId) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(responseData.response)}</div>
                <div class="message-meta">
                    <i class="fas fa-robot"></i>
                    <span class="model-name">${responseData.model_name}</span>
                    <span class="response-time">${responseData.response_time_ms.toFixed(1)}ms</span>
                    <span class="token-count">${responseData.token_count} tokens</span>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                </div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store in session
        const session = this.activeSessions.get(sessionId);
        if (session) {
            session.messages.push({
                role: 'assistant',
                content: responseData.response,
                timestamp: new Date(),
                model_id: responseData.model_id,
                response_time_ms: responseData.response_time_ms
            });
        }
    }

    async compareModels() {
        if (this.selectedModels.size < 2) {
            this.showError('Please select at least 2 models for comparison');
            return;
        }

        const input = document.getElementById('comparison-input');
        const message = input.value.trim();

        if (!message) {
            this.showError('Please enter a prompt to compare');
            return;
        }

        const resultsContainer = document.getElementById('comparison-results');
        resultsContainer.innerHTML = '<div class="loading">Comparing models...</div>';

        try {
            const response = await fetch(`${this.apiBaseUrl}/compare`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    model_ids: Array.from(this.selectedModels)
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.displayComparisonResults(data);
            } else {
                const error = await response.json();
                this.showError(`Comparison failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Comparison error:', error);
            this.showError('Failed to compare models');
        }
    }

    displayComparisonResults(data) {
        const resultsContainer = document.getElementById('comparison-results');

        resultsContainer.innerHTML = `
            <div class="comparison-header">
                <h5>Comparison Results</h5>
                <div class="prompt-display">
                    <strong>Prompt:</strong> "${this.escapeHtml(data.prompt)}"
                </div>
                <div class="timestamp">
                    <i class="fas fa-clock"></i> ${new Date(data.timestamp).toLocaleTimeString()}
                </div>
            </div>
            <div class="comparison-grid">
                ${data.comparisons.map(comp => `
                    <div class="comparison-item ${comp.error ? 'error' : ''}">
                        <div class="comparison-header">
                            <h6>${comp.model_name || comp.model_id}</h6>
                            ${comp.error ? '' : `
                                <div class="comparison-stats">
                                    <span class="response-time">${comp.response_time_ms.toFixed(1)}ms</span>
                                    <span class="token-count">${comp.token_count} tokens</span>
                                </div>
                            `}
                        </div>
                        <div class="comparison-response">
                            ${comp.error ? `
                                <div class="error-message">
                                    <i class="fas fa-exclamation-triangle"></i>
                                    ${comp.error}
                                </div>
                            ` : `
                                <div class="response-text">${this.escapeHtml(comp.response)}</div>
                            `}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    async registerModel(modelInfo) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/models/${modelInfo.model_id}/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(modelInfo)
            });

            if (response.ok) {
                console.log(`Model ${modelInfo.model_id} registered successfully`);
                await this.loadAvailableModels(); // Refresh available models
                this.showSuccess(`New model available: ${modelInfo.model_name || modelInfo.model_id}`);
            } else {
                console.error('Failed to register model:', await response.text());
            }
        } catch (error) {
            console.error('Model registration error:', error);
        }
    }

    startModelPolling() {
        // Poll for new models every 30 seconds
        setInterval(() => {
            this.loadAvailableModels();
        }, 30000);
    }

    async clearAllSessions() {
        if (!confirm('Are you sure you want to clear all chat sessions?')) {
            return;
        }

        try {
            // Close all WebSocket connections
            for (const [sessionId, ws] of this.wsConnections) {
                ws.close();
            }

            // Clear all sessions via API
            for (const sessionId of this.activeSessions.keys()) {
                await fetch(`${this.apiBaseUrl}/sessions/${sessionId}`, {
                    method: 'DELETE'
                });
            }

            // Clear local state
            this.wsConnections.clear();
            this.activeSessions.clear();

            // Hide chat interface
            document.getElementById('chat-interface').style.display = 'none';

            this.updateStatus();
            this.showSuccess('All sessions cleared');

        } catch (error) {
            console.error('Clear sessions error:', error);
            this.showError('Failed to clear all sessions');
        }
    }

    updateStatus() {
        document.getElementById('loaded-models-count').textContent =
            this.availableModels.filter(m => m.status === 'loaded').length;

        document.getElementById('active-sessions-count').textContent =
            this.activeSessions.size;
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');

        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-wifi text-success"></i> Connected';
        } else {
            statusElement.innerHTML = '<i class="fas fa-wifi text-danger"></i> Disconnected';
        }
    }

    generateSessionId() {
        return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
        console.error('Model Chat Error:', message);
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);

        // Manual close
        notification.querySelector('.notification-close').addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }

    cleanup() {
        // Close all WebSocket connections
        for (const [sessionId, ws] of this.wsConnections) {
            ws.close();
        }

        this.wsConnections.clear();
        this.activeSessions.clear();
    }
}

// Global API for external model registration
window.ModelChatAPI = {
    registerModel: (modelInfo) => {
        if (window.modelChat) {
            return window.modelChat.registerModel(modelInfo);
        } else {
            console.warn('Model Chat Interface not initialized yet');
            return Promise.resolve();
        }
    },

    refreshModels: () => {
        if (window.modelChat) {
            return window.modelChat.loadAvailableModels();
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.modelChat = new ModelChatInterface();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelChatInterface;
}
