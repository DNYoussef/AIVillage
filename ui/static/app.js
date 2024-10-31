// AI Village UI Application Logic

class AIVillageUI {
    constructor() {
        this.ws = null;
        this.performanceChart = null;
        this.knowledgeGraph = null;
        this.decisionTree = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    // Initialize WebSocket connection
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connection established');
            this.reconnectAttempts = 0;
            this.requestInitialData();
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket connection closed');
            this.handleReconnection();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    // Handle WebSocket reconnection
    handleReconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.initializeWebSocket(), 2000 * this.reconnectAttempts);
        } else {
            console.error('Max reconnection attempts reached');
            this.showError('Connection lost. Please refresh the page.');
        }
    }

    // Request initial data when connection is established
    requestInitialData() {
        this.sendWebSocketMessage({
            type: 'get_metrics'
        });
        
        this.sendWebSocketMessage({
            type: 'get_status'
        });
    }

    // Send message through WebSocket
    sendWebSocketMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket is not connected');
        }
    }

    // Handle incoming WebSocket messages
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'metrics_update':
                this.updateMetrics(data.data);
                break;
            case 'status_update':
                this.updateStatus(data.data);
                break;
            case 'knowledge_update':
                this.updateKnowledgeGraph(data.data);
                break;
            case 'decision_update':
                this.updateDecisionTree(data.data);
                break;
            case 'chat_message':
                this.updateChat(data.data);
                break;
            case 'error':
                this.showError(data.message);
                break;
            default:
                console.warn('Unknown message type:', data.type);
        }
    }

    // Initialize performance chart
    initializePerformanceChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Task Success Rate',
                        data: [],
                        borderColor: '#3B82F6',
                        tension: 0.4
                    },
                    {
                        label: 'Response Quality',
                        data: [],
                        borderColor: '#10B981',
                        tension: 0.4
                    },
                    {
                        label: 'System Load',
                        data: [],
                        borderColor: '#F59E0B',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    // Initialize knowledge graph visualization
    initializeKnowledgeGraph(container) {
        const data = {
            nodes: new vis.DataSet([]),
            edges: new vis.DataSet([])
        };

        const options = {
            nodes: {
                shape: 'dot',
                size: 30,
                font: {
                    size: 14
                }
            },
            edges: {
                arrows: 'to',
                smooth: {
                    type: 'cubicBezier'
                }
            },
            physics: {
                stabilization: true,
                barnesHut: {
                    gravitationalConstant: -80000,
                    springConstant: 0.001,
                    springLength: 200
                }
            }
        };

        this.knowledgeGraph = new vis.Network(container, data, options);
    }

    // Initialize decision tree visualization
    initializeDecisionTree(container) {
        const data = {
            nodes: new vis.DataSet([]),
            edges: new vis.DataSet([])
        };

        const options = {
            layout: {
                hierarchical: {
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100
                }
            },
            nodes: {
                shape: 'box',
                margin: 10,
                font: {
                    size: 14
                }
            },
            edges: {
                arrows: 'to',
                smooth: {
                    type: 'cubicBezier'
                }
            }
        };

        this.decisionTree = new vis.Network(container, data, options);
    }

    // Update metrics display
    updateMetrics(metrics) {
        if (this.performanceChart) {
            const timestamp = new Date().toLocaleTimeString();
            
            this.performanceChart.data.labels.push(timestamp);
            this.performanceChart.data.datasets[0].data.push(metrics.task_success_rate);
            this.performanceChart.data.datasets[1].data.push(metrics.response_quality);
            this.performanceChart.data.datasets[2].data.push(metrics.system_load);

            // Keep last 20 data points
            if (this.performanceChart.data.labels.length > 20) {
                this.performanceChart.data.labels.shift();
                this.performanceChart.data.datasets.forEach(dataset => dataset.data.shift());
            }

            this.performanceChart.update();
        }

        // Update other metric displays
        Object.entries(metrics).forEach(([key, value]) => {
            const element = document.getElementById(`metric-${key}`);
            if (element) {
                element.textContent = typeof value === 'number' ? value.toFixed(2) : value;
            }
        });
    }

    // Update system status display
    updateStatus(status) {
        Object.entries(status).forEach(([key, value]) => {
            const element = document.getElementById(`status-${key}`);
            if (element) {
                element.className = `agent-status status-${value.toLowerCase()}`;
                element.title = `${key}: ${value}`;
            }
        });
    }

    // Update knowledge graph visualization
    updateKnowledgeGraph(data) {
        if (this.knowledgeGraph) {
            this.knowledgeGraph.setData({
                nodes: new vis.DataSet(data.nodes),
                edges: new vis.DataSet(data.edges)
            });
        }
    }

    // Update decision tree visualization
    updateDecisionTree(data) {
        if (this.decisionTree) {
            this.decisionTree.setData({
                nodes: new vis.DataSet(data.nodes),
                edges: new vis.DataSet(data.edges)
            });
        }
    }

    // Update chat display
    updateChat(message) {
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            const messageElement = document.createElement('div');
            messageElement.className = `chat-message ${message.sender === 'user' ? 'user-message' : 'agent-message'}`;
            
            const senderElement = document.createElement('div');
            senderElement.className = 'font-semibold mb-1';
            senderElement.textContent = message.sender === 'user' ? 'You' : 'Agent';
            
            const contentElement = document.createElement('div');
            contentElement.textContent = message.content;
            
            messageElement.appendChild(senderElement);
            messageElement.appendChild(contentElement);
            
            if (message.context) {
                const contextElement = document.createElement('div');
                contextElement.className = 'mt-2 text-sm text-gray-600';
                
                if (message.context.relevant_concepts) {
                    const conceptsElement = document.createElement('div');
                    conceptsElement.textContent = `Related: ${message.context.relevant_concepts.join(', ')}`;
                    contextElement.appendChild(conceptsElement);
                }
                
                if (message.context.confidence) {
                    const confidenceElement = document.createElement('div');
                    confidenceElement.textContent = `Confidence: ${(message.context.confidence * 100).toFixed(1)}%`;
                    contextElement.appendChild(confidenceElement);
                }
                
                messageElement.appendChild(contextElement);
            }
            
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // Show error message
    showError(message) {
        const errorContainer = document.getElementById('error-container');
        if (errorContainer) {
            const errorElement = document.createElement('div');
            errorElement.className = 'error-message';
            errorElement.textContent = message;
            
            errorContainer.appendChild(errorElement);
            setTimeout(() => errorElement.remove(), 5000);
        }
    }

    // Initialize all UI components
    initialize() {
        this.initializeWebSocket();
        this.initializePerformanceChart();
        
        const knowledgeContainer = document.getElementById('knowledgeGraph');
        if (knowledgeContainer) {
            this.initializeKnowledgeGraph(knowledgeContainer);
        }
        
        const decisionContainer = document.getElementById('decisionTree');
        if (decisionContainer) {
            this.initializeDecisionTree(decisionContainer);
        }
    }
}

// Initialize UI when document is ready
document.addEventListener('DOMContentLoaded', () => {
    const ui = new AIVillageUI();
    ui.initialize();
    window.aiVillageUI = ui; // Make instance accessible globally
});
