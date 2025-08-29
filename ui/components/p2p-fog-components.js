/**
 * Shared P2P/Fog Computing UI Components
 *
 * Reusable JavaScript components for P2P network and Fog computing visualization
 * Used by both backend admin interface and frontend admin dashboard
 */

// P2P Network Status Component
class P2PNetworkStatus {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeComponent();
    }

    initializeComponent() {
        this.container.innerHTML = `
            <div class="network-topology">
                <div class="network-node" id="local-node">Local</div>
                <div class="connection-lines" id="connection-lines"></div>
                <div class="peer-nodes" id="peer-nodes"></div>
            </div>
            <div class="network-stats">
                <div class="stat-item">
                    <span class="stat-label">Peers</span>
                    <span class="stat-value" id="peer-count">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Latency</span>
                    <span class="stat-value" id="avg-latency">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Throughput</span>
                    <span class="stat-value" id="throughput">--</span>
                </div>
            </div>
        `;
        this.addStyles();
    }

    addStyles() {
        const styles = `
            .network-topology {
                position: relative;
                height: 200px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                margin: 15px 0;
            }

            .network-node {
                position: absolute;
                width: 40px;
                height: 40px;
                background: #4fc3f7;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 10px;
                font-weight: bold;
            }

            .peer-nodes .network-node {
                background: #66bb6a;
            }

            .network-stats {
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
            }

            .stat-item {
                text-align: center;
                flex: 1;
            }

            .stat-label {
                display: block;
                font-size: 12px;
                opacity: 0.8;
            }

            .stat-value {
                display: block;
                font-size: 18px;
                font-weight: bold;
                color: #4fc3f7;
            }
        `;

        if (!document.getElementById('p2p-network-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'p2p-network-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
    }

    updatePeerCount(count) {
        document.getElementById('peer-count').textContent = count;
        this.updatePeerVisualization(count);
    }

    updateLatency(latency) {
        document.getElementById('avg-latency').textContent = `${latency}ms`;
    }

    updateThroughput(throughput) {
        document.getElementById('throughput').textContent = throughput;
    }

    updatePeerVisualization(peerCount) {
        const peerContainer = document.getElementById('peer-nodes');
        peerContainer.innerHTML = '';

        for (let i = 0; i < Math.min(peerCount, 8); i++) {
            const peer = document.createElement('div');
            peer.className = 'network-node';
            peer.textContent = `P${i + 1}`;

            // Position peers in a circle around the local node
            const angle = (i / peerCount) * 2 * Math.PI;
            const x = 150 + 80 * Math.cos(angle);
            const y = 100 + 60 * Math.sin(angle);

            peer.style.left = `${x}px`;
            peer.style.top = `${y}px`;

            peerContainer.appendChild(peer);
        }

        // Position local node in center
        const localNode = document.getElementById('local-node');
        localNode.style.left = '150px';
        localNode.style.top = '100px';
    }
}

// Fog Computing Resource Chart Component
class FogResourceChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeComponent();
    }

    initializeComponent() {
        this.container.innerHTML = `
            <div class="resource-chart">
                <div class="resource-bar">
                    <div class="resource-label">CPU Utilization</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="cpu-progress"></div>
                        <span class="progress-text" id="cpu-text">0%</span>
                    </div>
                </div>
                <div class="resource-bar">
                    <div class="resource-label">Memory Usage</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="memory-progress"></div>
                        <span class="progress-text" id="memory-text">0%</span>
                    </div>
                </div>
                <div class="resource-bar">
                    <div class="resource-label">Network Bandwidth</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="bandwidth-progress"></div>
                        <span class="progress-text" id="bandwidth-text">0%</span>
                    </div>
                </div>
                <div class="resource-bar">
                    <div class="resource-label">Storage Capacity</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="storage-progress"></div>
                        <span class="progress-text" id="storage-text">0%</span>
                    </div>
                </div>
            </div>
        `;
        this.addStyles();
    }

    addStyles() {
        const styles = `
            .resource-chart {
                padding: 15px;
            }

            .resource-bar {
                margin: 15px 0;
            }

            .resource-label {
                font-size: 14px;
                margin-bottom: 5px;
                color: #e3f2fd;
            }

            .progress-bar {
                position: relative;
                width: 100%;
                height: 25px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4fc3f7, #29b6f6);
                border-radius: 12px;
                transition: width 0.3s ease;
                width: 0%;
            }

            .progress-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 12px;
                font-weight: bold;
                color: white;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
            }
        `;

        if (!document.getElementById('fog-resource-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'fog-resource-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
    }

    updateResource(type, percentage) {
        const progress = document.getElementById(`${type}-progress`);
        const text = document.getElementById(`${type}-text`);

        if (progress && text) {
            progress.style.width = `${percentage}%`;
            text.textContent = `${percentage}%`;

            // Color coding based on usage
            if (percentage > 80) {
                progress.style.background = 'linear-gradient(90deg, #f44336, #e53935)';
            } else if (percentage > 60) {
                progress.style.background = 'linear-gradient(90deg, #ff9800, #f57c00)';
            } else {
                progress.style.background = 'linear-gradient(90deg, #4fc3f7, #29b6f6)';
            }
        }
    }

    updateAllResources(resources) {
        this.updateResource('cpu', resources.cpu || 0);
        this.updateResource('memory', resources.memory || 0);
        this.updateResource('bandwidth', resources.bandwidth || 0);
        this.updateResource('storage', resources.storage || 0);
    }
}

// Token Economics Widget Component
class TokenEconomicsWidget {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeComponent();
    }

    initializeComponent() {
        this.container.innerHTML = `
            <div class="token-widget">
                <div class="token-balance">
                    <div class="balance-label">FOG Token Balance</div>
                    <div class="balance-value" id="token-balance">0 FOG</div>
                </div>
                <div class="token-stats">
                    <div class="token-stat">
                        <span class="stat-icon">⬆️</span>
                        <span class="stat-label">Earned Today</span>
                        <span class="stat-value" id="daily-earned">+0 FOG</span>
                    </div>
                    <div class="token-stat">
                        <span class="stat-icon">💰</span>
                        <span class="stat-label">Total Value</span>
                        <span class="stat-value" id="total-value">$0.00</span>
                    </div>
                    <div class="token-stat">
                        <span class="stat-icon">📊</span>
                        <span class="stat-label">Staked</span>
                        <span class="stat-value" id="staked-amount">0%</span>
                    </div>
                </div>
                <div class="token-actions">
                    <button class="token-btn" onclick="stakeFogTokens()">Stake</button>
                    <button class="token-btn" onclick="transferFogTokens()">Transfer</button>
                </div>
            </div>
        `;
        this.addStyles();
    }

    addStyles() {
        const styles = `
            .token-widget {
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
            }

            .token-balance {
                text-align: center;
                margin-bottom: 20px;
            }

            .balance-label {
                font-size: 14px;
                opacity: 0.8;
                margin-bottom: 5px;
            }

            .balance-value {
                font-size: 24px;
                font-weight: bold;
                color: #4fc3f7;
            }

            .token-stats {
                margin: 20px 0;
            }

            .token-stat {
                display: flex;
                align-items: center;
                margin: 10px 0;
                padding: 10px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }

            .stat-icon {
                margin-right: 10px;
                font-size: 16px;
            }

            .stat-label {
                flex: 1;
                font-size: 14px;
            }

            .stat-value {
                font-weight: bold;
                color: #4fc3f7;
            }

            .token-actions {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }

            .token-btn {
                flex: 1;
                padding: 10px;
                background: linear-gradient(45deg, #4fc3f7, #29b6f6);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s;
            }

            .token-btn:hover {
                transform: translateY(-2px);
            }
        `;

        if (!document.getElementById('token-widget-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'token-widget-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
    }

    updateBalance(balance) {
        document.getElementById('token-balance').textContent = `${balance.toLocaleString()} FOG`;
    }

    updateDailyEarned(earned) {
        document.getElementById('daily-earned').textContent = `+${earned.toFixed(1)} FOG`;
    }

    updateTotalValue(value) {
        document.getElementById('total-value').textContent = `$${value.toFixed(2)}`;
    }

    updateStakedAmount(percentage) {
        document.getElementById('staked-amount').textContent = `${percentage}%`;
    }

    updateAllStats(stats) {
        this.updateBalance(stats.balance || 0);
        this.updateDailyEarned(stats.dailyEarned || 0);
        this.updateTotalValue(stats.totalValue || 0);
        this.updateStakedAmount(stats.stakedPercentage || 0);
    }
}

// Privacy & Security Status Component
class PrivacySecurityStatus {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeComponent();
    }

    initializeComponent() {
        this.container.innerHTML = `
            <div class="privacy-status">
                <div class="privacy-indicator">
                    <div class="indicator-circle" id="privacy-circle"></div>
                    <div class="indicator-label">Privacy Level</div>
                    <div class="indicator-value" id="privacy-level">Unknown</div>
                </div>
                <div class="security-metrics">
                    <div class="security-item">
                        <span class="security-icon">🔒</span>
                        <span class="security-label">Onion Circuits</span>
                        <span class="security-value" id="circuit-count">0</span>
                    </div>
                    <div class="security-item">
                        <span class="security-icon">👤</span>
                        <span class="security-label">Anonymity Score</span>
                        <span class="security-value" id="anonymity-score">0/100</span>
                    </div>
                    <div class="security-item">
                        <span class="security-icon">🌐</span>
                        <span class="security-label">Hidden Services</span>
                        <span class="security-value" id="hidden-services">0</span>
                    </div>
                </div>
            </div>
        `;
        this.addStyles();
    }

    addStyles() {
        const styles = `
            .privacy-status {
                padding: 20px;
            }

            .privacy-indicator {
                text-align: center;
                margin-bottom: 20px;
            }

            .indicator-circle {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(45deg, #4caf50, #66bb6a);
                margin: 0 auto 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: white;
                position: relative;
            }

            .indicator-circle::after {
                content: '🛡️';
            }

            .indicator-label {
                font-size: 14px;
                opacity: 0.8;
                margin-bottom: 5px;
            }

            .indicator-value {
                font-size: 18px;
                font-weight: bold;
                color: #4fc3f7;
            }

            .security-metrics {
                margin-top: 20px;
            }

            .security-item {
                display: flex;
                align-items: center;
                margin: 12px 0;
                padding: 12px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }

            .security-icon {
                margin-right: 12px;
                font-size: 18px;
            }

            .security-label {
                flex: 1;
                font-size: 14px;
            }

            .security-value {
                font-weight: bold;
                color: #4fc3f7;
            }
        `;

        if (!document.getElementById('privacy-security-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'privacy-security-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
    }

    updatePrivacyLevel(level) {
        const indicator = document.getElementById('privacy-circle');
        const value = document.getElementById('privacy-level');

        value.textContent = level;

        // Update indicator color based on privacy level
        if (level === 'Very High' || level === 'Maximum') {
            indicator.style.background = 'linear-gradient(45deg, #4caf50, #66bb6a)';
        } else if (level === 'High') {
            indicator.style.background = 'linear-gradient(45deg, #2196f3, #42a5f5)';
        } else if (level === 'Medium') {
            indicator.style.background = 'linear-gradient(45deg, #ff9800, #ffb74d)';
        } else {
            indicator.style.background = 'linear-gradient(45deg, #f44336, #ef5350)';
        }
    }

    updateCircuitCount(count) {
        document.getElementById('circuit-count').textContent = count;
    }

    updateAnonymityScore(score) {
        document.getElementById('anonymity-score').textContent = `${score}/100`;
    }

    updateHiddenServices(count) {
        document.getElementById('hidden-services').textContent = count;
    }

    updateAllMetrics(metrics) {
        this.updatePrivacyLevel(metrics.privacyLevel || 'Unknown');
        this.updateCircuitCount(metrics.circuitCount || 0);
        this.updateAnonymityScore(metrics.anonymityScore || 0);
        this.updateHiddenServices(metrics.hiddenServices || 0);
    }
}

// Utility functions for token actions (to be implemented by consuming applications)
function stakeFogTokens() {
    alert('Staking functionality would be implemented here');
}

function transferFogTokens() {
    alert('Transfer functionality would be implemented here');
}

// Export components for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        P2PNetworkStatus,
        FogResourceChart,
        TokenEconomicsWidget,
        PrivacySecurityStatus
    };
}
