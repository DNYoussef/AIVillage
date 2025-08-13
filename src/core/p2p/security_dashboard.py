"""P2P Network Security Dashboard.

Real-time security monitoring and alerting for the P2P mesh network.
Provides web interface for security events, peer reputation, and threat detection.
"""

from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .secure_libp2p_mesh import SecurityEvent, SecurityLevel, SecurityMonitor

logger = logging.getLogger(__name__)


class SecurityDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for security dashboard."""

    def __init__(self, request, client_address, server, security_monitor: SecurityMonitor) -> None:
        self.security_monitor = security_monitor
        super().__init__(request, client_address, server)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/":
            self.serve_dashboard_html()
        elif path == "/api/security/summary":
            self.serve_security_summary()
        elif path == "/api/security/events":
            self.serve_security_events()
        elif path == "/api/security/peers":
            self.serve_peer_reputation()
        elif path == "/api/security/alerts":
            self.serve_security_alerts()
        elif path.startswith("/static/"):
            self.serve_static_file(path)
        else:
            self.send_error(404, "Not Found")

    def serve_dashboard_html(self) -> None:
        """Serve the main dashboard HTML."""
        html_content = self.get_dashboard_html()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html_content)))
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_security_summary(self) -> None:
        """Serve security summary data."""
        summary = self.security_monitor.get_security_summary()

        # Add additional metrics
        now = datetime.now()
        recent_threshold = now - timedelta(hours=1)

        recent_events = [log for log in self.security_monitor.security_logs if log.timestamp >= recent_threshold]

        critical_events = [log for log in recent_events if log.severity == SecurityLevel.CRITICAL]

        high_severity_events = [log for log in recent_events if log.severity == SecurityLevel.HIGH]

        enhanced_summary = {
            **summary,
            "recent_critical_events": len(critical_events),
            "recent_high_severity_events": len(high_severity_events),
            "security_health_score": self._calculate_security_health_score(),
            "threat_level": self._assess_threat_level(),
            "last_updated": now.isoformat(),
        }

        self.send_json_response(enhanced_summary)

    def serve_security_events(self) -> None:
        """Serve recent security events."""
        # Get query parameters
        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)

        limit = int(params.get("limit", ["100"])[0])
        severity_filter = params.get("severity", [None])[0]

        # Filter events
        events = list(self.security_monitor.security_logs)

        if severity_filter:
            try:
                severity_level = SecurityLevel(int(severity_filter))
                events = [e for e in events if e.severity == severity_level]
            except ValueError:
                pass

        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        events = events[:limit]

        # Convert to JSON-serializable format
        events_data = [
            {
                "timestamp": event.timestamp.isoformat(),
                "type": event.event_type.value,
                "peer_id": event.peer_id,
                "source_ip": event.source_ip,
                "severity": event.severity.value,
                "description": event.description,
                "metadata": event.metadata,
            }
            for event in events
        ]

        self.send_json_response(
            {
                "events": events_data,
                "total_count": len(self.security_monitor.security_logs),
                "filtered_count": len(events_data),
            }
        )

    def serve_peer_reputation(self) -> None:
        """Serve peer reputation data."""
        reputations = []

        for peer_id, reputation in self.security_monitor.peer_reputations.items():
            reputations.append(
                {
                    "peer_id": peer_id,
                    "trust_score": reputation.trust_score,
                    "successful_interactions": reputation.successful_interactions,
                    "failed_interactions": reputation.failed_interactions,
                    "last_interaction": reputation.last_interaction.isoformat(),
                    "blocked": reputation.blocked,
                    "first_seen": reputation.first_seen.isoformat(),
                    "total_connections": len(reputation.connection_attempts),
                    "reported_by_peers": len(reputation.reported_by_peers),
                }
            )

        # Sort by trust score (descending)
        reputations.sort(key=lambda x: x["trust_score"], reverse=True)

        self.send_json_response(
            {
                "peer_reputations": reputations,
                "total_peers": len(reputations),
                "blocked_peers": len(self.security_monitor.blocked_peers),
                "avg_trust_score": sum(r["trust_score"] for r in reputations) / max(1, len(reputations)),
            }
        )

    def serve_security_alerts(self) -> None:
        """Serve current security alerts."""
        alerts = []
        now = datetime.now()

        # Generate alerts based on current state

        # Critical peer blocks in last hour
        recent_blocks = [
            log
            for log in self.security_monitor.security_logs
            if (log.event_type == SecurityEvent.PEER_BLOCKED and (now - log.timestamp).total_seconds() < 3600)
        ]

        if len(recent_blocks) > 5:
            alerts.append(
                {
                    "id": "mass_peer_blocks",
                    "level": "critical",
                    "title": "Mass Peer Blocking Detected",
                    "description": f"{len(recent_blocks)} peers blocked in the last hour",
                    "timestamp": now.isoformat(),
                    "action_required": True,
                }
            )

        # High rate of authentication failures
        auth_failures = [
            log
            for log in self.security_monitor.security_logs
            if (log.event_type == SecurityEvent.AUTH_FAILURE and (now - log.timestamp).total_seconds() < 3600)
        ]

        if len(auth_failures) > 20:
            alerts.append(
                {
                    "id": "high_auth_failures",
                    "level": "high",
                    "title": "High Authentication Failure Rate",
                    "description": f"{len(auth_failures)} authentication failures in the last hour",
                    "timestamp": now.isoformat(),
                    "action_required": False,
                }
            )

        # Low average trust score
        if self.security_monitor.peer_reputations:
            avg_trust = sum(r.trust_score for r in self.security_monitor.peer_reputations.values()) / len(
                self.security_monitor.peer_reputations
            )

            if avg_trust < 0.4:
                alerts.append(
                    {
                        "id": "low_trust_network",
                        "level": "warning",
                        "title": "Low Network Trust Score",
                        "description": f"Average peer trust score: {avg_trust:.3f}",
                        "timestamp": now.isoformat(),
                        "action_required": False,
                    }
                )

        # Replay attacks detected
        replay_attacks = [
            log
            for log in self.security_monitor.security_logs
            if (log.event_type == SecurityEvent.REPLAY_ATTACK_DETECTED and (now - log.timestamp).total_seconds() < 3600)
        ]

        if replay_attacks:
            alerts.append(
                {
                    "id": "replay_attacks",
                    "level": "high",
                    "title": "Replay Attacks Detected",
                    "description": f"{len(replay_attacks)} replay attacks in the last hour",
                    "timestamp": now.isoformat(),
                    "action_required": True,
                }
            )

        self.send_json_response(
            {
                "alerts": alerts,
                "alert_count": len(alerts),
                "last_updated": now.isoformat(),
            }
        )

    def _calculate_security_health_score(self) -> float:
        """Calculate overall security health score (0.0 to 1.0)."""
        score = 1.0

        # Reduce score based on blocked peers
        blocked_count = len(self.security_monitor.blocked_peers)
        score -= min(0.3, blocked_count * 0.01)

        # Reduce score based on recent critical events
        now = datetime.now()
        recent_critical = [
            log
            for log in self.security_monitor.security_logs
            if (log.severity == SecurityLevel.CRITICAL and (now - log.timestamp).total_seconds() < 3600)
        ]
        score -= min(0.4, len(recent_critical) * 0.05)

        # Reduce score based on low average trust
        if self.security_monitor.peer_reputations:
            avg_trust = sum(r.trust_score for r in self.security_monitor.peer_reputations.values()) / len(
                self.security_monitor.peer_reputations
            )
            if avg_trust < 0.5:
                score -= 0.5 - avg_trust

        return max(0.0, score)

    def _assess_threat_level(self) -> str:
        """Assess current threat level."""
        health_score = self._calculate_security_health_score()

        if health_score >= 0.8:
            return "low"
        if health_score >= 0.6:
            return "medium"
        if health_score >= 0.4:
            return "high"
        return "critical"

    def send_json_response(self, data: dict[str, Any]) -> None:
        """Send JSON response."""
        json_data = json.dumps(data, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(json_data)))
        self.send_header("Access-Control-Allow-Origin", "*")  # For development
        self.end_headers()
        self.wfile.write(json_data.encode())

    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIVillage P2P Security Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1e1e2e;
            color: #cdd6f4;
        }

        .header {
            background: linear-gradient(135deg, #89b4fa, #cba6f7);
            color: #1e1e2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: #313244;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .card h3 {
            margin: 0 0 15px 0;
            color: #f38ba8;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }

        .metric-value {
            font-weight: bold;
            color: #94e2d5;
        }

        .alert {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }

        .alert-critical {
            background: rgba(243, 139, 168, 0.2);
            border-left: 4px solid #f38ba8;
        }

        .alert-high {
            background: rgba(250, 179, 135, 0.2);
            border-left: 4px solid #fab387;
        }

        .alert-warning {
            background: rgba(249, 226, 175, 0.2);
            border-left: 4px solid #f9e2af;
        }

        .threat-level {
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            text-align: center;
            display: inline-block;
        }

        .threat-low { background: #a6e3a1; color: #1e1e2e; }
        .threat-medium { background: #f9e2af; color: #1e1e2e; }
        .threat-high { background: #fab387; color: #1e1e2e; }
        .threat-critical { background: #f38ba8; color: #1e1e2e; }

        .events-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .events-table th,
        .events-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #45475a;
        }

        .events-table th {
            background: #45475a;
            font-weight: bold;
        }

        .refresh-btn {
            background: #89b4fa;
            color: #1e1e2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }

        .refresh-btn:hover {
            background: #74c7ec;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AIVillage P2P Security Dashboard</h1>
        <p>Real-time monitoring of mesh network security</p>
        <button class="refresh-btn" onclick="refreshDashboard()">Refresh Data</button>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h3>Security Overview</h3>
            <div id="security-overview">Loading...</div>
        </div>

        <div class="card">
            <h3>Active Alerts</h3>
            <div id="security-alerts">Loading...</div>
        </div>

        <div class="card">
            <h3>Peer Reputation</h3>
            <div id="peer-reputation">Loading...</div>
        </div>

        <div class="card">
            <h3>Network Statistics</h3>
            <div id="network-stats">Loading...</div>
        </div>
    </div>

    <div class="card">
        <h3>Recent Security Events</h3>
        <div id="recent-events">Loading...</div>
    </div>

    <script>
        let refreshInterval;

        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                return null;
            }
        }

        async function updateSecurityOverview() {
            const data = await fetchData('/api/security/summary');
            if (!data) return;

            const threatLevel = data.threat_level || 'unknown';
            const healthScore = (data.security_health_score * 100).toFixed(1);

            document.getElementById('security-overview').innerHTML = `
                <div class="metric">
                    <span>Threat Level:</span>
                    <span class="threat-level threat-${threatLevel}">${threatLevel.toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span>Security Health:</span>
                    <span class="metric-value">${healthScore}%</span>
                </div>
                <div class="metric">
                    <span>Total Events:</span>
                    <span class="metric-value">${data.total_events}</span>
                </div>
                <div class="metric">
                    <span>Recent Events (1h):</span>
                    <span class="metric-value">${data.recent_events_1h}</span>
                </div>
                <div class="metric">
                    <span>Blocked Peers:</span>
                    <span class="metric-value">${data.blocked_peers}</span>
                </div>
                <div class="metric">
                    <span>Average Trust:</span>
                    <span class="metric-value">${data.avg_trust_score.toFixed(3)}</span>
                </div>
            `;
        }

        async function updateSecurityAlerts() {
            const data = await fetchData('/api/security/alerts');
            if (!data) return;

            if (data.alerts.length === 0) {
                document.getElementById('security-alerts').innerHTML = '<p>No active alerts</p>';
                return;
            }

            let alertsHtml = '';
            for (const alert of data.alerts) {
                alertsHtml += `
                    <div class="alert alert-${alert.level}">
                        <strong>${alert.title}</strong><br>
                        ${alert.description}
                        ${alert.action_required ? '<br><em>Action Required</em>' : ''}
                    </div>
                `;
            }

            document.getElementById('security-alerts').innerHTML = alertsHtml;
        }

        async function updatePeerReputation() {
            const data = await fetchData('/api/security/peers');
            if (!data) return;

            const topPeers = data.peer_reputations.slice(0, 5);
            let peersHtml = '';

            for (const peer of topPeers) {
                const trustColor = peer.trust_score >= 0.7 ? '#a6e3a1' :
                                 peer.trust_score >= 0.4 ? '#f9e2af' : '#f38ba8';

                peersHtml += `
                    <div class="metric">
                        <span>${peer.peer_id.substring(0, 12)}...</span>
                        <span style="color: ${trustColor}">${peer.trust_score.toFixed(3)}</span>
                    </div>
                `;
            }

            document.getElementById('peer-reputation').innerHTML = `
                <div class="metric">
                    <span>Total Peers:</span>
                    <span class="metric-value">${data.total_peers}</span>
                </div>
                <div class="metric">
                    <span>Blocked:</span>
                    <span class="metric-value">${data.blocked_peers}</span>
                </div>
                <hr style="border-color: #45475a; margin: 15px 0;">
                <strong>Top Trusted Peers:</strong>
                ${peersHtml}
            `;
        }

        async function updateNetworkStats() {
            const data = await fetchData('/api/security/summary');
            if (!data) return;

            let eventTypesHtml = '';
            for (const [eventType, count] of Object.entries(data.event_types || {})) {
                eventTypesHtml += `
                    <div class="metric">
                        <span>${eventType.replace(/_/g, ' ')}:</span>
                        <span class="metric-value">${count}</span>
                    </div>
                `;
            }

            document.getElementById('network-stats').innerHTML = `
                <strong>Event Types (Recent):</strong>
                ${eventTypesHtml}
                <div class="metric">
                    <span>Critical Events (1h):</span>
                    <span class="metric-value" style="color: #f38ba8">${data.recent_critical_events || 0}</span>
                </div>
                <div class="metric">
                    <span>High Severity (1h):</span>
                    <span class="metric-value" style="color: #fab387">${data.recent_high_severity_events || 0}</span>
                </div>
            `;
        }

        async function updateRecentEvents() {
            const data = await fetchData('/api/security/events?limit=10');
            if (!data) return;

            if (data.events.length === 0) {
                document.getElementById('recent-events').innerHTML = '<p>No recent events</p>';
                return;
            }

            let tableHtml = `
                <table class="events-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Event</th>
                            <th>Peer</th>
                            <th>Severity</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            for (const event of data.events) {
                const timestamp = new Date(event.timestamp).toLocaleString();
                const severityColor = {
                    1: '#94e2d5',
                    2: '#f9e2af',
                    3: '#fab387',
                    4: '#f38ba8'
                }[event.severity] || '#cdd6f4';

                tableHtml += `
                    <tr>
                        <td>${timestamp}</td>
                        <td>${event.type.replace(/_/g, ' ')}</td>
                        <td>${event.peer_id ? event.peer_id.substring(0, 12) + '...' : 'N/A'}</td>
                        <td style="color: ${severityColor}">${event.severity}</td>
                        <td>${event.description}</td>
                    </tr>
                `;
            }

            tableHtml += '</tbody></table>';
            document.getElementById('recent-events').innerHTML = tableHtml;
        }

        async function refreshDashboard() {
            await Promise.all([
                updateSecurityOverview(),
                updateSecurityAlerts(),
                updatePeerReputation(),
                updateNetworkStats(),
                updateRecentEvents()
            ]);
        }

        // Initial load and auto-refresh
        refreshDashboard();
        refreshInterval = setInterval(refreshDashboard, 30000); // Refresh every 30 seconds

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>"""

    def log_message(self, format, *args) -> None:
        """Suppress default logging."""


class SecurityDashboardServer:
    """Security dashboard web server."""

    def __init__(self, security_monitor: SecurityMonitor, port: int = 8083) -> None:
        self.security_monitor = security_monitor
        self.port = port
        self.server = None
        self.running = False

    def start(self) -> None:
        """Start the dashboard server."""
        if self.running:
            return

        # Create custom handler with security monitor
        class CustomHandler(SecurityDashboardHandler):
            def __init__(self, request, client_address, server) -> None:
                super().__init__(request, client_address, server, server.security_monitor)

        # Create server
        self.server = HTTPServer(("localhost", self.port), CustomHandler)
        self.server.security_monitor = self.security_monitor

        self.running = True

        logger.info(f"Security dashboard started at http://localhost:{self.port}")
        print(f"\n{'=' * 60}")
        print("P2P SECURITY DASHBOARD STARTED")
        print(f"URL: http://localhost:{self.port}")
        print("Features:")
        print("  - Real-time security monitoring")
        print("  - Threat level assessment")
        print("  - Peer reputation tracking")
        print("  - Security event logging")
        print("  - Automated alerting")
        print(f"{'=' * 60}\n")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self.server and self.running:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            logger.info("Security dashboard stopped")


def start_security_dashboard(security_monitor: SecurityMonitor, port: int = 8083):
    """Start the security dashboard server."""
    dashboard = SecurityDashboardServer(security_monitor, port)
    dashboard.start()
    return dashboard


# Standalone server for testing
if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from secure_libp2p_mesh import SecureP2PNetworkConfig, SecurityMonitor

    # Create mock security monitor with sample data
    config = SecureP2PNetworkConfig()
    monitor = SecurityMonitor(config)

    # Add some sample data
    from datetime import datetime

    # Sample security events
    events = [
        (
            "peer_test_1",
            SecurityEvent.CONNECTION_ATTEMPT,
            SecurityLevel.LOW,
            "Test connection",
        ),
        (
            "peer_test_2",
            SecurityEvent.AUTH_FAILURE,
            SecurityLevel.MEDIUM,
            "Authentication failed",
        ),
        (
            "peer_malicious",
            SecurityEvent.RATE_LIMIT_EXCEEDED,
            SecurityLevel.HIGH,
            "Too many connections",
        ),
        (
            "peer_attacker",
            SecurityEvent.MALICIOUS_PEER_DETECTED,
            SecurityLevel.CRITICAL,
            "Malicious behavior",
        ),
    ]

    for peer_id, event_type, severity, description in events:
        event = SecurityEventLog(
            event_type=event_type,
            peer_id=peer_id,
            severity=severity,
            description=description,
        )
        monitor.log_security_event(event)

        # Update reputation
        delta = -0.3 if severity.value >= 3 else 0.1
        monitor.update_peer_reputation(peer_id, delta, description)

    # Block malicious peer
    monitor.block_peer("peer_attacker", "Detected malicious behavior")

    print("Starting security dashboard with sample data...")
    start_security_dashboard(monitor)
