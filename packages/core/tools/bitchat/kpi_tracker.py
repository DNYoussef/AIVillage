#!/usr/bin/env python3
"""
BitChat KPI Tracker and Instrumentation Tools

Tracks critical metrics for BitChat mesh network performance:
- Message delivery rates
- Network latency measurements
- Peer connectivity statistics
- Battery impact assessment
- Store-and-forward effectiveness
"""

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class MessageStatus(Enum):
    """Message delivery status tracking"""

    SENT = "sent"
    DELIVERED = "delivered"
    STORED = "stored"
    FORWARDED = "forwarded"
    DROPPED = "dropped"
    EXPIRED = "expired"


@dataclass
class MessageMetrics:
    """Metrics for individual message tracking"""

    message_id: str
    timestamp: float
    sender_id: str
    recipient_id: str | None
    message_size: int
    hop_count: int
    ttl: int
    status: MessageStatus
    latency_ms: float | None = None
    delivery_time: float | None = None
    transport_type: str | None = None


@dataclass
class PeerMetrics:
    """Metrics for peer connectivity"""

    peer_id: str
    platform: str
    first_seen: float
    last_seen: float
    messages_sent: int = 0
    messages_received: int = 0
    messages_relayed: int = 0
    connection_quality: float = 1.0
    battery_level: float | None = None
    is_active: bool = True


@dataclass
class NetworkSnapshot:
    """Network-wide performance snapshot"""

    timestamp: float
    active_peers: int
    total_peers: int
    messages_per_minute: float
    avg_latency_ms: float
    delivery_rate: float
    store_forward_rate: float
    avg_hop_count: float
    network_density: float
    battery_impact: float | None = None


class BitChatKPITracker:
    """Main KPI tracking and analysis system"""

    def __init__(self, db_path: str = "bitchat_kpis.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        # Message metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS message_metrics (
                message_id TEXT PRIMARY KEY,
                timestamp REAL,
                sender_id TEXT,
                recipient_id TEXT,
                message_size INTEGER,
                hop_count INTEGER,
                ttl INTEGER,
                status TEXT,
                latency_ms REAL,
                delivery_time REAL,
                transport_type TEXT
            )
        """
        )

        # Peer metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_metrics (
                peer_id TEXT PRIMARY KEY,
                platform TEXT,
                first_seen REAL,
                last_seen REAL,
                messages_sent INTEGER,
                messages_received INTEGER,
                messages_relayed INTEGER,
                connection_quality REAL,
                battery_level REAL,
                is_active INTEGER
            )
        """
        )

        # Network snapshots table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS network_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                active_peers INTEGER,
                total_peers INTEGER,
                messages_per_minute REAL,
                avg_latency_ms REAL,
                delivery_rate REAL,
                store_forward_rate REAL,
                avg_hop_count REAL,
                network_density REAL,
                battery_impact REAL
            )
        """
        )

        self.conn.commit()

    def record_message(self, metrics: MessageMetrics):
        """Record message transmission metrics"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO message_metrics
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.message_id,
                metrics.timestamp,
                metrics.sender_id,
                metrics.recipient_id,
                metrics.message_size,
                metrics.hop_count,
                metrics.ttl,
                metrics.status.value,
                metrics.latency_ms,
                metrics.delivery_time,
                metrics.transport_type,
            ),
        )
        self.conn.commit()

    def update_peer(self, peer: PeerMetrics):
        """Update peer connectivity metrics"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO peer_metrics
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                peer.peer_id,
                peer.platform,
                peer.first_seen,
                peer.last_seen,
                peer.messages_sent,
                peer.messages_received,
                peer.messages_relayed,
                peer.connection_quality,
                peer.battery_level,
                1 if peer.is_active else 0,
            ),
        )
        self.conn.commit()

    def calculate_delivery_rate(self, window_minutes: int = 5) -> float:
        """Calculate message delivery success rate"""
        cursor = self.conn.cursor()
        cutoff = time.time() - (window_minutes * 60)

        cursor.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) as delivered
            FROM message_metrics
            WHERE timestamp > ?
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        if row and row["total"] > 0:
            return row["delivered"] / row["total"]
        return 0.0

    def calculate_avg_latency(self, window_minutes: int = 5) -> float:
        """Calculate average message latency"""
        cursor = self.conn.cursor()
        cutoff = time.time() - (window_minutes * 60)

        cursor.execute(
            """
            SELECT AVG(latency_ms) as avg_latency
            FROM message_metrics
            WHERE timestamp > ? AND latency_ms IS NOT NULL
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        return row["avg_latency"] if row and row["avg_latency"] else 0.0

    def get_active_peer_count(self, timeout_seconds: int = 60) -> int:
        """Get count of currently active peers"""
        cursor = self.conn.cursor()
        cutoff = time.time() - timeout_seconds

        cursor.execute(
            """
            SELECT COUNT(*) as active_count
            FROM peer_metrics
            WHERE last_seen > ? AND is_active = 1
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        return row["active_count"] if row else 0

    def calculate_network_density(self) -> float:
        """Calculate network connection density"""
        active_peers = self.get_active_peer_count()
        if active_peers <= 1:
            return 0.0

        # Estimate based on average connections per peer
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT AVG(messages_sent + messages_received) as avg_messages
            FROM peer_metrics
            WHERE is_active = 1
        """
        )

        row = cursor.fetchone()
        if row and row["avg_messages"]:
            # Rough density estimate based on message patterns
            max_connections = active_peers * (active_peers - 1) / 2
            estimated_connections = min(row["avg_messages"] / 10, max_connections)
            return estimated_connections / max_connections if max_connections > 0 else 0.0
        return 0.0

    def take_network_snapshot(self) -> NetworkSnapshot:
        """Capture current network performance snapshot"""
        snapshot = NetworkSnapshot(
            timestamp=time.time(),
            active_peers=self.get_active_peer_count(),
            total_peers=self._get_total_peer_count(),
            messages_per_minute=self._calculate_message_rate(),
            avg_latency_ms=self.calculate_avg_latency(),
            delivery_rate=self.calculate_delivery_rate(),
            store_forward_rate=self._calculate_store_forward_rate(),
            avg_hop_count=self._calculate_avg_hop_count(),
            network_density=self.calculate_network_density(),
            battery_impact=self._estimate_battery_impact(),
        )

        # Store snapshot
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO network_snapshots
            VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                snapshot.timestamp,
                snapshot.active_peers,
                snapshot.total_peers,
                snapshot.messages_per_minute,
                snapshot.avg_latency_ms,
                snapshot.delivery_rate,
                snapshot.store_forward_rate,
                snapshot.avg_hop_count,
                snapshot.network_density,
                snapshot.battery_impact,
            ),
        )
        self.conn.commit()

        return snapshot

    def _get_total_peer_count(self) -> int:
        """Get total number of known peers"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM peer_metrics")
        row = cursor.fetchone()
        return row["total"] if row else 0

    def _calculate_message_rate(self) -> float:
        """Calculate messages per minute"""
        cursor = self.conn.cursor()
        cutoff = time.time() - 60

        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM message_metrics
            WHERE timestamp > ?
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        return row["count"] if row else 0.0

    def _calculate_store_forward_rate(self) -> float:
        """Calculate store-and-forward usage rate"""
        cursor = self.conn.cursor()
        cutoff = time.time() - 300  # 5 minutes

        cursor.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('stored', 'forwarded') THEN 1 ELSE 0 END) as sf_count
            FROM message_metrics
            WHERE timestamp > ?
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        if row and row["total"] > 0:
            return row["sf_count"] / row["total"]
        return 0.0

    def _calculate_avg_hop_count(self) -> float:
        """Calculate average hop count for delivered messages"""
        cursor = self.conn.cursor()
        cutoff = time.time() - 300

        cursor.execute(
            """
            SELECT AVG(hop_count) as avg_hops
            FROM message_metrics
            WHERE timestamp > ? AND status = 'delivered'
        """,
            (cutoff,),
        )

        row = cursor.fetchone()
        return row["avg_hops"] if row and row["avg_hops"] else 0.0

    def _estimate_battery_impact(self) -> float | None:
        """Estimate battery impact from peer reports"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(battery_level) as avg_battery
            FROM peer_metrics
            WHERE is_active = 1 AND battery_level IS NOT NULL
        """
        )

        row = cursor.fetchone()
        return row["avg_battery"] if row and row["avg_battery"] else None

    def generate_report(self) -> dict:
        """Generate comprehensive KPI report"""
        snapshot = self.take_network_snapshot()

        # Get historical data for trends
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM network_snapshots
            ORDER BY timestamp DESC
            LIMIT 100
        """
        )

        history = [dict(row) for row in cursor.fetchall()]

        # Calculate trends
        if len(history) >= 2:
            recent = history[0]
            older = history[min(10, len(history) - 1)]

            trends = {
                "delivery_rate_trend": recent["delivery_rate"] - older["delivery_rate"],
                "latency_trend": recent["avg_latency_ms"] - older["avg_latency_ms"],
                "peer_growth": recent["active_peers"] - older["active_peers"],
            }
        else:
            trends = {}

        return {
            "timestamp": datetime.now().isoformat(),
            "current_snapshot": asdict(snapshot),
            "trends": trends,
            "kpis": {
                "target_delivery_rate": 0.95,
                "actual_delivery_rate": snapshot.delivery_rate,
                "target_latency_ms": 100,
                "actual_latency_ms": snapshot.avg_latency_ms,
                "target_active_peers": 50,
                "actual_active_peers": snapshot.active_peers,
                "network_health_score": self._calculate_health_score(snapshot),
            },
            "recommendations": self._generate_recommendations(snapshot),
        }

    def _calculate_health_score(self, snapshot: NetworkSnapshot) -> float:
        """Calculate overall network health score (0-100)"""
        scores = []

        # Delivery rate score (40% weight)
        scores.append(min(snapshot.delivery_rate / 0.95, 1.0) * 40)

        # Latency score (30% weight)
        if snapshot.avg_latency_ms > 0:
            scores.append(min(100 / snapshot.avg_latency_ms, 1.0) * 30)
        else:
            scores.append(30)

        # Peer count score (20% weight)
        scores.append(min(snapshot.active_peers / 50, 1.0) * 20)

        # Network density score (10% weight)
        scores.append(snapshot.network_density * 10)

        return sum(scores)

    def _generate_recommendations(self, snapshot: NetworkSnapshot) -> list[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        if snapshot.delivery_rate < 0.9:
            recommendations.append("Low delivery rate: Consider increasing TTL or improving store-and-forward")

        if snapshot.avg_latency_ms > 150:
            recommendations.append("High latency: Optimize routing algorithms or reduce hop counts")

        if snapshot.active_peers < 10:
            recommendations.append("Low peer count: Improve discovery mechanisms or incentivize participation")

        if snapshot.network_density < 0.3:
            recommendations.append("Low network density: Encourage more peer connections")

        if snapshot.battery_impact and snapshot.battery_impact < 0.3:
            recommendations.append("High battery drain: Optimize power management and duty cycling")

        return recommendations

    def export_metrics(self, output_file: str = "bitchat_metrics.json"):
        """Export metrics to JSON file"""
        report = self.generate_report()

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Metrics exported to {output_file}")

    def close(self):
        """Close database connection"""
        self.conn.close()


def simulate_test_data(tracker: BitChatKPITracker):
    """Generate simulated test data for demonstration"""
    import random
    import uuid

    # Create test peers
    platforms = ["android", "ios", "linux"]
    peers = []
    for i in range(20):
        peer = PeerMetrics(
            peer_id=f"peer_{i}",
            platform=random.choice(platforms),
            first_seen=time.time() - random.randint(0, 3600),
            last_seen=time.time() - random.randint(0, 60),
            messages_sent=random.randint(10, 100),
            messages_received=random.randint(10, 100),
            messages_relayed=random.randint(5, 50),
            connection_quality=random.uniform(0.7, 1.0),
            battery_level=random.uniform(0.2, 1.0),
            is_active=random.random() > 0.2,
        )
        peers.append(peer)
        tracker.update_peer(peer)

    # Create test messages
    statuses = [MessageStatus.DELIVERED] * 8 + [MessageStatus.DROPPED] * 1 + [MessageStatus.STORED] * 1
    for _ in range(100):
        sender = random.choice(peers)
        recipient = random.choice([p for p in peers if p != sender])

        message = MessageMetrics(
            message_id=str(uuid.uuid4()),
            timestamp=time.time() - random.randint(0, 300),
            sender_id=sender.peer_id,
            recipient_id=recipient.peer_id,
            message_size=random.randint(100, 10000),
            hop_count=random.randint(1, 5),
            ttl=random.randint(3, 7),
            status=random.choice(statuses),
            latency_ms=random.uniform(50, 200) if random.random() > 0.2 else None,
            delivery_time=random.uniform(1, 10) if random.random() > 0.3 else None,
            transport_type=random.choice(["bluetooth", "wifi_direct", "multipeer"]),
        )
        tracker.record_message(message)

    print("Test data generated successfully")


if __name__ == "__main__":
    # Initialize tracker
    tracker = BitChatKPITracker("bitchat_test_kpis.db")

    # Generate test data
    print("Generating test data...")
    simulate_test_data(tracker)

    # Take network snapshot
    print("\nCapturing network snapshot...")
    snapshot = tracker.take_network_snapshot()

    # Print current KPIs
    print("\n=== BitChat Network KPIs ===")
    print(f"Active Peers: {snapshot.active_peers}")
    print(f"Total Peers: {snapshot.total_peers}")
    print(f"Delivery Rate: {snapshot.delivery_rate:.2%}")
    print(f"Avg Latency: {snapshot.avg_latency_ms:.1f} ms")
    print(f"Messages/min: {snapshot.messages_per_minute:.1f}")
    print(f"Store-Forward Rate: {snapshot.store_forward_rate:.2%}")
    print(f"Avg Hop Count: {snapshot.avg_hop_count:.1f}")
    print(f"Network Density: {snapshot.network_density:.2%}")

    # Generate and save report
    print("\nGenerating comprehensive report...")
    report = tracker.generate_report()

    print(f"\nNetwork Health Score: {report['kpis']['network_health_score']:.1f}/100")

    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    # Export metrics
    tracker.export_metrics("bitchat_kpi_report.json")

    # Cleanup
    tracker.close()
    print("\nKPI tracking complete!")
