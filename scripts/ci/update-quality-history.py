#!/usr/bin/env python3
"""
Quality History Updater for CI/CD Pipeline
Maintains historical quality metrics for trend analysis.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class QualityHistoryManager:
    """Manages historical quality metrics for trend analysis."""
    
    def __init__(self, db_path: str = "quality_history.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the quality history database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create quality metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                branch TEXT DEFAULT 'main',
                overall_score REAL NOT NULL,
                coupling_score REAL,
                complexity_score REAL,
                god_objects_count INTEGER,
                magic_literals_density REAL,
                connascence_violations INTEGER,
                antipatterns_count INTEGER,
                test_coverage REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create trend analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                avg_overall_score REAL,
                trend_direction TEXT,  -- 'improving', 'declining', 'stable'
                quality_debt_score REAL,
                violations_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create quality alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                alert_type TEXT NOT NULL,  -- 'regression', 'threshold', 'trend'
                severity TEXT NOT NULL,    -- 'low', 'medium', 'high', 'critical'
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def update_quality_metrics(
        self,
        commit_sha: str,
        quality_score: float,
        reports_dir: Path,
        branch: str = "main"
    ):
        """Update quality metrics with latest data."""
        
        # Load additional metrics from reports
        coupling_data = self._load_coupling_data(reports_dir)
        antipatterns_data = self._load_antipatterns_data(reports_dir)
        quality_gate_data = self._load_quality_gate_data(reports_dir)
        
        # Extract individual metrics
        metrics = self._extract_metrics(
            quality_gate_data, coupling_data, antipatterns_data
        )
        
        # Store in database
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO quality_metrics (
                timestamp, commit_sha, branch, overall_score,
                coupling_score, complexity_score, god_objects_count,
                magic_literals_density, connascence_violations,
                antipatterns_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, commit_sha, branch, quality_score,
            metrics.get('coupling_score'),
            metrics.get('complexity_score'),
            metrics.get('god_objects_count'),
            metrics.get('magic_literals_density'),
            metrics.get('connascence_violations'),
            metrics.get('antipatterns_count')
        ))
        
        conn.commit()
        
        # Check for quality regressions
        self._check_quality_regression(cursor, commit_sha, quality_score, metrics)
        
        # Update daily trends
        self._update_daily_trends(cursor)
        
        conn.commit()
        conn.close()
        
        print(f"Quality metrics updated for commit {commit_sha[:8]}")
        print(f"Overall score: {quality_score:.1f}")
    
    def _extract_metrics(
        self,
        quality_gate_data: Dict,
        coupling_data: Dict,
        antipatterns_data: Dict
    ) -> Dict:
        """Extract individual metrics from report data."""
        metrics = {}
        
        # Extract from quality gate metrics
        if quality_gate_data and 'metrics' in quality_gate_data:
            for metric in quality_gate_data['metrics']:
                name = metric.get('name', '').lower()
                value = metric.get('value', 0)
                
                if 'coupling' in name:
                    metrics['coupling_score'] = value
                elif 'complexity' in name:
                    metrics['complexity_score'] = value
                elif 'god object' in name:
                    metrics['god_objects_count'] = int(value)
                elif 'magic literal' in name:
                    metrics['magic_literals_density'] = value
                elif 'connascence' in name:
                    metrics['connascence_violations'] = int(value)
        
        # Extract from coupling data
        if coupling_data:
            metrics['coupling_score'] = coupling_data.get('average_coupling_score', 0)
        
        # Extract from antipatterns data
        if antipatterns_data:
            patterns = antipatterns_data.get('detected_patterns', [])
            metrics['antipatterns_count'] = len(patterns)
        
        return metrics
    
    def _check_quality_regression(
        self,
        cursor: sqlite3.Cursor,
        commit_sha: str,
        current_score: float,
        current_metrics: Dict
    ):
        """Check for quality regressions and create alerts."""
        
        # Get previous scores for comparison
        cursor.execute("""
            SELECT overall_score, coupling_score, god_objects_count
            FROM quality_metrics
            WHERE commit_sha != ?
            ORDER BY timestamp DESC
            LIMIT 5
        """, (commit_sha,))
        
        recent_scores = cursor.fetchall()
        
        if not recent_scores:
            return  # No previous data to compare
        
        # Calculate average of recent scores
        avg_recent_score = sum(score[0] for score in recent_scores) / len(recent_scores)
        
        # Check for significant regression (>5 point drop)
        score_drop = avg_recent_score - current_score
        
        if score_drop > 5:
            self._create_alert(
                cursor, commit_sha, "regression", "high",
                f"Quality score dropped by {score_drop:.1f} points (from {avg_recent_score:.1f} to {current_score:.1f})"
            )
        
        # Check for specific metric regressions
        if len(recent_scores) > 0:
            prev_coupling = recent_scores[0][1] if recent_scores[0][1] else 0
            prev_god_objects = recent_scores[0][2] if recent_scores[0][2] else 0
            
            current_coupling = current_metrics.get('coupling_score', 0)
            current_god_objects = current_metrics.get('god_objects_count', 0)
            
            # Coupling regression
            if current_coupling > prev_coupling + 2:
                self._create_alert(
                    cursor, commit_sha, "regression", "medium",
                    f"Coupling score increased from {prev_coupling:.1f} to {current_coupling:.1f}"
                )
            
            # God objects regression
            if current_god_objects > prev_god_objects:
                self._create_alert(
                    cursor, commit_sha, "regression", "medium",
                    f"God objects count increased from {prev_god_objects} to {current_god_objects}"
                )
    
    def _create_alert(
        self,
        cursor: sqlite3.Cursor,
        commit_sha: str,
        alert_type: str,
        severity: str,
        message: str
    ):
        """Create a quality alert."""
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO quality_alerts (timestamp, commit_sha, alert_type, severity, message)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, commit_sha, alert_type, severity, message))
        
        print(f"🚨 Quality Alert ({severity}): {message}")
    
    def _update_daily_trends(self, cursor: sqlite3.Cursor):
        """Update daily trend analysis."""
        today = datetime.now().date().isoformat()
        
        # Calculate today's average score
        cursor.execute("""
            SELECT AVG(overall_score), COUNT(*)
            FROM quality_metrics
            WHERE DATE(timestamp) = ?
        """, (today,))
        
        result = cursor.fetchone()
        avg_score = result[0] if result[0] else 0
        count = result[1] if result[1] else 0
        
        if count == 0:
            return  # No data for today
        
        # Get yesterday's average for trend calculation
        cursor.execute("""
            SELECT avg_overall_score
            FROM quality_trends
            ORDER BY date DESC
            LIMIT 1
        """)
        
        prev_result = cursor.fetchone()
        prev_score = prev_result[0] if prev_result else avg_score
        
        # Determine trend direction
        score_diff = avg_score - prev_score
        if abs(score_diff) < 1:
            trend = "stable"
        elif score_diff > 0:
            trend = "improving"
        else:
            trend = "declining"
        
        # Calculate quality debt score (inverse of quality score)
        quality_debt = max(0, 100 - avg_score)
        
        # Count current violations
        cursor.execute("""
            SELECT SUM(god_objects_count + connascence_violations + antipatterns_count)
            FROM quality_metrics
            WHERE DATE(timestamp) = ?
        """, (today,))
        
        violations_result = cursor.fetchone()
        violations_count = violations_result[0] if violations_result[0] else 0
        
        # Insert or update today's trend
        cursor.execute("""
            INSERT OR REPLACE INTO quality_trends (
                date, avg_overall_score, trend_direction, quality_debt_score, violations_count
            ) VALUES (?, ?, ?, ?, ?)
        """, (today, avg_score, trend, quality_debt, violations_count))
    
    def _load_coupling_data(self, reports_dir: Path) -> Dict:
        """Load coupling analysis data."""
        coupling_file = reports_dir / "coupling_report.json"
        if coupling_file.exists():
            try:
                with open(coupling_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _load_antipatterns_data(self, reports_dir: Path) -> Dict:
        """Load anti-patterns data."""
        antipatterns_file = reports_dir / "antipatterns_report.json"
        if antipatterns_file.exists():
            try:
                with open(antipatterns_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _load_quality_gate_data(self, reports_dir: Path) -> Dict:
        """Load quality gate data."""
        quality_gate_file = reports_dir / "quality_gate_result.json"
        if quality_gate_file.exists():
            try:
                with open(quality_gate_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def get_quality_trends(self, days: int = 30) -> List[Dict]:
        """Get quality trends for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, avg_overall_score, trend_direction, quality_debt_score, violations_count
            FROM quality_trends
            ORDER BY date DESC
            LIMIT ?
        """, (days,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'date': row[0],
                'avg_score': row[1],
                'trend': row[2],
                'debt_score': row[3],
                'violations': row[4]
            }
            for row in results
        ]
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent quality alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, commit_sha, alert_type, severity, message, resolved
            FROM quality_alerts
            WHERE resolved = FALSE
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': row[0],
                'commit_sha': row[1],
                'type': row[2],
                'severity': row[3],
                'message': row[4],
                'resolved': bool(row[5])
            }
            for row in results
        ]


def main():
    """Main entry point for quality history management."""
    parser = argparse.ArgumentParser(description="Update quality history")
    parser.add_argument("--commit-sha", required=True,
                       help="Git commit SHA")
    parser.add_argument("--quality-score", type=float, required=True,
                       help="Overall quality score")
    parser.add_argument("--reports-dir", type=str, default=".",
                       help="Directory containing quality reports")
    parser.add_argument("--branch", type=str, default="main",
                       help="Git branch name")
    parser.add_argument("--db-path", type=str, default="quality_history.db",
                       help="Path to quality history database")
    
    args = parser.parse_args()
    
    # Initialize quality history manager
    history_manager = QualityHistoryManager(db_path=args.db_path)
    
    # Update quality metrics
    reports_dir = Path(args.reports_dir)
    history_manager.update_quality_metrics(
        commit_sha=args.commit_sha,
        quality_score=args.quality_score,
        reports_dir=reports_dir,
        branch=args.branch
    )
    
    # Show recent alerts
    alerts = history_manager.get_recent_alerts(limit=5)
    if alerts:
        print("\n🚨 Recent Quality Alerts:")
        for alert in alerts:
            severity_emoji = {
                'low': '📝',
                'medium': '⚠️',
                'high': '🚨',
                'critical': '🔥'
            }.get(alert['severity'], '📝')
            
            print(f"  {severity_emoji} {alert['message']} (commit: {alert['commit_sha'][:8]})")
    
    print(f"\nQuality history updated successfully!")


if __name__ == "__main__":
    main()