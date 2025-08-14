#!/usr/bin/env python3
"""
Battle System Monitor

Monitors the Sword/Shield security battle system and provides:
- Real-time battle performance tracking
- Historical analysis and trends
- Agent improvement recommendations
- King Agent status reports
- System health monitoring

Usage:
    python battle_monitor.py [--report-interval MINUTES] [--king-reports]
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from software.meta_agents import BattleMetrics, BattleOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("battle_monitor.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class BattleSystemMonitor:
    """
    Monitors the Sword/Shield security battle system performance.
    """

    def __init__(
        self, report_interval_minutes: int = 60, enable_king_reports: bool = False
    ):
        self.report_interval = timedelta(minutes=report_interval_minutes)
        self.enable_king_reports = enable_king_reports
        self.last_report_time = datetime.now()

        # Monitoring data
        self.performance_history: list[dict[str, Any]] = []
        self.battle_history: list[BattleMetrics] = []
        self.agent_status_history: list[dict[str, Any]] = []

        logger.info(
            f"Battle System Monitor initialized - Report interval: {report_interval_minutes} minutes"
        )

    async def start_monitoring(self, orchestrator: BattleOrchestrator):
        """
        Start continuous monitoring of the battle system.

        Args:
            orchestrator: The battle orchestrator to monitor
        """
        logger.info("🔍 Starting battle system monitoring...")

        try:
            while True:
                # Collect current status
                await self._collect_system_status(orchestrator)

                # Check if it's time for a report
                if datetime.now() - self.last_report_time >= self.report_interval:
                    await self._generate_status_report(orchestrator)
                    self.last_report_time = datetime.now()

                # Sleep before next monitoring cycle
                await asyncio.sleep(300)  # Check every 5 minutes

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")

    async def _collect_system_status(self, orchestrator: BattleOrchestrator):
        """Collect current system status data."""
        try:
            timestamp = datetime.now()

            # Get orchestrator status
            orchestrator_status = orchestrator.get_orchestrator_status()

            # Get agent statuses if available
            sword_status = None
            shield_status = None

            if orchestrator.sword_agent:
                sword_status = orchestrator.sword_agent.get_offensive_status()

            if orchestrator.shield_agent:
                shield_status = orchestrator.shield_agent.get_defensive_status()

            # Compile status snapshot
            status_snapshot = {
                "timestamp": timestamp.isoformat(),
                "orchestrator": orchestrator_status,
                "sword_agent": sword_status,
                "shield_agent": shield_status,
                "system_health": self._assess_system_health(
                    orchestrator_status, sword_status, shield_status
                ),
            }

            self.agent_status_history.append(status_snapshot)

            # Keep only last 24 hours of status history
            cutoff_time = timestamp - timedelta(hours=24)
            self.agent_status_history = [
                s
                for s in self.agent_status_history
                if datetime.fromisoformat(s["timestamp"]) > cutoff_time
            ]

        except Exception as e:
            logger.error(f"Error collecting system status: {e}")

    def _assess_system_health(
        self,
        orchestrator_status: dict,
        sword_status: dict | None,
        shield_status: dict | None,
    ) -> dict[str, Any]:
        """Assess overall system health."""
        health_score = 100
        issues = []
        warnings = []

        # Check orchestrator health
        if orchestrator_status["status"] != "active":
            health_score -= 50
            issues.append("orchestrator_inactive")

        if orchestrator_status["available_scenarios"] == 0:
            health_score -= 30
            issues.append("no_battle_scenarios")

        # Check agent health
        if sword_status:
            if sword_status["status"] != "active":
                health_score -= 25
                issues.append("sword_agent_inactive")

            if sword_status["attack_techniques_count"] == 0:
                health_score -= 15
                warnings.append("sword_no_techniques")
        else:
            health_score -= 25
            issues.append("sword_agent_unavailable")

        if shield_status:
            if shield_status["status"] != "active":
                health_score -= 25
                issues.append("shield_agent_inactive")

            if shield_status["defensive_patterns_count"] == 0:
                health_score -= 15
                warnings.append("shield_no_patterns")
        else:
            health_score -= 25
            issues.append("shield_agent_unavailable")

        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        elif health_score >= 40:
            status = "poor"
        else:
            status = "critical"

        return {
            "health_score": max(0, health_score),
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "assessment_time": datetime.now().isoformat(),
        }

    async def _generate_status_report(self, orchestrator: BattleOrchestrator):
        """Generate comprehensive status report."""
        try:
            logger.info("📊 Generating battle system status report...")

            # Get current status
            current_status = (
                self.agent_status_history[-1] if self.agent_status_history else None
            )

            if not current_status:
                logger.warning("No status data available for reporting")
                return

            # Analyze recent battle history
            recent_battles = self._get_recent_battles(orchestrator)
            battle_analysis = self._analyze_battle_performance(recent_battles)

            # Generate trend analysis
            trend_analysis = self._analyze_performance_trends()

            # Compile comprehensive report
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "reporting_period": {
                    "start_time": (datetime.now() - self.report_interval).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_minutes": self.report_interval.total_seconds() / 60,
                },
                "system_status": current_status,
                "battle_performance": battle_analysis,
                "performance_trends": trend_analysis,
                "recommendations": self._generate_recommendations(
                    current_status, battle_analysis, trend_analysis
                ),
            }

            # Log report summary
            self._log_report_summary(report)

            # Send to King Agent if enabled
            if self.enable_king_reports:
                await self._send_king_report(report)

            # Store report
            self.performance_history.append(report)

            # Keep only last 7 days of reports
            cutoff_time = datetime.now() - timedelta(days=7)
            self.performance_history = [
                r
                for r in self.performance_history
                if datetime.fromisoformat(r["report_timestamp"]) > cutoff_time
            ]

        except Exception as e:
            logger.error(f"Error generating status report: {e}")

    def _get_recent_battles(
        self, orchestrator: BattleOrchestrator
    ) -> list[BattleMetrics]:
        """Get recent battle history from orchestrator."""
        # Get battles from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        cutoff_date = cutoff_time.strftime("%Y-%m-%d")

        recent_battles = [
            battle
            for battle in orchestrator.battle_history
            if battle.battle_date >= cutoff_date
        ]

        return recent_battles

    def _analyze_battle_performance(
        self, recent_battles: list[BattleMetrics]
    ) -> dict[str, Any]:
        """Analyze recent battle performance."""
        if not recent_battles:
            return {
                "battles_analyzed": 0,
                "analysis_available": False,
                "message": "No recent battles to analyze",
            }

        # Calculate performance metrics
        total_battles = len(recent_battles)
        sword_wins = sum(1 for b in recent_battles if b.overall_winner == "sword")
        shield_wins = sum(1 for b in recent_battles if b.overall_winner == "shield")
        draws = sum(1 for b in recent_battles if b.overall_winner == "draw")

        avg_duration = sum(b.duration_minutes for b in recent_battles) / total_battles
        avg_attack_success = (
            sum(b.attack_success_rate for b in recent_battles) / total_battles
        )
        avg_defense_success = (
            sum(b.defense_success_rate for b in recent_battles) / total_battles
        )
        avg_sword_score = sum(b.sword_score for b in recent_battles) / total_battles
        avg_shield_score = sum(b.shield_score for b in recent_battles) / total_battles

        return {
            "battles_analyzed": total_battles,
            "analysis_available": True,
            "win_distribution": {
                "sword_wins": sword_wins,
                "shield_wins": shield_wins,
                "draws": draws,
                "sword_win_rate": sword_wins / total_battles,
                "shield_win_rate": shield_wins / total_battles,
            },
            "performance_averages": {
                "battle_duration_minutes": avg_duration,
                "attack_success_rate": avg_attack_success,
                "defense_success_rate": avg_defense_success,
                "sword_score": avg_sword_score,
                "shield_score": avg_shield_score,
            },
            "battle_balance": {
                "is_balanced": abs(sword_wins - shield_wins) <= 1,
                "dominant_side": "sword"
                if sword_wins > shield_wins
                else "shield"
                if shield_wins > sword_wins
                else "balanced",
            },
        }

    def _analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.performance_history) < 2:
            return {
                "trend_analysis_available": False,
                "message": "Insufficient data for trend analysis",
            }

        # Compare current period with previous period
        current_report = self.performance_history[-1]
        previous_report = self.performance_history[-2]

        current_perf = current_report.get("battle_performance", {}).get(
            "performance_averages", {}
        )
        previous_perf = previous_report.get("battle_performance", {}).get(
            "performance_averages", {}
        )

        if not current_perf or not previous_perf:
            return {
                "trend_analysis_available": False,
                "message": "Incomplete performance data for trend analysis",
            }

        trends = {}
        for metric in current_perf:
            if metric in previous_perf:
                current_value = current_perf[metric]
                previous_value = previous_perf[metric]
                change = current_value - previous_value
                percent_change = (
                    (change / previous_value) * 100 if previous_value != 0 else 0
                )

                trends[metric] = {
                    "current_value": current_value,
                    "previous_value": previous_value,
                    "change": change,
                    "percent_change": percent_change,
                    "trend": "improving"
                    if change > 0
                    else "declining"
                    if change < 0
                    else "stable",
                }

        return {
            "trend_analysis_available": True,
            "metric_trends": trends,
            "overall_trend": self._determine_overall_trend(trends),
        }

    def _determine_overall_trend(self, trends: dict[str, dict]) -> str:
        """Determine overall performance trend."""
        positive_trends = 0
        negative_trends = 0

        # Weight different metrics
        metric_weights = {
            "attack_success_rate": 1,
            "defense_success_rate": 1,
            "sword_score": 1,
            "shield_score": 1,
            "battle_duration_minutes": -1,  # Lower duration is better
        }

        for metric, trend_data in trends.items():
            weight = metric_weights.get(metric, 1)
            change = trend_data["change"]

            if (weight > 0 and change > 0) or (weight < 0 and change < 0):
                positive_trends += 1
            elif (weight > 0 and change < 0) or (weight < 0 and change > 0):
                negative_trends += 1

        if positive_trends > negative_trends:
            return "improving"
        elif negative_trends > positive_trends:
            return "declining"
        else:
            return "stable"

    def _generate_recommendations(
        self, current_status: dict, battle_analysis: dict, trend_analysis: dict
    ) -> list[str]:
        """Generate system improvement recommendations."""
        recommendations = []

        # System health recommendations
        health = current_status["system_health"]
        if health["health_score"] < 80:
            recommendations.append("investigate_system_health_issues")

        for issue in health["issues"]:
            if issue == "sword_agent_inactive":
                recommendations.append("restart_sword_agent")
            elif issue == "shield_agent_inactive":
                recommendations.append("restart_shield_agent")
            elif issue == "no_battle_scenarios":
                recommendations.append("reload_battle_scenarios")

        # Battle performance recommendations
        if battle_analysis.get("analysis_available"):
            perf = battle_analysis["performance_averages"]
            balance = battle_analysis["battle_balance"]

            if not balance["is_balanced"]:
                if balance["dominant_side"] == "sword":
                    recommendations.append("strengthen_shield_defenses")
                elif balance["dominant_side"] == "shield":
                    recommendations.append("enhance_sword_attacks")

            if perf.get("attack_success_rate", 0) < 0.4:
                recommendations.append("improve_sword_techniques")
            elif perf.get("attack_success_rate", 0) > 0.9:
                recommendations.append("increase_battle_difficulty")

            if perf.get("defense_success_rate", 0) < 0.4:
                recommendations.append("improve_shield_patterns")
            elif perf.get("defense_success_rate", 0) > 0.9:
                recommendations.append("enhance_attack_sophistication")

        # Trend-based recommendations
        if trend_analysis.get("trend_analysis_available"):
            overall_trend = trend_analysis["overall_trend"]

            if overall_trend == "declining":
                recommendations.append("investigate_performance_decline")
                recommendations.append("review_recent_changes")
            elif overall_trend == "stable":
                recommendations.append("introduce_new_scenarios")
                recommendations.append("increase_training_complexity")

        return recommendations

    def _log_report_summary(self, report: dict[str, Any]):
        """Log key points from the status report."""
        logger.info("📈 Battle System Status Report Summary:")

        # System health
        health = report["system_status"]["system_health"]
        logger.info(
            f"  System Health: {health['status'].upper()} ({health['health_score']}/100)"
        )

        if health["issues"]:
            logger.warning(f"  Issues: {', '.join(health['issues'])}")
        if health["warnings"]:
            logger.warning(f"  Warnings: {', '.join(health['warnings'])}")

        # Battle performance
        battle_perf = report["battle_performance"]
        if battle_perf.get("analysis_available"):
            wins = battle_perf["win_distribution"]
            logger.info(f"  Recent Battles: {battle_perf['battles_analyzed']} total")
            logger.info(
                f"  Win Distribution: Sword {wins['sword_wins']}, Shield {wins['shield_wins']}, Draws {wins['draws']}"
            )

            perf = battle_perf["performance_averages"]
            logger.info(f"  Avg Attack Success: {perf['attack_success_rate']:.1%}")
            logger.info(f"  Avg Defense Success: {perf['defense_success_rate']:.1%}")

        # Trends
        trends = report["performance_trends"]
        if trends.get("trend_analysis_available"):
            logger.info(f"  Performance Trend: {trends['overall_trend'].upper()}")

        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            logger.info(
                f"  Recommendations ({len(recommendations)}): {', '.join(recommendations[:3])}{'...' if len(recommendations) > 3 else ''}"
            )

    async def _send_king_report(self, report: dict[str, Any]):
        """Send status report to King Agent."""
        try:
            logger.info("👑 Sending status report to King Agent...")

            # Create King Agent communication
            king_communication = {
                "from_agent": "battle_monitor",
                "to_agent": "king",
                "message_type": "battle_system_status_report",
                "timestamp": datetime.now().isoformat(),
                "priority": "normal",
                "content": {
                    "summary": self._create_executive_summary(report),
                    "full_report": report,
                },
                "encrypted": False,  # King communications are unencrypted
            }

            # In a real implementation, this would send via the communication system
            logger.info("📤 Status report prepared for King Agent")
            logger.info(
                f"Executive Summary: {king_communication['content']['summary']}"
            )

        except Exception as e:
            logger.error(f"Error sending King Agent report: {e}")

    def _create_executive_summary(self, report: dict[str, Any]) -> str:
        """Create executive summary for King Agent."""
        health = report["system_status"]["system_health"]
        battle_perf = report["battle_performance"]

        summary_parts = []

        # System status
        summary_parts.append(
            f"Battle system health: {health['status']} ({health['health_score']}/100)"
        )

        # Recent activity
        if battle_perf.get("analysis_available"):
            battles = battle_perf["battles_analyzed"]
            if battles > 0:
                wins = battle_perf["win_distribution"]
                summary_parts.append(
                    f"Recent battles: {battles} completed, Sword {wins['sword_wins']} wins, Shield {wins['shield_wins']} wins"
                )
            else:
                summary_parts.append("No recent battle activity")

        # Key issues
        if health["issues"]:
            summary_parts.append(f"Critical issues: {', '.join(health['issues'])}")

        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            summary_parts.append(
                f"Top recommendation: {recommendations[0].replace('_', ' ')}"
            )

        return ". ".join(summary_parts) + "."

    def get_monitoring_statistics(self) -> dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            "monitoring_start_time": (
                datetime.now() - timedelta(seconds=len(self.agent_status_history) * 300)
            ).isoformat(),
            "status_snapshots_collected": len(self.agent_status_history),
            "reports_generated": len(self.performance_history),
            "report_interval_minutes": self.report_interval.total_seconds() / 60,
            "king_reports_enabled": self.enable_king_reports,
            "last_report_time": self.last_report_time.isoformat(),
            "next_report_due": (
                self.last_report_time + self.report_interval
            ).isoformat(),
        }


async def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(
        description="Monitor Sword/Shield Security Battle System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python battle_monitor.py                           # Monitor with default 60-minute reports
  python battle_monitor.py --report-interval 30     # Monitor with 30-minute reports
  python battle_monitor.py --king-reports           # Enable reports to King Agent
        """,
    )

    parser.add_argument(
        "--report-interval",
        type=int,
        default=60,
        help="Status report interval in minutes (default: 60)",
    )

    parser.add_argument(
        "--king-reports",
        action="store_true",
        help="Enable status reports to King Agent",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create monitor
    monitor = BattleSystemMonitor(
        report_interval_minutes=args.report_interval,
        enable_king_reports=args.king_reports,
    )

    # For monitoring, we need to connect to an existing battle orchestrator
    # In a real deployment, this would connect to the running system
    logger.info("🔍 Battle System Monitor starting...")
    logger.info(
        "Note: In production, this would connect to the running battle orchestrator"
    )

    # Create a mock orchestrator for demonstration
    mock_orchestrator = BattleOrchestrator("monitoring_orchestrator")
    await mock_orchestrator.initialize_agents()

    try:
        await monitor.start_monitoring(mock_orchestrator)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

        # Show final statistics
        stats = monitor.get_monitoring_statistics()
        logger.info("📊 Monitoring Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
