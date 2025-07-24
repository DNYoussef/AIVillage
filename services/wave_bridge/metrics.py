"""
Response Time Monitoring and Metrics for WhatsApp Wave Bridge
Target: <5 second response time with comprehensive tracking
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import logging
import wandb
import json

logger = logging.getLogger(__name__)

class ResponseMetrics:
    """
    Comprehensive response time and performance metrics tracking
    """
    
    def __init__(self):
        # Response time tracking
        self.response_times = deque(maxlen=1000)  # Keep last 1000 responses
        self.hourly_metrics = defaultdict(list)
        self.daily_metrics = defaultdict(list)
        
        # Performance targets
        self.target_response_time = 5.0  # seconds
        self.warning_threshold = 4.0    # seconds
        self.excellent_threshold = 2.0  # seconds
        
        # Language-specific metrics
        self.language_metrics = defaultdict(lambda: {
            'response_times': deque(maxlen=100),
            'success_rate': 0.0,
            'total_requests': 0,
            'avg_response_time': 0.0
        })
        
        # Subject-specific metrics
        self.subject_metrics = defaultdict(lambda: {
            'response_times': deque(maxlen=100),
            'complexity_scores': deque(maxlen=100),
            'satisfaction_scores': deque(maxlen=100)
        })
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.timeout_count = 0
        self.fallback_usage = 0
        
        # Performance alerts
        self.alert_history = deque(maxlen=50)
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes between same alert types
        
        # Initialize performance dashboard
        self.initialize_dashboard()
    
    def initialize_dashboard(self):
        """Initialize W&B dashboard configuration"""
        
        # Define dashboard metrics
        dashboard_config = {
            "charts": [
                {
                    "title": "Response Time Distribution",
                    "type": "histogram",
                    "data": "response_time"
                },
                {
                    "title": "Performance Target Achievement",
                    "type": "line",
                    "data": ["target_met_rate", "warning_rate", "excellent_rate"]
                },
                {
                    "title": "Language Performance",
                    "type": "bar",
                    "data": "language_avg_time"
                },
                {
                    "title": "Error Rates",
                    "type": "pie",
                    "data": ["success_rate", "timeout_rate", "error_rate"]
                }
            ],
            "refresh_interval": 30  # seconds
        }
        
        wandb.config.update({"dashboard_config": dashboard_config})
    
    async def update_metrics(self, metrics_data: Dict[str, Any]):
        """Update all metrics with new response data"""
        
        try:
            response_time = metrics_data.get('response_time', 0.0)
            language = metrics_data.get('language', 'en')
            session_id = metrics_data.get('session_id', '')
            is_fallback = metrics_data.get('is_fallback', False)
            
            # Update overall response times
            self.response_times.append(response_time)
            
            # Update hourly metrics
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            self.hourly_metrics[current_hour].append(response_time)
            
            # Update daily metrics
            current_day = datetime.now().strftime('%Y-%m-%d')
            self.daily_metrics[current_day].append(response_time)
            
            # Update language-specific metrics
            lang_metrics = self.language_metrics[language]
            lang_metrics['response_times'].append(response_time)
            lang_metrics['total_requests'] += 1
            
            if lang_metrics['response_times']:
                lang_metrics['avg_response_time'] = statistics.mean(lang_metrics['response_times'])
            
            # Track fallback usage
            if is_fallback:
                self.fallback_usage += 1
            
            # Check for performance alerts
            await self.check_performance_alerts(response_time, language, session_id)
            
            # Log comprehensive metrics to W&B
            await self.log_to_wandb(metrics_data, response_time)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def check_performance_alerts(self, response_time: float, language: str, session_id: str):
        """Check for performance issues and generate alerts"""
        
        current_time = time.time()
        
        # Check response time alerts
        if response_time > self.target_response_time:
            alert_type = "response_time_exceeded"
            if self.should_send_alert(alert_type, current_time):
                await self.send_alert({
                    "type": alert_type,
                    "severity": "high" if response_time > 8.0 else "medium",
                    "response_time": response_time,
                    "target": self.target_response_time,
                    "language": language,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check recent performance trends
        if len(self.response_times) >= 10:
            recent_avg = statistics.mean(list(self.response_times)[-10:])
            if recent_avg > self.warning_threshold:
                alert_type = "performance_degradation"
                if self.should_send_alert(alert_type, current_time):
                    await self.send_alert({
                        "type": alert_type,
                        "severity": "medium",
                        "recent_avg": recent_avg,
                        "threshold": self.warning_threshold,
                        "sample_size": 10,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Check error rate alerts
        total_requests = len(self.response_times)
        if total_requests >= 50:
            error_rate = sum(self.error_counts.values()) / total_requests
            if error_rate > 0.1:  # 10% error rate
                alert_type = "high_error_rate"
                if self.should_send_alert(alert_type, current_time):
                    await self.send_alert({
                        "type": alert_type,
                        "severity": "high",
                        "error_rate": error_rate,
                        "total_requests": total_requests,
                        "errors": dict(self.error_counts),
                        "timestamp": datetime.now().isoformat()
                    })
    
    def should_send_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if alert should be sent (respects cooldown)"""
        
        last_sent = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_sent) > self.alert_cooldown
    
    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send performance alert"""
        
        try:
            # Log alert to W&B
            wandb.log({"performance_alert": alert_data})
            
            # Add to alert history
            self.alert_history.append(alert_data)
            
            # Update last alert time
            self.last_alert_time[alert_data["type"]] = time.time()
            
            logger.warning(f"Performance alert: {alert_data['type']} - {alert_data}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def log_to_wandb(self, metrics_data: Dict[str, Any], response_time: float):
        """Log comprehensive metrics to W&B"""
        
        try:
            # Calculate performance indicators
            performance_data = {
                "response_time": response_time,
                "target_met": response_time < self.target_response_time,
                "performance_level": self.get_performance_level(response_time),
                "language": metrics_data.get('language', 'en'),
                "session_id": metrics_data.get('session_id', ''),
                "is_fallback": metrics_data.get('is_fallback', False)
            }
            
            # Add aggregate metrics if we have enough data
            if len(self.response_times) >= 10:
                recent_times = list(self.response_times)[-10:]
                performance_data.update({
                    "avg_response_time_10": statistics.mean(recent_times),
                    "median_response_time_10": statistics.median(recent_times),
                    "p95_response_time_10": self.percentile(recent_times, 95),
                    "target_achievement_rate_10": sum(1 for t in recent_times if t < self.target_response_time) / len(recent_times)
                })
            
            # Add hourly performance if available
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            if current_hour in self.hourly_metrics and len(self.hourly_metrics[current_hour]) >= 5:
                hourly_times = self.hourly_metrics[current_hour]
                performance_data.update({
                    "hourly_avg_response_time": statistics.mean(hourly_times),
                    "hourly_target_achievement": sum(1 for t in hourly_times if t < self.target_response_time) / len(hourly_times),
                    "hourly_request_count": len(hourly_times)
                })
            
            # Log to W&B
            wandb.log(performance_data)
            
        except Exception as e:
            logger.error(f"Error logging to W&B: {e}")
    
    def get_performance_level(self, response_time: float) -> str:
        """Categorize performance level based on response time"""
        
        if response_time <= self.excellent_threshold:
            return "excellent"
        elif response_time <= self.warning_threshold:
            return "good"
        elif response_time <= self.target_response_time:
            return "acceptable"
        else:
            return "poor"
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of response times"""
        
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        try:
            if not self.response_times:
                return {
                    "status": "no_data",
                    "message": "No response time data available yet"
                }
            
            recent_times = list(self.response_times)
            
            # Overall performance metrics
            summary = {
                "overall_performance": {
                    "total_requests": len(recent_times),
                    "avg_response_time": statistics.mean(recent_times),
                    "median_response_time": statistics.median(recent_times),
                    "min_response_time": min(recent_times),
                    "max_response_time": max(recent_times),
                    "p95_response_time": self.percentile(recent_times, 95),
                    "p99_response_time": self.percentile(recent_times, 99),
                    "target_achievement_rate": sum(1 for t in recent_times if t < self.target_response_time) / len(recent_times),
                    "excellent_rate": sum(1 for t in recent_times if t <= self.excellent_threshold) / len(recent_times),
                    "warning_rate": sum(1 for t in recent_times if t > self.warning_threshold) / len(recent_times)
                },
                
                # Performance targets
                "targets": {
                    "target_response_time": self.target_response_time,
                    "warning_threshold": self.warning_threshold,
                    "excellent_threshold": self.excellent_threshold
                },
                
                # Language breakdown
                "language_performance": {},
                
                # Recent hourly performance
                "hourly_performance": {},
                
                # Error summary
                "error_summary": {
                    "total_errors": sum(self.error_counts.values()),
                    "error_breakdown": dict(self.error_counts),
                    "timeout_count": self.timeout_count,
                    "fallback_usage": self.fallback_usage,
                    "error_rate": sum(self.error_counts.values()) / len(recent_times) if recent_times else 0
                },
                
                # Alert summary
                "alerts": {
                    "recent_alerts": len([a for a in self.alert_history if 
                                         datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]),
                    "total_alert_types": len(set(a['type'] for a in self.alert_history))
                },
                
                "last_updated": datetime.now().isoformat()
            }
            
            # Add language-specific metrics
            for language, metrics in self.language_metrics.items():
                if metrics['response_times']:
                    summary["language_performance"][language] = {
                        "avg_response_time": statistics.mean(metrics['response_times']),
                        "total_requests": metrics['total_requests'],
                        "target_achievement_rate": sum(1 for t in metrics['response_times'] if t < self.target_response_time) / len(metrics['response_times'])
                    }
            
            # Add recent hourly performance
            current_time = datetime.now()
            for i in range(6):  # Last 6 hours
                hour_key = (current_time - timedelta(hours=i)).strftime('%Y-%m-%d-%H')
                if hour_key in self.hourly_metrics:
                    hourly_data = self.hourly_metrics[hour_key]
                    summary["hourly_performance"][hour_key] = {
                        "request_count": len(hourly_data),
                        "avg_response_time": statistics.mean(hourly_data),
                        "target_achievement_rate": sum(1 for t in hourly_data if t < self.target_response_time) / len(hourly_data)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def record_error(self, error_type: str, session_id: str = None):
        """Record an error occurrence"""
        
        self.error_counts[error_type] += 1
        
        # Log error to W&B
        wandb.log({
            "error_occurrence": {
                "error_type": error_type,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "total_count": self.error_counts[error_type]
            }
        })
    
    def record_timeout(self, session_id: str = None):
        """Record a timeout occurrence"""
        
        self.timeout_count += 1
        self.record_error("timeout", session_id)
    
    async def generate_performance_report(self) -> str:
        """Generate human-readable performance report"""
        
        summary = await self.get_summary()
        
        if "error" in summary:
            return f"Error generating report: {summary['error']}"
        
        if summary.get("status") == "no_data":
            return "No performance data available yet."
        
        overall = summary["overall_performance"]
        
        report = f"""
📊 **WhatsApp Wave Bridge Performance Report**
Generated: {summary['last_updated']}

🎯 **Overall Performance**
• Total Requests: {overall['total_requests']:,}
• Average Response Time: {overall['avg_response_time']:.2f}s
• Target Achievement: {overall['target_achievement_rate']:.1%}
• Excellent Performance: {overall['excellent_rate']:.1%}

⚡ **Response Time Breakdown**
• Median: {overall['median_response_time']:.2f}s
• 95th Percentile: {overall['p95_response_time']:.2f}s
• 99th Percentile: {overall['p99_response_time']:.2f}s
• Range: {overall['min_response_time']:.2f}s - {overall['max_response_time']:.2f}s

🌍 **Language Performance**
"""
        
        for lang, perf in summary["language_performance"].items():
            lang_name = {'en': 'English', 'es': 'Spanish', 'hi': 'Hindi', 'sw': 'Swahili', 'ar': 'Arabic', 'pt': 'Portuguese', 'fr': 'French'}.get(lang, lang)
            report += f"• {lang_name}: {perf['avg_response_time']:.2f}s avg ({perf['total_requests']} requests)\n"
        
        error_summary = summary["error_summary"]
        report += f"""
🚨 **Error Summary**
• Total Errors: {error_summary['total_errors']}
• Error Rate: {error_summary['error_rate']:.1%}
• Timeouts: {error_summary['timeout_count']}
• Fallback Usage: {error_summary['fallback_usage']}

📈 **Performance Status**
"""
        
        if overall['target_achievement_rate'] >= 0.95:
            report += "✅ **EXCELLENT** - Consistently meeting response time targets"
        elif overall['target_achievement_rate'] >= 0.90:
            report += "🟡 **GOOD** - Mostly meeting response time targets"
        elif overall['target_achievement_rate'] >= 0.80:
            report += "🟠 **WARNING** - Frequently missing response time targets"
        else:
            report += "🔴 **CRITICAL** - Consistently missing response time targets"
        
        return report

# Global metrics instance
metrics = ResponseMetrics()