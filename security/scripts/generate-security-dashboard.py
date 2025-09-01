#!/usr/bin/env python3
"""
Security Dashboard Generator
Creates a visual security dashboard with charts and metrics.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import html


def load_security_results() -> Dict[str, Any]:
    """Load aggregated security results."""
    results_file = "security/reports/aggregated-security-results.json"
    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load security results: {e}")
        return {"summary": {}, "detailed_results": []}


def generate_severity_chart_data(severity_counts: Dict[str, int]) -> str:
    """Generate data for severity distribution chart."""
    labels = list(severity_counts.keys())
    values = list(severity_counts.values())

    chart_data = {
        "labels": labels,
        "datasets": [
            {
                "data": values,
                "backgroundColor": [
                    "#ff4444",  # Critical - Red
                    "#ff8800",  # High - Orange
                    "#ffaa00",  # Medium - Yellow
                    "#00aa00",  # Low - Green
                ],
                "borderWidth": 1,
            }
        ],
    }

    return json.dumps(chart_data)


def generate_tools_comparison_data(tools_summary: Dict[str, Any]) -> str:
    """Generate data for tools comparison chart."""
    tool_names = list(tools_summary.keys())
    issue_counts = [tools_summary[tool]["total_issues"] for tool in tool_names]

    chart_data = {
        "labels": tool_names,
        "datasets": [
            {
                "label": "Issues Found",
                "data": issue_counts,
                "backgroundColor": "#4CAF50",
                "borderColor": "#45a049",
                "borderWidth": 1,
            }
        ],
    }

    return json.dumps(chart_data)


def generate_timeline_data(detailed_results: List[Dict[str, Any]]) -> str:
    """Generate timeline data for security issues."""
    # For now, create a simple timeline based on current data
    # In a real implementation, this would track issues over time
    timeline_data = {"labels": [datetime.now().strftime("%Y-%m-%d")], "datasets": []}

    colors = {"critical": "#ff4444", "high": "#ff8800", "medium": "#ffaa00", "low": "#00aa00"}

    for severity in ["critical", "high", "medium", "low"]:
        total_issues = 0
        for result in detailed_results:
            severity_breakdown = result.get("severity_breakdown", {})
            for key, count in severity_breakdown.items():
                if key.lower() in [severity, severity.upper(), severity.capitalize()]:
                    total_issues += count

        timeline_data["datasets"].append(
            {
                "label": severity.capitalize(),
                "data": [total_issues],
                "borderColor": colors[severity],
                "backgroundColor": colors[severity] + "30",  # Add transparency
                "fill": False,
            }
        )

    return json.dumps(timeline_data)


def generate_dashboard_html(results: Dict[str, Any]) -> str:
    """Generate HTML dashboard."""
    summary = results.get("summary", {})
    detailed_results = results.get("detailed_results", [])

    severity_counts = summary.get("overall_severity_counts", {})
    tools_summary = summary.get("tools_summary", {})
    critical_issues = summary.get("critical_issues", [])
    recommendations = summary.get("recommendations", [])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Dashboard - AIVillage</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .critical {{ color: #e74c3c; }}
        .high {{ color: #f39c12; }}
        .medium {{ color: #f1c40f; }}
        .low {{ color: #27ae60; }}
        .total {{ color: #3498db; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .chart-card h3 {{
            margin-bottom: 20px;
            color: #333;
            font-size: 1.3em;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .issues-section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .issues-section h3 {{
            margin-bottom: 20px;
            color: #333;
            font-size: 1.3em;
        }}
        
        .issue-item {{
            padding: 15px;
            border-left: 4px solid #e74c3c;
            margin-bottom: 10px;
            background: #fdf2f2;
            border-radius: 0 5px 5px 0;
        }}
        
        .issue-item.high {{
            border-left-color: #f39c12;
            background: #fdf6e3;
        }}
        
        .issue-tool {{
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .issue-description {{
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        
        .issue-file {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #666;
        }}
        
        .recommendations {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .recommendations h3 {{
            margin-bottom: 20px;
            color: #333;
            font-size: 1.3em;
        }}
        
        .recommendation-item {{
            padding: 10px 15px;
            background: #e8f5e8;
            border-left: 4px solid #27ae60;
            margin-bottom: 10px;
            border-radius: 0 5px 5px 0;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .chart-container {{
                height: 250px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Security Dashboard</h1>
            <div class="subtitle">
                Generated on {summary.get('timestamp', datetime.now().isoformat())}
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value critical">{severity_counts.get('critical', 0)}</div>
                <div class="metric-label">Critical Issues</div>
            </div>
            <div class="metric-card">
                <div class="metric-value high">{severity_counts.get('high', 0)}</div>
                <div class="metric-label">High Severity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value medium">{severity_counts.get('medium', 0)}</div>
                <div class="metric-label">Medium Severity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value low">{severity_counts.get('low', 0)}</div>
                <div class="metric-label">Low Severity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value total">{sum(severity_counts.values())}</div>
                <div class="metric-label">Total Issues</div>
            </div>
            <div class="metric-card">
                <div class="metric-value total">{len(tools_summary)}</div>
                <div class="metric-label">Tools Used</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h3>📊 Severity Distribution</h3>
                <div class="chart-container">
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>🔧 Tool Comparison</h3>
                <div class="chart-container">
                    <canvas id="toolsChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>📈 Security Timeline</h3>
                <div class="chart-container">
                    <canvas id="timelineChart"></canvas>
                </div>
            </div>
        </div>
"""

    # Add critical issues section if any exist
    if critical_issues:
        html_content += f"""
        <div class="issues-section">
            <h3>🚨 Critical & High Severity Issues</h3>
            {"".join([f'''
            <div class="issue-item {'critical' if issue.get('severity', '').lower() in ['critical', 'error'] else 'high'}">
                <div class="issue-tool">{html.escape(str(issue.get('tool', 'Unknown')))}</div>
                <div class="issue-description">{html.escape(str(issue.get('issue_text', issue.get('message', issue.get('advisory', 'No description')))))}</div>
                <div class="issue-file">{html.escape(str(issue.get('filename', issue.get('path', issue.get('package_name', 'Unknown location')))))}</div>
            </div>
            ''' for issue in critical_issues[:10]])}
            {f'<p>... and {len(critical_issues) - 10} more critical/high severity issues</p>' if len(critical_issues) > 10 else ''}
        </div>
        """

    # Add recommendations section
    if recommendations:
        html_content += f"""
        <div class="recommendations">
            <h3>💡 Security Recommendations</h3>
            {"".join([f'<div class="recommendation-item">{html.escape(str(rec))}</div>' for rec in recommendations])}
        </div>
        """

    html_content += f"""
        <div class="footer">
            Dashboard generated by AIVillage Security System
        </div>
    </div>
    
    <script>
        // Severity Distribution Chart
        const severityCtx = document.getElementById('severityChart').getContext('2d');
        new Chart(severityCtx, {{
            type: 'doughnut',
            data: {generate_severity_chart_data(severity_counts)},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Tools Comparison Chart
        const toolsCtx = document.getElementById('toolsChart').getContext('2d');
        new Chart(toolsCtx, {{
            type: 'bar',
            data: {generate_tools_comparison_data(tools_summary)},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Timeline Chart
        const timelineCtx = document.getElementById('timelineChart').getContext('2d');
        new Chart(timelineCtx, {{
            type: 'line',
            data: {generate_timeline_data(detailed_results)},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html_content


def main():
    """Main dashboard generation function."""
    print("Generating security dashboard...")

    # Load security results
    results = load_security_results()

    # Create dashboard directory
    os.makedirs("security/dashboard", exist_ok=True)

    # Generate HTML dashboard
    dashboard_html = generate_dashboard_html(results)

    # Save dashboard
    with open("security/dashboard/index.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print("SUCCESS: Security dashboard generated: security/dashboard/index.html")

    # Generate additional JSON report for programmatic access
    dashboard_data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": results.get("summary", {}),
        "charts_data": {
            "severity_distribution": generate_severity_chart_data(
                results.get("summary", {}).get("overall_severity_counts", {})
            ),
            "tools_comparison": generate_tools_comparison_data(results.get("summary", {}).get("tools_summary", {})),
            "timeline": generate_timeline_data(results.get("detailed_results", [])),
        },
    }

    with open("security/dashboard/dashboard-data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print("SUCCESS: Dashboard data saved: security/dashboard/dashboard-data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
