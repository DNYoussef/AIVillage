# Dashboard Configuration Guide

## Introduction

This guide covers the configuration and customization of AIVillage monitoring dashboards. The platform provides multiple specialized dashboards for different operational needs, from real-time system monitoring to Agent Forge pipeline tracking and security oversight.

## Dashboard Architecture Overview

### Dashboard Types and Purpose

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIVillage Dashboard Ecosystem                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ System Health   │    │ Security        │    │ Agent Forge     │  │
│  │ Dashboard       │    │ Dashboard       │    │ Dashboard       │  │
│  │                 │    │                 │    │                 │  │
│  │ • Resource      │    │ • Threats       │    │ • Pipeline      │  │
│  │   Monitoring    │    │ • Incidents     │    │   Status        │  │
│  │ • Performance   │    │ • Alerts        │    │ • Model         │  │
│  │ • Health Status │    │ • Analytics     │    │   Evolution     │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           │                       │                       │          │
│           └───────────────────────┼───────────────────────┘          │
│                                   │                                  │
│  ┌─────────────────────────────────┼─────────────────────────────────┐  │
│  │                    Unified Data Layer                            │  │
│  │                                                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  │ Metrics     │  │ Logs        │  │ Traces      │  │ Events    │  │
│  │  │ Collection  │  │ Aggregation │  │ Correlation │  │ Streaming │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend Technologies**:
- **Streamlit**: Real-time web-based dashboards with Python integration
- **Plotly**: Interactive charts and visualizations
- **Pandas**: Data processing and analysis
- **HTML/CSS**: Custom styling and layout

**Backend Integration**:
- **SQLite**: Metrics and observability data storage
- **AsyncIO**: Non-blocking data fetching and processing
- **WebSocket**: Real-time data streaming for live updates
- **JSON API**: RESTful data access for external integrations

## Agent Forge Dashboard

### Implementation Overview

**Location**: `packages/monitoring/dashboard.py:26-433`

The Agent Forge Dashboard provides comprehensive real-time monitoring of the Agent Forge pipeline execution, model evolution progress, and system performance.

```python
class AgentForgeDashboard:
    """Real-time monitoring dashboard for Agent Forge pipeline."""

    def __init__(self):
        self.data_dir = Path("./forge_output_enhanced")
        self.checkpoint_dir = Path("./forge_checkpoints_enhanced")
        self.logs_dir = Path("./logs")

        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.checkpoint_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for dashboard display."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

            # GPU metrics (if available)
            gpu_metrics = self.get_gpu_metrics()

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {"percent": cpu_percent, "count": cpu_count},
                "memory": {
                    "used_gb": memory_used_gb,
                    "total_gb": memory_total_gb,
                    "percent": memory.percent
                },
                "gpu": gpu_metrics
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
```

### Dashboard Configuration

#### Streamlit Configuration

```python
# streamlit_config.toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false

[logger]
level = "info"
enableCORS = false
```

#### Dashboard Layout Configuration

```python
def configure_dashboard_layout():
    """Configure Streamlit dashboard layout and styling."""

    st.set_page_config(
        page_title="Agent Forge Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.aivillage.ai/monitoring',
            'Report a bug': 'https://github.com/aivillage/issues',
            'About': 'AIVillage Agent Forge Monitoring Dashboard v2.0'
        }
    )

    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-critical { background-color: #dc3545; }

    .phase-progress {
        background: linear-gradient(90deg, #28a745 0%, #17a2b8 100%);
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
    }

    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
```

### Real-Time Dashboard Implementation

```python
def main():
    """Main dashboard function with real-time updates."""

    configure_dashboard_layout()

    # Dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🤖 Agent Forge Real-Time Dashboard</h1>
        <p>Monitor pipeline execution, model evolution, and system performance</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize dashboard
    dashboard = AgentForgeDashboard()

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

    # Metrics selection
    show_system_metrics = st.sidebar.checkbox("System Metrics", value=True)
    show_pipeline_status = st.sidebar.checkbox("Pipeline Status", value=True)
    show_performance_charts = st.sidebar.checkbox("Performance Charts", value=True)
    show_wandb_integration = st.sidebar.checkbox("W&B Integration", value=True)

    # Main dashboard loop
    placeholder = st.empty()

    while auto_refresh:
        with placeholder.container():

            # System Metrics Section
            if show_system_metrics:
                render_system_metrics_section(dashboard)

            # Pipeline Status Section
            if show_pipeline_status:
                render_pipeline_status_section(dashboard)

            # Performance Charts Section
            if show_performance_charts:
                render_performance_charts_section(dashboard)

            # W&B Integration Section
            if show_wandb_integration:
                render_wandb_integration_section(dashboard)

            # Footer with refresh info
            render_dashboard_footer(auto_refresh, refresh_interval)

        if auto_refresh:
            time.sleep(refresh_interval)
        else:
            break

def render_system_metrics_section(dashboard: AgentForgeDashboard):
    """Render system metrics section with real-time data."""

    st.header("📊 System Metrics")

    system_metrics = dashboard.get_system_metrics()

    if "error" not in system_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cpu_percent = system_metrics['cpu']['percent']
            cpu_status = get_status_indicator(cpu_percent, 80, 90)

            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {cpu_status}"></span>
                <strong>CPU Usage</strong><br>
                <span style="font-size: 24px">{cpu_percent:.1f}%</span><br>
                <small>{system_metrics['cpu']['count']} cores</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            memory_percent = system_metrics['memory']['percent']
            memory_status = get_status_indicator(memory_percent, 80, 90)

            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {memory_status}"></span>
                <strong>Memory Usage</strong><br>
                <span style="font-size: 24px">{memory_percent:.1f}%</span><br>
                <small>{system_metrics['memory']['used_gb']:.1f} / {system_metrics['memory']['total_gb']:.1f} GB</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Disk usage (would need implementation)
            disk_percent = 45.2  # Placeholder
            disk_status = get_status_indicator(disk_percent, 80, 90)

            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {disk_status}"></span>
                <strong>Disk Usage</strong><br>
                <span style="font-size: 24px">{disk_percent:.1f}%</span><br>
                <small>512 GB SSD</small>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            gpu_metrics = system_metrics.get('gpu', {})
            if gpu_metrics.get('available'):
                gpu_percent = gpu_metrics['memory']['percent']
                gpu_status = get_status_indicator(gpu_percent, 80, 90)

                st.markdown(f"""
                <div class="metric-card">
                    <span class="status-indicator {gpu_status}"></span>
                    <strong>GPU Memory</strong><br>
                    <span style="font-size: 24px">{gpu_percent:.1f}%</span><br>
                    <small>{gpu_metrics['device_name'][:20]}...</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <span class="status-indicator status-warning"></span>
                    <strong>GPU</strong><br>
                    <span style="font-size: 16px">Not Available</span><br>
                    <small>CPU-only mode</small>
                </div>
                """, unsafe_allow_html=True)

def get_status_indicator(value: float, warning_threshold: float, critical_threshold: float) -> str:
    """Get CSS class for status indicator based on thresholds."""
    if value >= critical_threshold:
        return "status-critical"
    elif value >= warning_threshold:
        return "status-warning"
    else:
        return "status-healthy"
```

### Pipeline Status Visualization

```python
def render_pipeline_status_section(dashboard: AgentForgeDashboard):
    """Render pipeline status with phase progress tracking."""

    st.header("🔄 Pipeline Status")

    pipeline_status = dashboard.get_pipeline_status()

    # Pipeline summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Runs", pipeline_status["active_runs"])
    with col2:
        st.metric("Completed Runs", pipeline_status["completed_runs"])
    with col3:
        st.metric("Failed Runs", pipeline_status["failed_runs"])
    with col4:
        total_runs = (pipeline_status["active_runs"] +
                     pipeline_status["completed_runs"] +
                     pipeline_status["failed_runs"])
        st.metric("Total Runs", total_runs)

    # Current phase progress
    if pipeline_status["current_phase"]:
        st.subheader(f"Current Phase: {pipeline_status['current_phase']}")

        # Progress calculation
        total_phases = 7  # EvoMerge, Quiet-STaR, BitNet, Training, Tool/Persona, ADAS, Compression
        completed_phases = len(pipeline_status["phases_completed"])
        progress_percent = (completed_phases / total_phases) * 100

        # Progress bar
        st.progress(progress_percent / 100)
        st.write(f"Completed phases: {', '.join(pipeline_status['phases_completed'])}")

        # Phase timeline visualization
        render_phase_timeline(pipeline_status["phases_completed"], pipeline_status["current_phase"])

    # Latest run information
    if pipeline_status["latest_run"]:
        st.subheader("Latest Run Details")
        latest = pipeline_status["latest_run"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Run ID:** {latest['run_id']}")
            st.write(f"**Current Phase:** {latest['phase']}")
        with col_b:
            st.write(f"**Timestamp:** {latest['timestamp']}")
            st.write(f"**Checkpoint:** {Path(latest['checkpoint_file']).name}")

def render_phase_timeline(completed_phases: List[str], current_phase: str):
    """Render interactive phase timeline."""

    phases = [
        {"name": "EvoMerge", "description": "Model merging and optimization"},
        {"name": "Quiet-STaR", "description": "Prompt baking and thought integration"},
        {"name": "BitNet", "description": "Initial quantization and compression"},
        {"name": "Training", "description": "Grokfast training with edge-of-chaos"},
        {"name": "Tool/Persona", "description": "Tool integration and persona baking"},
        {"name": "ADAS", "description": "Architecture discovery and search"},
        {"name": "Compression", "description": "Final compression pipeline"},
    ]

    # Create timeline data
    timeline_data = []
    for i, phase in enumerate(phases):
        if phase["name"] in completed_phases:
            status = "completed"
            color = "#28a745"
        elif phase["name"] == current_phase:
            status = "active"
            color = "#ffc107"
        else:
            status = "pending"
            color = "#6c757d"

        timeline_data.append({
            "phase": phase["name"],
            "description": phase["description"],
            "status": status,
            "position": i + 1,
            "color": color
        })

    # Create timeline chart
    fig = go.Figure()

    for item in timeline_data:
        fig.add_trace(go.Scatter(
            x=[item["position"]],
            y=[1],
            mode='markers+text',
            marker=dict(
                size=30,
                color=item["color"],
                symbol='circle'
            ),
            text=item["phase"],
            textposition="bottom center",
            name=item["status"],
            hovertemplate=f"<b>{item['phase']}</b><br>" +
                         f"{item['description']}<br>" +
                         f"Status: {item['status']}<extra></extra>"
        ))

    # Add connecting line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(phases) + 1)),
        y=[1] * len(phases),
        mode='lines',
        line=dict(color='#dee2e6', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Agent Forge Pipeline Phase Progress",
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0.5, len(phases) + 0.5]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0.5, 1.5]
        ),
        height=200,
        margin=dict(l=0, r=0, t=40, b=60),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)
```

### Performance Charts and Analytics

```python
def render_performance_charts_section(dashboard: AgentForgeDashboard):
    """Render performance analytics charts."""

    st.header("📈 Performance Analytics")

    # Resource usage trends
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resource Usage Trends")
        render_resource_trends_chart()

    with col2:
        st.subheader("Pipeline Performance")
        render_pipeline_performance_chart()

    # Model metrics
    st.subheader("Model Training Metrics")
    render_model_metrics_chart()

def render_resource_trends_chart():
    """Generate resource usage trends chart."""

    # Generate sample time series data (replace with real data)
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        end=datetime.now(),
        periods=60
    )

    # Simulate realistic resource patterns
    cpu_data = [50 + 20 * np.sin(i / 10) + np.random.normal(0, 5) for i in range(60)]
    memory_data = [60 + 15 * np.sin(i / 8) + np.random.normal(0, 3) for i in range(60)]
    gpu_data = [70 + 25 * np.sin(i / 12) + np.random.normal(0, 8) for i in range(60)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_data,
        mode='lines',
        name='CPU %',
        line=dict(color='#007bff', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_data,
        mode='lines',
        name='Memory %',
        line=dict(color='#28a745', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=gpu_data,
        mode='lines',
        name='GPU %',
        line=dict(color='#ffc107', width=2)
    ))

    fig.update_layout(
        title="System Resource Usage (Last Hour)",
        xaxis_title="Time",
        yaxis_title="Usage %",
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

def render_pipeline_performance_chart():
    """Generate pipeline performance metrics chart."""

    # Sample pipeline performance data
    phases = ["EvoMerge", "Quiet-STaR", "BitNet", "Training", "Tool/Persona", "ADAS", "Compression"]
    durations = [45, 120, 30, 180, 90, 60, 75]  # minutes
    success_rates = [95, 92, 98, 88, 94, 85, 96]  # percentage

    fig = go.Figure()

    # Duration bars
    fig.add_trace(go.Bar(
        x=phases,
        y=durations,
        name='Duration (min)',
        marker_color='#007bff',
        yaxis='y1'
    ))

    # Success rate line
    fig.add_trace(go.Scatter(
        x=phases,
        y=success_rates,
        mode='lines+markers',
        name='Success Rate (%)',
        line=dict(color='#28a745', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))

    fig.update_layout(
        title="Phase Performance Metrics",
        xaxis_title="Pipeline Phase",
        yaxis=dict(
            title="Duration (minutes)",
            side="left"
        ),
        yaxis2=dict(
            title="Success Rate (%)",
            side="right",
            overlaying="y",
            range=[80, 100]
        ),
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

def render_model_metrics_chart():
    """Generate model training metrics visualization."""

    # Sample training metrics
    epochs = list(range(1, 51))
    loss_values = [2.5 * np.exp(-i/20) + 0.1 + np.random.normal(0, 0.05) for i in epochs]
    accuracy_values = [70 + 25 * (1 - np.exp(-i/15)) + np.random.normal(0, 1) for i in epochs]

    col1, col2 = st.columns(2)

    with col1:
        # Loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs,
            y=loss_values,
            mode='lines',
            name='Training Loss',
            line=dict(color='#dc3545', width=2)
        ))

        fig_loss.update_layout(
            title="Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=250
        )

        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        # Accuracy curve
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs,
            y=accuracy_values,
            mode='lines',
            name='Accuracy',
            line=dict(color='#28a745', width=2)
        ))

        fig_acc.update_layout(
            title="Model Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            height=250
        )

        st.plotly_chart(fig_acc, use_container_width=True)
```

## System Health Dashboard

### Component Health Monitoring

**Location**: `packages/monitoring/system_health_dashboard.py:453-596`

```python
class SystemHealthDashboard:
    """System health monitoring with component analysis."""

    def __init__(self, project_root: Path = None):
        self.health_checker = ComponentHealthChecker(project_root)
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.device_profiler = DeviceProfiler()

    async def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete system health dashboard data."""

        # Scan all components
        component_results = await self.health_checker.scan_all_components()

        # Collect system metrics
        system_metrics = self.collect_system_metrics()

        # Calculate overall health
        overall_health = self.health_checker.calculate_overall_system_health(component_results)

        # Generate health report
        health_report = self.health_checker.generate_health_report(component_results, overall_health)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": overall_health,
            "component_results": component_results,
            "system_metrics": system_metrics,
            "health_report": health_report,
            "sprint_success": overall_health["completion_percentage"] > 60.0
        }

    def print_dashboard_summary(self, dashboard_data: Dict[str, Any]):
        """Print comprehensive dashboard summary to console."""

        overall = dashboard_data["overall_health"]

        print("\n" + "=" * 60)
        print("🏥 AIVILLAGE SYSTEM HEALTH DASHBOARD")
        print("=" * 60)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 System Completion: {overall['completion_percentage']:.1f}%")
        print(f"🏥 Health Status: {overall['health_status'].upper()}")
        print(f"✅ Healthy Components: {overall['healthy_components']}/{overall['total_components']}")
        print(f"⚠️  Partial Components: {overall['partial_components']}/{overall['total_components']}")
        print(f"❌ Unhealthy Components: {overall['unhealthy_components']}/{overall['total_components']}")

        # Sprint success indicator
        if dashboard_data["sprint_success"]:
            print("\n🎉 SPRINT SUCCESS: Target >60% completion achieved!")
        else:
            print(f"\n⚠️  Sprint target not met. Need >60%, current: {overall['completion_percentage']:.1f}%")

        print("=" * 60)
```

### Health Dashboard Configuration

```python
# health_dashboard_config.yaml
dashboard:
  title: "AIVillage System Health Monitor"
  refresh_interval: 30  # seconds
  auto_refresh: true

  sections:
    - name: "Executive Summary"
      components:
        - overall_health_score
        - completion_percentage
        - component_status_breakdown
        - sprint_progress

    - name: "Component Health"
      components:
        - component_health_grid
        - implementation_scores
        - functionality_analysis
        - stub_detection_results

    - name: "System Resources"
      components:
        - cpu_memory_usage
        - disk_space_analysis
        - network_performance
        - process_monitoring

    - name: "Performance Trends"
      components:
        - health_score_trends
        - completion_progress
        - error_rate_analysis
        - performance_regression

  thresholds:
    healthy_score: 0.7      # 70% implementation score
    partial_score: 0.3      # 30% implementation score
    critical_health: 0.1    # 10% implementation score

  alerts:
    enable_alerts: true
    health_degradation_threshold: 0.1  # 10% drop in health score
    component_failure_threshold: 3     # Number of failed components
    performance_degradation: 0.2       # 20% performance drop

# Component monitoring configuration
components:
  scan_directories:
    - "src/core"
    - "src/production"
    - "src/agent_forge"
    - "src/communications"
    - "src/ingestion"
    - "src/mcp_servers"

  health_checks:
    - name: "syntax_validation"
      description: "Python syntax validation"
      weight: 0.2

    - name: "import_resolution"
      description: "Import statement validation"
      weight: 0.15

    - name: "function_implementation"
      description: "Function implementation completeness"
      weight: 0.25

    - name: "error_handling"
      description: "Exception handling presence"
      weight: 0.2

    - name: "documentation"
      description: "Docstring and comment coverage"
      weight: 0.1

    - name: "testing_coverage"
      description: "Test coverage analysis"
      weight: 0.1

  stub_detection:
    patterns:
      - "pass"
      - "NotImplementedError"
      - "TODO"
      - "STUB"
      - "raise NotImplementedError"
      - "return None"
      - "return []"
      - "return {}"

    ast_analysis:
      detect_empty_functions: true
      detect_docstring_only: true
      detect_placeholder_returns: true
```

## Security Dashboard

### Security Monitoring Interface

```python
class SecurityDashboard:
    """Security monitoring dashboard with threat analysis."""

    def __init__(self, security_monitor: SecurityMonitor):
        self.security_monitor = security_monitor
        self.threat_analyzer = ThreatAnalyzer()

    def render_security_overview(self):
        """Render security overview dashboard."""

        st.header("🔒 Security Monitoring Dashboard")

        # Security status metrics
        security_status = self.security_monitor.get_security_status()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Security Status",
                security_status["status"].title(),
                delta=None
            )

        with col2:
            st.metric(
                "Recent Alerts",
                security_status["recent_alerts_count"],
                delta=f"+{security_status['recent_alerts_count'] - 10}"  # Sample delta
            )

        with col3:
            st.metric(
                "Critical Alerts",
                security_status["critical_alerts"],
                delta=None
            )

        with col4:
            st.metric(
                "Monitoring Active",
                "✅" if security_status["monitoring_active"] else "❌",
                delta=None
            )

        # Threat landscape visualization
        self.render_threat_landscape()

        # Recent security events
        self.render_recent_security_events()

        # Attack pattern analysis
        self.render_attack_pattern_analysis()

    def render_threat_landscape(self):
        """Render threat landscape heatmap."""

        st.subheader("Threat Landscape")

        # Sample threat data (replace with real data)
        threat_data = {
            "brute_force": {"count": 15, "severity": "high"},
            "sql_injection": {"count": 3, "severity": "critical"},
            "rate_limiting": {"count": 8, "severity": "medium"},
            "anomalous_behavior": {"count": 12, "severity": "medium"}
        }

        # Create threat severity chart
        threat_types = list(threat_data.keys())
        threat_counts = [threat_data[t]["count"] for t in threat_types]
        severity_colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#28a745"
        }

        colors = [severity_colors[threat_data[t]["severity"]] for t in threat_types]

        fig = go.Figure(data=[go.Bar(
            x=threat_types,
            y=threat_counts,
            marker_color=colors,
            text=threat_counts,
            textposition='auto'
        )])

        fig.update_layout(
            title="Security Threats by Type (Last 24 Hours)",
            xaxis_title="Threat Type",
            yaxis_title="Event Count",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_recent_security_events(self):
        """Render recent security events table."""

        st.subheader("Recent Security Events")

        # Sample security events (replace with real data)
        events_data = [
            {
                "timestamp": "2024-08-19 14:30:15",
                "type": "brute_force",
                "severity": "HIGH",
                "user_id": "admin",
                "source_ip": "192.168.1.100",
                "threat_score": 0.8
            },
            {
                "timestamp": "2024-08-19 14:28:42",
                "type": "sql_injection",
                "severity": "CRITICAL",
                "user_id": "guest",
                "source_ip": "10.0.0.50",
                "threat_score": 0.95
            },
            {
                "timestamp": "2024-08-19 14:25:33",
                "type": "rate_limiting",
                "severity": "MEDIUM",
                "user_id": "api_user",
                "source_ip": "203.0.113.10",
                "threat_score": 0.6
            }
        ]

        # Convert to DataFrame for display
        df = pd.DataFrame(events_data)

        # Style the dataframe
        def color_severity(val):
            if val == "CRITICAL":
                return "background-color: #f8d7da; color: #721c24"
            elif val == "HIGH":
                return "background-color: #fff3cd; color: #856404"
            elif val == "MEDIUM":
                return "background-color: #d1ecf1; color: #0c5460"
            return ""

        styled_df = df.style.applymap(color_severity, subset=['severity'])
        st.dataframe(styled_df, use_container_width=True)
```

## Dashboard Deployment and Hosting

### Docker Deployment

```dockerfile
# Dockerfile for dashboard deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run dashboard
CMD ["streamlit", "run", "packages/monitoring/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Kubernetes Deployment

```yaml
# k8s-dashboard-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivillage-dashboard
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aivillage-dashboard
  template:
    metadata:
      labels:
        app: aivillage-dashboard
    spec:
      containers:
      - name: dashboard
        image: aivillage/dashboard:latest
        ports:
        - containerPort: 8501
        env:
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: aivillage-dashboard-service
  namespace: monitoring
spec:
  selector:
    app: aivillage-dashboard
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aivillage-dashboard-ingress
  namespace: monitoring
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - dashboard.aivillage.ai
    secretName: dashboard-tls
  rules:
  - host: dashboard.aivillage.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aivillage-dashboard-service
            port:
              number: 80
```

### Environment Configuration

```bash
# .env file for dashboard configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Data source configuration
METRICS_DATABASE_PATH=/data/observability.db
LOGS_DIRECTORY=/data/logs
CHECKPOINTS_DIRECTORY=/data/checkpoints

# External integrations
WANDB_API_KEY=your_wandb_api_key
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Security configuration
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD_HASH=your_hashed_password

# Performance tuning
DASHBOARD_CACHE_TTL=300  # 5 minutes
DASHBOARD_MAX_CONNECTIONS=100
DASHBOARD_TIMEOUT=30
```

### Monitoring Dashboard Performance

```python
class DashboardPerformanceMonitor:
    """Monitor dashboard performance and optimize rendering."""

    def __init__(self):
        self.render_times = []
        self.cache_hit_rates = {}
        self.memory_usage = []

    @functools.lru_cache(maxsize=100)
    def cached_data_fetch(self, data_type: str, time_window: int) -> Dict[str, Any]:
        """Cache expensive data fetching operations."""

        if data_type == "system_metrics":
            return self._fetch_system_metrics(time_window)
        elif data_type == "pipeline_status":
            return self._fetch_pipeline_status()
        elif data_type == "security_events":
            return self._fetch_security_events(time_window)

        return {}

    def track_render_performance(self, component_name: str, render_time: float):
        """Track component rendering performance."""

        self.render_times.append({
            "component": component_name,
            "render_time": render_time,
            "timestamp": time.time()
        })

        # Alert if rendering is slow
        if render_time > 5.0:  # 5 seconds
            logger.warning(f"Slow dashboard rendering: {component_name} took {render_time:.2f}s")

    def optimize_data_loading(self):
        """Implement data loading optimizations."""

        # Use Streamlit caching for expensive operations
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def load_dashboard_data():
            return {
                "system_metrics": self.cached_data_fetch("system_metrics", 3600),
                "pipeline_status": self.cached_data_fetch("pipeline_status", 0),
                "security_events": self.cached_data_fetch("security_events", 86400)
            }

        return load_dashboard_data()
```

This comprehensive dashboard configuration guide provides the foundation for creating powerful, real-time monitoring interfaces for the AIVillage platform. The dashboards combine live data visualization, intelligent caching, and responsive design to deliver optimal user experience while maintaining high performance.
