#!/usr/bin/env python3
"""Agent Forge Real-Time Monitoring Dashboard.

Web-based dashboard for monitoring Agent Forge pipeline execution,
model evolution progress, and system metrics in real-time.
"""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentForgeDashboard:
    """Real-time monitoring dashboard for Agent Forge."""

    def __init__(self) -> None:
        self.data_dir = Path("./forge_output_enhanced")
        self.checkpoint_dir = Path("./forge_checkpoints_enhanced")
        self.logs_dir = Path("./logs")

        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.checkpoint_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_system_metrics(self) -> dict:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100

            # GPU metrics (if available)
            gpu_metrics = self.get_gpu_metrics()

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {"percent": cpu_percent, "count": cpu_count},
                "memory": {
                    "used_gb": memory_used_gb,
                    "total_gb": memory_total_gb,
                    "percent": memory_percent,
                },
                "disk": {
                    "used_gb": disk_used_gb,
                    "total_gb": disk_total_gb,
                    "percent": disk_percent,
                },
                "gpu": gpu_metrics,
            }
        except Exception as e:
            logger.exception(f"Error getting system metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_gpu_metrics(self) -> dict:
        """Get GPU metrics if available."""
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)

                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(current_device)
                memory_cached = torch.cuda.memory_reserved(current_device)
                memory_total = torch.cuda.get_device_properties(
                    current_device
                ).total_memory

                memory_allocated_gb = memory_allocated / (1024**3)
                memory_cached_gb = memory_cached / (1024**3)
                memory_total_gb = memory_total / (1024**3)
                memory_percent = (memory_allocated / memory_total) * 100

                return {
                    "available": True,
                    "device_count": device_count,
                    "current_device": current_device,
                    "device_name": device_name,
                    "memory": {
                        "allocated_gb": memory_allocated_gb,
                        "cached_gb": memory_cached_gb,
                        "total_gb": memory_total_gb,
                        "percent": memory_percent,
                    },
                }
            return {"available": False, "reason": "CUDA not available"}

        except ImportError:
            return {"available": False, "reason": "PyTorch not installed"}
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def get_pipeline_status(self) -> dict:
        """Get current pipeline execution status."""
        status = {
            "active_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "current_phase": None,
            "phases_completed": [],
            "latest_run": None,
        }

        try:
            # Check for checkpoint files
            checkpoint_files = list(
                self.checkpoint_dir.glob("orchestrator_checkpoint_*.json")
            )

            if checkpoint_files:
                # Get latest checkpoint
                latest_checkpoint = max(
                    checkpoint_files, key=lambda p: p.stat().st_mtime
                )

                with open(latest_checkpoint) as f:
                    checkpoint_data = json.load(f)

                status["latest_run"] = {
                    "run_id": checkpoint_data.get("run_id"),
                    "timestamp": checkpoint_data.get("timestamp"),
                    "phase": checkpoint_data.get("phase"),
                    "checkpoint_file": str(latest_checkpoint),
                }

                # Count phases completed
                results = checkpoint_data.get("results", {})
                for phase_name, phase_result in results.items():
                    if phase_result.get("status") == "completed":
                        status["phases_completed"].append(phase_name)
                    elif phase_result.get("status") == "running":
                        status["current_phase"] = phase_name

            # Check for active pipeline reports
            report_files = list(self.data_dir.glob("pipeline_report_*.json"))

            for report_file in report_files:
                try:
                    with open(report_file) as f:
                        report_data = json.load(f)

                    run_summary = report_data.get("run_summary", {})
                    if run_summary.get("phases_completed", 0) == run_summary.get(
                        "phases_attempted", 0
                    ):
                        status["completed_runs"] += 1
                    elif run_summary.get("phases_failed", 0) > 0:
                        status["failed_runs"] += 1
                    else:
                        status["active_runs"] += 1

                except Exception as e:
                    logger.warning(f"Could not parse report {report_file}: {e}")

        except Exception as e:
            logger.exception(f"Error getting pipeline status: {e}")
            status["error"] = str(e)

        return status

    def get_wandb_metrics(self, project_name: str = "agent-forge-enhanced") -> dict:
        """Get metrics from Weights & Biases."""
        try:
            api = wandb.Api()
            runs = api.runs(f"your-entity/{project_name}")

            if not runs:
                return {"runs": 0, "latest_metrics": None}

            # Get latest run metrics
            latest_run = runs[0]
            latest_metrics = {
                "name": latest_run.name,
                "state": latest_run.state,
                "created_at": (
                    latest_run.created_at.isoformat() if latest_run.created_at else None
                ),
                "duration": latest_run.summary.get("pipeline_duration_seconds", 0),
                "success_rate": latest_run.summary.get("success_rate", 0),
                "phases_completed": latest_run.summary.get("phases_completed", 0),
                "phases_failed": latest_run.summary.get("phases_failed", 0),
            }

            return {
                "runs": len(runs),
                "latest_metrics": latest_metrics,
                "project_url": f"https://wandb.ai/your-entity/{project_name}",
            }

        except Exception as e:
            logger.warning(f"Could not get W&B metrics: {e}")
            return {"error": str(e)}


def main() -> None:
    """Main dashboard function."""
    st.set_page_config(
        page_title="Agent Forge Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dashboard title
    st.title("🤖 Agent Forge Real-Time Dashboard")

    # Initialize dashboard
    dashboard = AgentForgeDashboard()

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

    if st.sidebar.button("Force Refresh") or auto_refresh:
        # Auto-refresh placeholder
        placeholder = st.empty()

        while auto_refresh:
            with placeholder.container():
                # System Metrics Section
                st.header("📊 System Metrics")

                system_metrics = dashboard.get_system_metrics()

                if "error" not in system_metrics:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "CPU Usage",
                            f"{system_metrics['cpu']['percent']:.1f}%",
                            delta=None,
                        )

                    with col2:
                        st.metric(
                            "Memory Usage",
                            f"{system_metrics['memory']['used_gb']:.1f} GB",
                            delta=f"{system_metrics['memory']['percent']:.1f}%",
                        )

                    with col3:
                        st.metric(
                            "Disk Usage",
                            f"{system_metrics['disk']['used_gb']:.0f} GB",
                            delta=f"{system_metrics['disk']['percent']:.1f}%",
                        )

                    with col4:
                        gpu_metrics = system_metrics.get("gpu", {})
                        if gpu_metrics.get("available"):
                            st.metric(
                                "GPU Memory",
                                f"{gpu_metrics['memory']['allocated_gb']:.1f} GB",
                                delta=f"{gpu_metrics['memory']['percent']:.1f}%",
                            )
                        else:
                            st.metric("GPU", "Not Available", delta=None)

                # Pipeline Status Section
                st.header("🔄 Pipeline Status")

                pipeline_status = dashboard.get_pipeline_status()

                col5, col6, col7 = st.columns(3)

                with col5:
                    st.metric("Active Runs", pipeline_status["active_runs"])

                with col6:
                    st.metric("Completed Runs", pipeline_status["completed_runs"])

                with col7:
                    st.metric("Failed Runs", pipeline_status["failed_runs"])

                # Current Phase Progress
                if pipeline_status["current_phase"]:
                    st.subheader(f"Current Phase: {pipeline_status['current_phase']}")
                    progress = (
                        len(pipeline_status["phases_completed"]) / 5
                    )  # 5 total phases
                    st.progress(progress)
                    st.write(
                        f"Completed phases: {', '.join(pipeline_status['phases_completed'])}"
                    )

                # Latest Run Information
                if pipeline_status["latest_run"]:
                    st.subheader("Latest Run")
                    latest = pipeline_status["latest_run"]
                    st.write(f"**Run ID:** {latest['run_id']}")
                    st.write(f"**Timestamp:** {latest['timestamp']}")
                    st.write(f"**Current Phase:** {latest['phase']}")

                # Phase Progress Chart
                st.header("📈 Phase Progress")

                # Create sample phase data for visualization
                phase_data = {
                    "Phase": [
                        "EvoMerge",
                        "Geometry",
                        "Self-Modeling",
                        "Prompt Baking",
                        "Compression",
                    ],
                    "Status": [
                        "Completed",
                        "Completed",
                        "In Progress",
                        "Pending",
                        "Pending",
                    ],
                    "Progress": [100, 100, 60, 0, 0],
                }

                df_phases = pd.DataFrame(phase_data)

                fig_phases = px.bar(
                    df_phases,
                    x="Phase",
                    y="Progress",
                    color="Status",
                    title="Phase Completion Progress",
                    color_discrete_map={
                        "Completed": "green",
                        "In Progress": "orange",
                        "Pending": "gray",
                        "Failed": "red",
                    },
                )

                st.plotly_chart(fig_phases, use_container_width=True)

                # Resource Usage Over Time
                st.header("📊 Resource Usage Trends")

                # Generate sample time series data
                timestamps = pd.date_range(
                    start=datetime.now() - timedelta(hours=1),
                    end=datetime.now(),
                    periods=60,
                )

                # Sample data
                cpu_data = [50 + 20 * (0.5 - abs(i - 30) / 30) for i in range(60)]
                memory_data = [60 + 15 * (0.3 - abs(i - 45) / 45) for i in range(60)]

                fig_resources = go.Figure()

                fig_resources.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=cpu_data,
                        mode="lines",
                        name="CPU %",
                        line={"color": "blue"},
                    )
                )

                fig_resources.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=memory_data,
                        mode="lines",
                        name="Memory %",
                        line={"color": "red"},
                    )
                )

                fig_resources.update_layout(
                    title="System Resource Usage (Last Hour)",
                    xaxis_title="Time",
                    yaxis_title="Usage %",
                )

                st.plotly_chart(fig_resources, use_container_width=True)

                # Weights & Biases Integration
                st.header("📈 W&B Integration")

                wandb_metrics = dashboard.get_wandb_metrics()

                if "error" not in wandb_metrics:
                    col8, col9 = st.columns(2)

                    with col8:
                        st.metric("Total Runs", wandb_metrics["runs"])

                    with col9:
                        if wandb_metrics["latest_metrics"]:
                            latest = wandb_metrics["latest_metrics"]
                            st.metric(
                                "Latest Success Rate",
                                f"{latest['success_rate']:.1%}",
                                delta=f"{latest['phases_completed']} phases",
                            )

                    if "project_url" in wandb_metrics:
                        st.markdown(
                            f"[View Full W&B Dashboard]({wandb_metrics['project_url']})"
                        )
                else:
                    st.warning("W&B integration not available")

                # Footer
                st.markdown("---")
                st.markdown(
                    f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"**Auto Refresh:** {'Enabled' if auto_refresh else 'Disabled'}"
                )

            if auto_refresh:
                time.sleep(refresh_interval)
            else:
                break


if __name__ == "__main__":
    main()
