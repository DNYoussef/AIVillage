#!/usr/bin/env python3
"""
Architectural Health Dashboard

A web-based dashboard for monitoring architectural fitness functions,
displaying real-time metrics, trends, and recommendations.

Usage:
    python scripts/architecture_dashboard.py --port 8080
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


class ArchitectureDashboard:
    """Interactive dashboard for architectural health monitoring"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports" / "architecture"
        self.config_dir = project_root / "config"

        # Load configuration
        self.load_config()

        # Load historical data
        self.historical_data = self.load_historical_data()

    def load_config(self):
        """Load dashboard configuration"""
        config_file = self.config_dir / "architecture_rules.yaml"
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    def load_historical_data(self) -> list[dict]:
        """Load historical architectural analysis reports"""
        if not self.reports_dir.exists():
            return []

        reports = []
        for report_file in self.reports_dir.glob("architecture_report_*.json"):
            try:
                with open(report_file) as f:
                    report_data = json.load(f)
                    reports.append(report_data)
            except Exception as e:
                st.warning(f"Could not load report {report_file}: {e}")
                continue

        # Sort by timestamp
        reports.sort(key=lambda x: x.get("timestamp", ""))
        return reports

    def get_latest_report(self) -> dict | None:
        """Get the most recent architectural report"""
        if self.historical_data:
            return self.historical_data[-1]
        return None

    def create_overview_metrics(self):
        """Create overview metrics section"""
        st.header("🏗️ Architecture Health Overview")

        latest_report = self.get_latest_report()
        if not latest_report:
            st.error("No architectural reports found. Run architectural analysis first.")
            return

        summary = latest_report.get("summary", {})
        quality_gates = latest_report.get("quality_gates", {})

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gates_passed = sum(quality_gates.values())
            gates_total = len(quality_gates)
            gate_percentage = (gates_passed / gates_total * 100) if gates_total > 0 else 0

            st.metric("Quality Gates", f"{gates_passed}/{gates_total}", delta=f"{gate_percentage:.0f}%")

        with col2:
            coupling = summary.get("average_coupling", 0)
            st.metric("Avg Coupling", f"{coupling:.3f}", delta="Good" if coupling < 0.3 else "High")

        with col3:
            debt_items = summary.get("technical_debt_items", 0)
            st.metric("Tech Debt Items", debt_items, delta="Good" if debt_items < 5 else "Needs Attention")

        with col4:
            critical_violations = summary.get("critical_violations", 0)
            st.metric(
                "Critical Issues", critical_violations, delta="Good" if critical_violations == 0 else "Action Required"
            )

        # Health score calculation
        health_score = self.calculate_health_score(latest_report)

        # Health score gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Architecture Health Score"},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "lightgreen"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
                },
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    def calculate_health_score(self, report: dict) -> float:
        """Calculate overall architecture health score"""
        quality_gates = report.get("quality_gates", {})
        summary = report.get("summary", {})

        # Quality gates (40% weight)
        gates_passed = sum(quality_gates.values())
        gates_total = len(quality_gates)
        if gates_total > 0:
            gates_score = (gates_passed / gates_total) * 40
        else:
            gates_score = 0

        # Coupling (20% weight)
        coupling = summary.get("average_coupling", 0)
        coupling_score = max(0, (0.5 - coupling) / 0.5) * 20

        # Technical debt (20% weight)
        debt_items = summary.get("technical_debt_items", 0)
        debt_score = max(0, (10 - debt_items) / 10) * 20

        # Critical violations (20% weight)
        critical_violations = summary.get("critical_violations", 0)
        critical_score = max(0, (5 - critical_violations) / 5) * 20

        total_score = gates_score + coupling_score + debt_score + critical_score
        return min(100, max(0, total_score))

    def create_trends_section(self):
        """Create trends and historical analysis section"""
        st.header("📈 Trends & Historical Analysis")

        if len(self.historical_data) < 2:
            st.info("Need at least 2 reports to show trends")
            return

        # Prepare trend data
        trend_data = []
        for report in self.historical_data:
            summary = report.get("summary", {})
            quality_gates = report.get("quality_gates", {})

            trend_data.append(
                {
                    "timestamp": report.get("timestamp"),
                    "health_score": self.calculate_health_score(report),
                    "coupling": summary.get("average_coupling", 0),
                    "debt_items": summary.get("technical_debt_items", 0),
                    "critical_violations": summary.get("critical_violations", 0),
                    "quality_gates_passed": sum(quality_gates.values()),
                    "quality_gates_total": len(quality_gates),
                }
            )

        df = pd.DataFrame(trend_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Multi-metric trend chart
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Health Score", "Average Coupling", "Technical Debt", "Critical Violations"),
            vertical_spacing=0.1,
        )

        # Health score trend
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["health_score"], name="Health Score", line=dict(color="green")),
            row=1,
            col=1,
        )

        # Coupling trend
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["coupling"], name="Avg Coupling", line=dict(color="blue")), row=1, col=2
        )

        # Technical debt trend
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["debt_items"], name="Tech Debt", line=dict(color="orange")), row=2, col=1
        )

        # Critical violations trend
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["critical_violations"], name="Critical Issues", line=dict(color="red")),
            row=2,
            col=2,
        )

        fig.update_layout(height=500, showlegend=False, title_text="Architecture Metrics Trends")

        st.plotly_chart(fig, use_container_width=True)

    def create_dependencies_section(self):
        """Create dependency analysis section"""
        st.header("🔗 Dependency Analysis")

        latest_report = self.get_latest_report()
        if not latest_report:
            return

        dependency_metrics = latest_report.get("dependency_metrics", {})
        coupling_metrics = latest_report.get("coupling_metrics", [])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dependency Overview")

            total_modules = dependency_metrics.get("total_modules", 0)
            total_deps = dependency_metrics.get("total_dependencies", 0)
            circular_deps = dependency_metrics.get("circular_dependencies", 0)

            st.metric("Total Modules", total_modules)
            st.metric("Total Dependencies", total_deps)
            st.metric("Circular Dependencies", circular_deps, delta="Good" if circular_deps == 0 else "Fix Required")

        with col2:
            st.subheader("Coupling Distribution")

            if coupling_metrics:
                coupling_df = pd.DataFrame(coupling_metrics)

                fig = px.histogram(coupling_df, x="instability", title="Module Instability Distribution", nbins=20)
                fig.update_xaxis(title="Instability (0=Stable, 1=Unstable)")
                fig.update_yaxis(title="Number of Modules")

                st.plotly_chart(fig, use_container_width=True)

    def create_violations_section(self):
        """Create violations and issues section"""
        st.header("⚠️ Violations & Issues")

        latest_report = self.get_latest_report()
        if not latest_report:
            return

        # Connascence violations
        connascence_metrics = latest_report.get("connascence_metrics", [])
        if connascence_metrics:
            st.subheader("Connascence Violations")

            # Group by severity and type
            connascence_df = pd.DataFrame(connascence_metrics)

            # Severity distribution
            severity_counts = connascence_df["severity"].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values, names=severity_counts.index, title="Violations by Severity"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_severity, use_container_width=True)

            # Type distribution
            with col2:
                type_counts = connascence_df["type"].value_counts()
                fig_type = px.bar(x=type_counts.index, y=type_counts.values, title="Violations by Type")
                st.plotly_chart(fig_type, use_container_width=True)

            # Detailed violations table
            st.subheader("Detailed Violations")
            violations_display = connascence_df[["type", "severity", "instances", "locality"]]
            st.dataframe(violations_display, use_container_width=True)

        # Technical debt
        technical_debt = latest_report.get("technical_debt", [])
        if technical_debt:
            st.subheader("Technical Debt")

            debt_df = pd.DataFrame(technical_debt)

            # Debt by category
            category_amounts = debt_df.groupby("category")["amount"].sum().sort_values(ascending=True)

            fig_debt = px.bar(
                x=category_amounts.values, y=category_amounts.index, orientation="h", title="Technical Debt by Category"
            )
            fig_debt.update_xaxis(title="Debt Amount")

            st.plotly_chart(fig_debt, use_container_width=True)

            # Risk distribution
            risk_counts = debt_df["risk_level"].value_counts()
            colors = {"low": "green", "medium": "yellow", "high": "red"}

            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Technical Debt by Risk Level",
                color=risk_counts.index,
                color_discrete_map=colors,
            )

            st.plotly_chart(fig_risk, use_container_width=True)

    def create_recommendations_section(self):
        """Create recommendations section"""
        st.header("💡 Recommendations")

        latest_report = self.get_latest_report()
        if not latest_report:
            return

        recommendations = latest_report.get("recommendations", [])

        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                st.info(f"**{i}.** {recommendation}")
        else:
            st.success("🎉 No architectural improvements needed at this time!")

        # Priority matrix for improvements
        st.subheader("Improvement Priority Matrix")

        # Calculate priority scores based on violations and debt
        connascence_metrics = latest_report.get("connascence_metrics", [])
        technical_debt = latest_report.get("technical_debt", [])
        architectural_drift = latest_report.get("architectural_drift", [])

        priority_items = []

        # High severity connascence violations
        for violation in connascence_metrics:
            if violation["severity"] in ["critical", "high"]:
                priority_items.append(
                    {
                        "item": f"Fix {violation['type']} connascence",
                        "impact": 8 if violation["severity"] == "critical" else 6,
                        "effort": violation["instances"] * 2,  # Simplified effort calculation
                        "type": "Connascence",
                    }
                )

        # High risk technical debt
        for debt in technical_debt:
            if debt["risk_level"] == "high":
                priority_items.append(
                    {
                        "item": f"Address {debt['category']} debt",
                        "impact": 7,
                        "effort": debt["effort_hours"],
                        "type": "Technical Debt",
                    }
                )

        # Critical architectural drift
        for drift in architectural_drift:
            if drift["severity"] > 0.5:
                priority_items.append(
                    {
                        "item": f"Fix {drift['drift_type']}",
                        "impact": int(drift["severity"] * 10),
                        "effort": len(drift["files_affected"]),
                        "type": "Architectural Drift",
                    }
                )

        if priority_items:
            df_priority = pd.DataFrame(priority_items)

            fig_priority = px.scatter(
                df_priority,
                x="effort",
                y="impact",
                color="type",
                size="impact",
                hover_name="item",
                title="Improvement Priority Matrix (High Impact, Low Effort = High Priority)",
                labels={"effort": "Effort Required", "impact": "Impact Score"},
            )

            # Add quadrant lines
            fig_priority.add_hline(y=df_priority["impact"].median(), line_dash="dash", line_color="gray")
            fig_priority.add_vline(x=df_priority["effort"].median(), line_dash="dash", line_color="gray")

            st.plotly_chart(fig_priority, use_container_width=True)

            # Priority table
            df_priority["priority_score"] = df_priority["impact"] / (df_priority["effort"] + 1)
            df_priority_sorted = df_priority.sort_values("priority_score", ascending=False)

            st.subheader("Prioritized Action Items")
            st.dataframe(
                df_priority_sorted[["item", "type", "impact", "effort", "priority_score"]], use_container_width=True
            )

    def create_configuration_section(self):
        """Create configuration and rules section"""
        st.header("⚙️ Configuration & Rules")

        st.subheader("Current Architecture Rules")

        if self.config:
            # Display key configuration items
            col1, col2 = st.columns(2)

            with col1:
                st.write("**File & Complexity Limits:**")
                st.write(f"- Max file lines: {self.config.get('max_file_lines', 'Not set')}")
                st.write(f"- Max function complexity: {self.config.get('max_function_complexity', 'Not set')}")
                st.write(f"- Max function parameters: {self.config.get('max_function_parameters', 'Not set')}")
                st.write(f"- Max coupling threshold: {self.config.get('max_coupling_threshold', 'Not set')}")

            with col2:
                st.write("**Quality Thresholds:**")
                quality_thresholds = self.config.get("quality_thresholds", {})
                for key, value in quality_thresholds.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")

            # Dependency rules
            st.subheader("Layer Dependencies")
            allowed_deps = self.config.get("allowed_dependencies", {})
            if allowed_deps:
                for layer, dependencies in allowed_deps.items():
                    st.write(f"**{layer}** can depend on: {', '.join(dependencies)}")

            # Security rules
            st.subheader("Security Rules")
            forbidden_patterns = self.config.get("forbidden_patterns", [])
            if forbidden_patterns:
                st.write("**Forbidden patterns:**")
                for pattern in forbidden_patterns:
                    st.code(pattern, language="text")
        else:
            st.warning("No configuration file loaded")

        # Configuration editor (simple)
        st.subheader("Quick Configuration Updates")

        with st.expander("Edit Thresholds"):
            st.number_input(
                "Max Function Complexity",
                value=self.config.get("max_function_complexity", 10),
                min_value=1,
                max_value=50,
            )

            st.number_input(
                "Max Coupling Threshold",
                value=self.config.get("max_coupling_threshold", 0.3),
                min_value=0.0,
                max_value=1.0,
                step=0.1,
            )

            if st.button("Update Configuration"):
                # This would update the configuration file
                st.success("Configuration updated! (This is a demo - actual file not modified)")


def main():
    st.set_page_config(page_title="AIVillage Architecture Dashboard", page_icon="🏗️", layout="wide")

    st.title("🏗️ AIVillage Architecture Health Dashboard")
    st.sidebar.title("Navigation")

    dashboard = ArchitectureDashboard(PROJECT_ROOT)

    # Sidebar navigation
    sections = {
        "Overview": dashboard.create_overview_metrics,
        "Trends": dashboard.create_trends_section,
        "Dependencies": dashboard.create_dependencies_section,
        "Violations": dashboard.create_violations_section,
        "Recommendations": dashboard.create_recommendations_section,
        "Configuration": dashboard.create_configuration_section,
    }

    selected_section = st.sidebar.selectbox("Choose Section", list(sections.keys()))

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Last update info
    latest_report = dashboard.get_latest_report()
    if latest_report:
        last_update = latest_report.get("timestamp", "Unknown")
        st.sidebar.info(f"Last Analysis: {last_update}")

    # Display selected section
    sections[selected_section]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**AIVillage Architecture Dashboard**")
    st.sidebar.markdown("Built with Streamlit & Plotly")


if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(description="Architecture Health Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on")
    parser.add_argument("--host", default="localhost", help="Host to run dashboard on")

    args = parser.parse_args()

    print(f"Starting Architecture Dashboard on http://{args.host}:{args.port}")
    print("Run architectural analysis first to populate the dashboard with data.")
    print("Use: python scripts/architectural_analysis.py --output-dir reports/architecture")

    main()
