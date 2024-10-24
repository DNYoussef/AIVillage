"""Visualization utilities for MAGI agent system."""

import os
import json
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
from ..core.exceptions import MAGIException
from .logging import get_logger

logger = get_logger(__name__)

class Visualizer:
    """Base class for visualization utilities."""
    
    def __init__(self, output_dir: Union[str, Path] = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_figure(self, fig: plt.Figure, filename: str) -> str:
        """Save figure to file."""
        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        return str(filepath)

class NetworkVisualizer(Visualizer):
    """Visualize network graphs."""
    
    def visualize_agent_network(
        self,
        graph: nx.Graph,
        filename: str = "agent_network.png",
        layout: str = "spring"
    ) -> str:
        """Visualize agent interaction network."""
        plt.figure(figsize=(12, 8))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(graph)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        else:
            pos = nx.random_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color='lightblue',
            node_size=1000,
            alpha=0.6
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color='gray',
            width=1,
            alpha=0.5
        )
        
        # Add labels
        nx.draw_networkx_labels(graph, pos)
        
        plt.title("Agent Interaction Network")
        return self.save_figure(plt.gcf(), filename)

class PerformanceVisualizer(Visualizer):
    """Visualize performance metrics."""
    
    def plot_performance_trend(
        self,
        metrics: List[Dict[str, Any]],
        filename: str = "performance_trend.png"
    ) -> str:
        """Plot performance trend over time."""
        timestamps = [m['timestamp'] for m in metrics]
        values = [m['value'] for m in metrics]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker='o')
        plt.title("Performance Trend")
        plt.xlabel("Time")
        plt.ylabel("Performance")
        plt.grid(True)
        
        return self.save_figure(plt.gcf(), filename)
    
    def plot_resource_usage(
        self,
        cpu_usage: List[float],
        memory_usage: List[float],
        filename: str = "resource_usage.png"
    ) -> str:
        """Plot resource usage over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU usage
        ax1.plot(cpu_usage, color='blue')
        ax1.set_title("CPU Usage")
        ax1.set_ylabel("Percentage")
        ax1.grid(True)
        
        # Memory usage
        ax2.plot(memory_usage, color='red')
        ax2.set_title("Memory Usage")
        ax2.set_ylabel("Percentage")
        ax2.grid(True)
        
        plt.tight_layout()
        return self.save_figure(plt.gcf(), filename)

class TaskVisualizer(Visualizer):
    """Visualize task-related information."""
    
    def plot_task_distribution(
        self,
        tasks: List[Dict[str, Any]],
        filename: str = "task_distribution.png"
    ) -> str:
        """Plot distribution of task types."""
        task_types = [task['type'] for task in tasks]
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=task_types)
        plt.title("Task Type Distribution")
        plt.xticks(rotation=45)
        
        return self.save_figure(plt.gcf(), filename)
    
    def plot_task_timeline(
        self,
        tasks: List[Dict[str, Any]],
        filename: str = "task_timeline.png"
    ) -> str:
        """Plot task execution timeline."""
        fig = go.Figure()
        
        for task in tasks:
            fig.add_trace(go.Scatter(
                x=[task['start_time'], task['end_time']],
                y=[task['id'], task['id']],
                mode='lines',
                name=f"Task {task['id']}"
            ))
        
        fig.update_layout(
            title="Task Execution Timeline",
            xaxis_title="Time",
            yaxis_title="Task ID"
        )
        
        filepath = self.output_dir / filename
        fig.write_image(str(filepath))
        return str(filepath)

class LearningVisualizer(Visualizer):
    """Visualize learning-related information."""
    
    def plot_learning_curve(
        self,
        training_scores: List[float],
        validation_scores: List[float],
        filename: str = "learning_curve.png"
    ) -> str:
        """Plot learning curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(training_scores, label='Training')
        plt.plot(validation_scores, label='Validation')
        plt.title("Learning Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        return self.save_figure(plt.gcf(), filename)
    
    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        filename: str = "confusion_matrix.png"
    ) -> str:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        return self.save_figure(plt.gcf(), filename)

class SystemVisualizer(Visualizer):
    """Visualize system-wide information."""
    
    def plot_system_health(
        self,
        metrics: Dict[str, List[float]],
        filename: str = "system_health.png"
    ) -> str:
        """Plot system health metrics."""
        fig = go.Figure()
        
        for metric_name, values in metrics.items():
            fig.add_trace(go.Scatter(
                y=values,
                name=metric_name,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="System Health Metrics",
            yaxis_title="Value",
            xaxis_title="Time"
        )
        
        filepath = self.output_dir / filename
        fig.write_image(str(filepath))
        return str(filepath)
    
    def plot_component_status(
        self,
        statuses: Dict[str, str],
        filename: str = "component_status.png"
    ) -> str:
        """Plot component status overview."""
        components = list(statuses.keys())
        status_values = [1 if status == 'healthy' else 0 for status in statuses.values()]
        
        plt.figure(figsize=(12, 6))
        plt.bar(components, status_values, color=['green' if v == 1 else 'red' for v in status_values])
        plt.title("Component Status Overview")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.2)
        
        return self.save_figure(plt.gcf(), filename)

def create_visualizer(visualizer_type: str) -> Visualizer:
    """Factory function to create visualizer instances."""
    visualizers = {
        'network': NetworkVisualizer,
        'performance': PerformanceVisualizer,
        'task': TaskVisualizer,
        'learning': LearningVisualizer,
        'system': SystemVisualizer
    }
    
    if visualizer_type not in visualizers:
        raise MAGIException(f"Unknown visualizer type: {visualizer_type}")
    
    return visualizers[visualizer_type]()
