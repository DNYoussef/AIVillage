"""
Performance Visualization Tools

Creates comprehensive visual reports and charts for benchmark results,
performance comparisons, and system metrics analysis.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dataclasses import asdict

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Creates comprehensive visualizations for benchmark results"""
    
    def __init__(self, output_dir: str = "swarm/phase2/benchmarks/backend/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comparison_dashboard(self, results: Dict[str, Any], timestamp: str = None) -> str:
        """Create a comprehensive comparison dashboard"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dashboard_file = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Throughput Comparison',
                'Latency Distribution', 
                'Memory Usage Comparison',
                'Success Rate Comparison',
                'Resource Utilization',
                'Performance Score Summary'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Extract data for visualizations
        mono_data = results.get('monolithic', {})
        micro_data = results.get('microservices', {})
        comparison = results.get('comparison', {})
        
        # 1. Throughput Comparison
        benchmarks = list(comparison.keys())
        mono_throughput = [mono_data.get(b, {}).throughput for b in benchmarks]
        micro_throughput = [micro_data.get(b, {}).throughput for b in benchmarks]
        
        fig.add_trace(
            go.Bar(name='Monolithic', x=benchmarks, y=mono_throughput, 
                   marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Microservices', x=benchmarks, y=micro_throughput,
                   marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 2. Latency Distribution (P50, P95, P99)
        percentiles = ['p50', 'p95', 'p99']
        for i, benchmark in enumerate(benchmarks[:3]):  # Limit to first 3 for clarity
            mono_latencies = [
                mono_data.get(benchmark, {}).latency_stats.get(p, 0) 
                for p in percentiles
            ]
            micro_latencies = [
                micro_data.get(benchmark, {}).latency_stats.get(p, 0) 
                for p in percentiles
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=percentiles, y=mono_latencies,
                    mode='lines+markers', name=f'Mono-{benchmark}',
                    line=dict(dash='solid')
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=percentiles, y=micro_latencies,
                    mode='lines+markers', name=f'Micro-{benchmark}',
                    line=dict(dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Memory Usage Comparison
        mono_memory = [
            mono_data.get(b, {}).resource_usage.get('memory', {}).get('peak_mb', 0)
            for b in benchmarks
        ]
        micro_memory = [
            micro_data.get(b, {}).resource_usage.get('memory', {}).get('peak_mb', 0)
            for b in benchmarks
        ]
        
        fig.add_trace(
            go.Bar(name='Monolithic Memory', x=benchmarks, y=mono_memory,
                   marker_color='lightgreen'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Microservices Memory', x=benchmarks, y=micro_memory,
                   marker_color='orange'),
            row=2, col=1
        )
        
        # 4. Success Rate Comparison
        mono_success = [mono_data.get(b, {}).success_rate * 100 for b in benchmarks]
        micro_success = [micro_data.get(b, {}).success_rate * 100 for b in benchmarks]
        
        fig.add_trace(
            go.Scatter(
                x=benchmarks, y=mono_success,
                mode='lines+markers', name='Monolithic Success Rate',
                line=dict(color='blue')
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=benchmarks, y=micro_success,
                mode='lines+markers', name='Microservices Success Rate',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # 5. Resource Utilization (CPU and Memory combined)
        mono_cpu = [
            mono_data.get(b, {}).resource_usage.get('cpu', {}).get('avg', 0)
            for b in benchmarks
        ]
        micro_cpu = [
            micro_data.get(b, {}).resource_usage.get('cpu', {}).get('avg', 0)
            for b in benchmarks
        ]
        
        fig.add_trace(
            go.Scatter(
                x=benchmarks, y=mono_cpu,
                mode='lines+markers', name='Monolithic CPU',
                yaxis='y5'
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=benchmarks, y=micro_cpu,
                mode='lines+markers', name='Microservices CPU',
                yaxis='y5'
            ),
            row=3, col=1
        )
        
        # 6. Performance Score Summary
        scores = [comparison.get(b, {}).get('overall_score', 'POOR') for b in benchmarks]
        score_values = {'POOR': 1, 'FAIR': 2, 'GOOD': 3, 'EXCELLENT': 4}
        score_nums = [score_values.get(score, 1) for score in scores]
        
        fig.add_trace(
            go.Bar(
                x=benchmarks, y=score_nums,
                text=scores, textposition='inside',
                marker_color=['red' if s < 2 else 'orange' if s < 3 else 'lightgreen' 
                             for s in score_nums]
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Performance Benchmark Comparison Dashboard - {timestamp}',
            height=900,
            showlegend=True
        )
        
        # Save interactive dashboard
        fig.write_html(dashboard_file)
        
        return dashboard_file
    
    def create_latency_heatmap(self, results: Dict[str, Any], timestamp: str = None) -> str:
        """Create latency distribution heatmap"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for heatmap
        benchmarks = list(results.get('comparison', {}).keys())
        percentiles = ['p50', 'p95', 'p99']
        
        mono_data = []
        micro_data = []
        
        for benchmark in benchmarks:
            mono_latencies = []
            micro_latencies = []
            
            for p in percentiles:
                mono_lat = results.get('monolithic', {}).get(benchmark, {}).latency_stats.get(p, 0)
                micro_lat = results.get('microservices', {}).get(benchmark, {}).latency_stats.get(p, 0)
                
                mono_latencies.append(mono_lat)
                micro_latencies.append(micro_lat)
            
            mono_data.append(mono_latencies)
            micro_data.append(micro_latencies)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Monolithic heatmap
        sns.heatmap(mono_data, 
                   xticklabels=percentiles,
                   yticklabels=benchmarks,
                   annot=True, fmt='.1f',
                   cmap='YlOrRd', ax=ax1,
                   cbar_kws={'label': 'Latency (ms)'})
        ax1.set_title('Monolithic Architecture - Latency Distribution')
        ax1.set_xlabel('Percentiles')
        ax1.set_ylabel('Benchmarks')
        
        # Microservices heatmap
        sns.heatmap(micro_data,
                   xticklabels=percentiles,
                   yticklabels=benchmarks,
                   annot=True, fmt='.1f',
                   cmap='YlOrRd', ax=ax2,
                   cbar_kws={'label': 'Latency (ms)'})
        ax2.set_title('Microservices Architecture - Latency Distribution')
        ax2.set_xlabel('Percentiles')
        ax2.set_ylabel('Benchmarks')
        
        plt.tight_layout()
        
        heatmap_file = os.path.join(self.output_dir, f"latency_heatmap_{timestamp}.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_file
    
    def create_performance_timeline(self, results: Dict[str, Any], timestamp: str = None) -> str:
        """Create performance metrics timeline"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract timeline data from concurrent load benchmark
        timeline_data = None
        for arch in ['monolithic', 'microservices']:
            concurrent_result = results.get(arch, {}).get('concurrent_load', {})
            if hasattr(concurrent_result, 'metadata'):
                timeline = concurrent_result.metadata.get('concurrent_timeline', [])
                if timeline:
                    timeline_data = timeline
                    break
        
        if not timeline_data:
            print("No timeline data available for visualization")
            return None
        
        # Create timeline visualization
        times = [point[0] for point in timeline_data]
        concurrent_requests = [point[1] for point in timeline_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=concurrent_requests,
            mode='lines+markers',
            name='Concurrent Requests',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Concurrent Request Timeline',
            xaxis_title='Time (seconds)',
            yaxis_title='Number of Concurrent Requests',
            height=400
        )
        
        timeline_file = os.path.join(self.output_dir, f"timeline_{timestamp}.html")
        fig.write_html(timeline_file)
        
        return timeline_file
    
    def create_memory_usage_charts(self, results: Dict[str, Any], timestamp: str = None) -> str:
        """Create detailed memory usage analysis charts"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract memory data
        benchmarks = list(results.get('comparison', {}).keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Peak Memory Usage',
                'Memory Growth Rate',
                'Memory Efficiency Score',
                'GC Activity Comparison'
            )
        )
        
        # 1. Peak Memory Usage
        mono_peak = []
        micro_peak = []
        
        for benchmark in benchmarks:
            mono_mem = results.get('monolithic', {}).get(benchmark, {}).resource_usage.get('memory', {}).get('peak_mb', 0)
            micro_mem = results.get('microservices', {}).get(benchmark, {}).resource_usage.get('memory', {}).get('peak_mb', 0)
            
            mono_peak.append(mono_mem)
            micro_peak.append(micro_mem)
        
        fig.add_trace(
            go.Bar(name='Monolithic', x=benchmarks, y=mono_peak),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Microservices', x=benchmarks, y=micro_peak),
            row=1, col=1
        )
        
        # 2. Memory Growth Comparison
        growth_comparison = []
        for benchmark in benchmarks:
            change = results.get('comparison', {}).get(benchmark, {}).get('memory_change_percent', 0)
            growth_comparison.append(change)
        
        colors = ['green' if g < 0 else 'red' if g > 10 else 'orange' for g in growth_comparison]
        
        fig.add_trace(
            go.Bar(
                x=benchmarks, y=growth_comparison,
                marker_color=colors,
                name='Memory Change %'
            ),
            row=1, col=2
        )
        
        # 3. Memory Efficiency Score (derived from comparison data)
        efficiency_scores = []
        for benchmark in benchmarks:
            score_text = results.get('comparison', {}).get(benchmark, {}).get('overall_score', 'POOR')
            score_value = {'POOR': 1, 'FAIR': 2, 'GOOD': 3, 'EXCELLENT': 4}.get(score_text, 1)
            efficiency_scores.append(score_value)
        
        fig.add_trace(
            go.Scatter(
                x=benchmarks, y=efficiency_scores,
                mode='lines+markers+text',
                text=[f'{s}/4' for s in efficiency_scores],
                textposition='top center',
                name='Efficiency Score'
            ),
            row=2, col=1
        )
        
        # 4. Resource utilization comparison
        mono_cpu = []
        micro_cpu = []
        
        for benchmark in benchmarks:
            mono_c = results.get('monolithic', {}).get(benchmark, {}).resource_usage.get('cpu', {}).get('avg', 0)
            micro_c = results.get('microservices', {}).get(benchmark, {}).resource_usage.get('cpu', {}).get('avg', 0)
            
            mono_cpu.append(mono_c)
            micro_cpu.append(micro_c)
        
        fig.add_trace(
            go.Bar(name='Monolithic CPU', x=benchmarks, y=mono_cpu),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name='Microservices CPU', x=benchmarks, y=micro_cpu),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Memory Usage Analysis',
            height=800
        )
        
        memory_file = os.path.join(self.output_dir, f"memory_analysis_{timestamp}.html")
        fig.write_html(memory_file)
        
        return memory_file
    
    def create_summary_report(self, results: Dict[str, Any], validation: Dict[str, bool], 
                             timestamp: str = None) -> str:
        """Create a comprehensive summary report"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_file = os.path.join(self.output_dir, f"visual_summary_{timestamp}.html")
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Summary - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; min-width: 200px; }}
                .good {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
                .bad {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .validation {{ margin: 10px 0; padding: 10px; border-radius: 3px; }}
                .pass {{ background-color: #d4edda; }}
                .fail {{ background-color: #f8d7da; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Performance Benchmark Summary Report</h1>
                <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>
                <p><strong>Comparison:</strong> Monolithic vs Microservices Architecture</p>
            </div>
        """
        
        # Validation Summary
        html_content += """
            <div class="section">
                <h2>📋 Validation Summary</h2>
        """
        
        passed_checks = sum(validation.values())
        total_checks = len(validation)
        
        html_content += f"""
                <div class="metric {'good' if passed_checks == total_checks else 'warning' if passed_checks > total_checks/2 else 'bad'}">
                    <h3>Overall Result</h3>
                    <p><strong>{passed_checks}/{total_checks}</strong> checks passed</p>
                </div>
        """
        
        for check, result in validation.items():
            status_class = 'pass' if result else 'fail'
            status_text = '✅ PASS' if result else '❌ FAIL'
            
            html_content += f"""
                <div class="validation {status_class}">
                    <strong>{status_text}</strong> {check.replace('_', ' ').title()}
                </div>
            """
        
        html_content += "</div>"
        
        # Performance Metrics Table
        html_content += """
            <div class="section">
                <h2>📊 Performance Metrics Comparison</h2>
                <table>
                    <tr>
                        <th>Benchmark</th>
                        <th>Overall Score</th>
                        <th>Throughput Change</th>
                        <th>Latency Change</th>
                        <th>Memory Change</th>
                        <th>Status</th>
                    </tr>
        """
        
        comparison = results.get('comparison', {})
        for benchmark, analysis in comparison.items():
            score = analysis.get('overall_score', 'POOR')
            throughput_change = analysis.get('throughput_change_percent', 0)
            latency_change = analysis.get('latency_change_percent', 0)
            memory_change = analysis.get('memory_change_percent', 0)
            
            # Determine row class based on overall performance
            if score in ['EXCELLENT', 'GOOD']:
                row_class = 'good'
                status = '✅'
            elif score == 'FAIR':
                row_class = 'warning'
                status = '⚠️'
            else:
                row_class = 'bad'
                status = '❌'
            
            html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{benchmark.replace('_', ' ').title()}</strong></td>
                    <td>{score}</td>
                    <td>{throughput_change:+.1f}%</td>
                    <td>{latency_change:+.1f}%</td>
                    <td>{memory_change:+.1f}%</td>
                    <td>{status}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Key Insights
        html_content += """
            <div class="section">
                <h2>🔍 Key Insights</h2>
        """
        
        insights = self._generate_insights(results, validation)
        for insight in insights:
            html_content += f"<p>• {insight}</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>📈 Generated Visualizations</h2>
                <p>Additional charts and dashboards have been generated:</p>
                <ul>
                    <li>Interactive Performance Dashboard</li>
                    <li>Latency Distribution Heatmap</li>
                    <li>Memory Usage Analysis</li>
                    <li>Performance Timeline (if available)</li>
                </ul>
            </div>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc;">
                <p><em>Generated by AI Village Performance Benchmark Suite</em></p>
            </footer>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        return report_file
    
    def _generate_insights(self, results: Dict[str, Any], validation: Dict[str, bool]) -> List[str]:
        """Generate key insights from benchmark results"""
        insights = []
        
        comparison = results.get('comparison', {})
        
        # Overall assessment
        if validation.get('no_performance_regression', False):
            insights.append("✅ No significant performance regressions detected in microservices architecture")
        else:
            insights.append("⚠️ Some performance regressions detected - review optimization opportunities")
        
        # Memory insights
        if validation.get('memory_efficiency', False):
            insights.append("💾 Memory usage is efficient or improved in microservices")
        else:
            insights.append("📈 Memory usage increased - consider memory optimization strategies")
        
        # Latency insights
        avg_latency_change = np.mean([
            analysis.get('latency_change_percent', 0)
            for analysis in comparison.values()
        ])
        
        if avg_latency_change < -5:
            insights.append(f"⚡ Average latency improved by {abs(avg_latency_change):.1f}%")
        elif avg_latency_change > 10:
            insights.append(f"🐌 Average latency increased by {avg_latency_change:.1f}% - needs attention")
        
        # Throughput insights
        avg_throughput_change = np.mean([
            analysis.get('throughput_change_percent', 0)
            for analysis in comparison.values()
        ])
        
        if avg_throughput_change > 5:
            insights.append(f"🚀 Average throughput improved by {avg_throughput_change:.1f}%")
        elif avg_throughput_change < -10:
            insights.append(f"📉 Average throughput decreased by {abs(avg_throughput_change):.1f}%")
        
        # Scalability insights
        if validation.get('scalability_improvement', False):
            insights.append("📊 Concurrent request handling improved")
        else:
            insights.append("⚖️ Concurrent handling needs optimization")
        
        # Best performing benchmarks
        best_benchmark = max(comparison.items(), 
                           key=lambda x: {'EXCELLENT': 4, 'GOOD': 3, 'FAIR': 2, 'POOR': 1}
                           .get(x[1].get('overall_score', 'POOR'), 1))
        
        insights.append(f"🏆 Best performing benchmark: {best_benchmark[0].replace('_', ' ').title()}")
        
        return insights
    
    def create_all_visualizations(self, results: Dict[str, Any], validation: Dict[str, bool]) -> Dict[str, str]:
        """Create all visualization files and return their paths"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files = {}
        
        try:
            files['dashboard'] = self.create_comparison_dashboard(results, timestamp)
            print(f"✅ Created interactive dashboard: {files['dashboard']}")
        except Exception as e:
            print(f"⚠️ Failed to create dashboard: {e}")
        
        try:
            files['heatmap'] = self.create_latency_heatmap(results, timestamp)
            print(f"✅ Created latency heatmap: {files['heatmap']}")
        except Exception as e:
            print(f"⚠️ Failed to create heatmap: {e}")
        
        try:
            files['memory_analysis'] = self.create_memory_usage_charts(results, timestamp)
            print(f"✅ Created memory analysis: {files['memory_analysis']}")
        except Exception as e:
            print(f"⚠️ Failed to create memory analysis: {e}")
        
        try:
            timeline_file = self.create_performance_timeline(results, timestamp)
            if timeline_file:
                files['timeline'] = timeline_file
                print(f"✅ Created performance timeline: {files['timeline']}")
        except Exception as e:
            print(f"⚠️ Failed to create timeline: {e}")
        
        try:
            files['summary'] = self.create_summary_report(results, validation, timestamp)
            print(f"✅ Created summary report: {files['summary']}")
        except Exception as e:
            print(f"⚠️ Failed to create summary report: {e}")
        
        return files

# Export main class
__all__ = ['PerformanceVisualizer']