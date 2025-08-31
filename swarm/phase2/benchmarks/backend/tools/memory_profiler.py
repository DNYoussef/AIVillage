"""
Memory Profiling Tools

Advanced memory usage profiling and analysis for performance benchmarking.
Tracks memory patterns, detects leaks, and provides optimization insights.
"""

import asyncio
import gc
import tracemalloc
import psutil
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
import sys
import weakref
import linecache

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    heap_size: float
    heap_used: float
    gc_objects: int
    gc_collections: Dict[int, int]
    top_allocations: List[Dict[str, Any]]
    tracemalloc_stats: Optional[Dict[str, Any]] = None

@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    source_file: str
    line_number: int
    function_name: str
    leaked_size_mb: float
    growth_rate_mb_per_sec: float
    first_seen: float
    last_seen: float
    allocation_count: int

class MemoryTracker:
    """Advanced memory tracking and leak detection"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.snapshots: List[MemorySnapshot] = []
        self.tracking_active = False
        self.process = psutil.Process()
        self.allocation_history = deque(maxlen=1000)
        self.leak_candidates = {}
        
    def start_tracking(self) -> None:
        """Start memory tracking"""
        if self.tracking_active:
            return
            
        tracemalloc.start(10)  # Track 10 stack frames
        gc.collect()  # Clean start
        
        self.tracking_active = True
        self.snapshots.clear()
        self.allocation_history.clear()
        
        # Start background tracking
        self.tracking_thread = threading.Thread(target=self._background_tracking)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
    
    def stop_tracking(self) -> None:
        """Stop memory tracking"""
        if not self.tracking_active:
            return
            
        self.tracking_active = False
        
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.join(timeout=5)
        
        tracemalloc.stop()
    
    def _background_tracking(self) -> None:
        """Background thread for continuous memory tracking"""
        while self.tracking_active:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self._update_leak_detection(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Memory tracking error: {e}")
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        # Process memory info
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        # GC statistics
        gc_stats = gc.get_stats()
        gc_collections = {i: stat['collections'] for i, stat in enumerate(gc_stats)}
        gc_objects = len(gc.get_objects())
        
        # Tracemalloc statistics
        tracemalloc_stats = None
        top_allocations = []
        
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            heap_size = peak / 1024 / 1024
            heap_used = current / 1024 / 1024
            
            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            top_allocations = []
            for stat in top_stats:
                allocation = {
                    'file': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                top_allocations.append(allocation)
            
            tracemalloc_stats = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024,
                'traced_memory': current
            }
        else:
            heap_size = 0
            heap_used = 0
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            heap_size=heap_size,
            heap_used=heap_used,
            gc_objects=gc_objects,
            gc_collections=gc_collections,
            top_allocations=top_allocations,
            tracemalloc_stats=tracemalloc_stats
        )
    
    def _update_leak_detection(self, snapshot: MemorySnapshot) -> None:
        """Update memory leak detection with new snapshot"""
        if len(self.snapshots) < 10:  # Need some history
            return
        
        # Check for consistent memory growth
        recent_snapshots = self.snapshots[-10:]
        memory_values = [s.rss_mb for s in recent_snapshots]
        timestamps = [s.timestamp for s in recent_snapshots]
        
        # Calculate growth rate
        if len(memory_values) >= 3:
            growth_rate = np.polyfit(timestamps, memory_values, 1)[0]  # MB per second
            
            # If growth rate is significant, investigate allocations
            if growth_rate > 0.1:  # More than 0.1 MB/s growth
                self._investigate_potential_leak(snapshot, growth_rate)
    
    def _investigate_potential_leak(self, snapshot: MemorySnapshot, growth_rate: float) -> None:
        """Investigate potential memory leak"""
        for allocation in snapshot.top_allocations:
            file_info = allocation['file']
            size_mb = allocation['size_mb']
            
            if file_info not in self.leak_candidates:
                self.leak_candidates[file_info] = {
                    'first_seen': snapshot.timestamp,
                    'initial_size': size_mb,
                    'growth_samples': []
                }
            
            candidate = self.leak_candidates[file_info]
            candidate['growth_samples'].append((snapshot.timestamp, size_mb))
            
            # Keep only recent samples
            if len(candidate['growth_samples']) > 20:
                candidate['growth_samples'] = candidate['growth_samples'][-20:]
    
    def get_memory_leaks(self, min_growth_rate: float = 0.05) -> List[MemoryLeak]:
        """Detect and return potential memory leaks"""
        leaks = []
        
        for file_info, candidate in self.leak_candidates.items():
            if len(candidate['growth_samples']) < 5:
                continue
            
            # Calculate growth rate for this allocation
            samples = candidate['growth_samples']
            timestamps = [s[0] for s in samples]
            sizes = [s[1] for s in samples]
            
            if len(sizes) >= 3:
                try:
                    growth_rate = np.polyfit(timestamps, sizes, 1)[0]
                    
                    if growth_rate > min_growth_rate:
                        # Parse file info
                        parts = file_info.split(':')
                        source_file = parts[0] if len(parts) > 0 else 'unknown'
                        line_number = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                        
                        leak = MemoryLeak(
                            source_file=source_file,
                            line_number=line_number,
                            function_name='unknown',
                            leaked_size_mb=sizes[-1] - sizes[0],
                            growth_rate_mb_per_sec=growth_rate,
                            first_seen=candidate['first_seen'],
                            last_seen=timestamps[-1],
                            allocation_count=len(samples)
                        )
                        leaks.append(leak)
                        
                except (ValueError, np.RankWarning):
                    continue
        
        return sorted(leaks, key=lambda x: x.growth_rate_mb_per_sec, reverse=True)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.snapshots:
            return {}
        
        rss_values = [s.rss_mb for s in self.snapshots]
        heap_values = [s.heap_used for s in self.snapshots if s.heap_used > 0]
        
        stats = {
            'duration_seconds': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'sample_count': len(self.snapshots),
            'rss_memory': {
                'initial_mb': rss_values[0],
                'final_mb': rss_values[-1],
                'peak_mb': max(rss_values),
                'average_mb': np.mean(rss_values),
                'growth_mb': rss_values[-1] - rss_values[0],
                'growth_rate_mb_per_sec': (rss_values[-1] - rss_values[0]) / (self.snapshots[-1].timestamp - self.snapshots[0].timestamp)
            }
        }
        
        if heap_values:
            stats['heap_memory'] = {
                'initial_mb': heap_values[0],
                'final_mb': heap_values[-1],
                'peak_mb': max(heap_values),
                'average_mb': np.mean(heap_values),
                'growth_mb': heap_values[-1] - heap_values[0]
            }
        
        # GC statistics
        if len(self.snapshots) > 1:
            initial_gc = self.snapshots[0].gc_collections
            final_gc = self.snapshots[-1].gc_collections
            
            stats['gc_activity'] = {
                gen: final_gc.get(gen, 0) - initial_gc.get(gen, 0)
                for gen in range(3)
            }
        
        # Leak detection results
        leaks = self.get_memory_leaks()
        stats['potential_leaks'] = len(leaks)
        stats['leak_details'] = [
            {
                'file': leak.source_file,
                'line': leak.line_number,
                'growth_rate_mb_per_sec': leak.growth_rate_mb_per_sec,
                'leaked_size_mb': leak.leaked_size_mb
            }
            for leak in leaks[:5]  # Top 5 leaks
        ]
        
        return stats

class MemoryProfiler:
    """High-level memory profiler for benchmark integration"""
    
    def __init__(self):
        self.tracker = MemoryTracker(sampling_interval=0.5)
        self.profiling_active = False
        
    async def profile_function(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Profile memory usage of a function"""
        self.tracker.start_tracking()
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Wait a bit for final measurements
            await asyncio.sleep(1)
            
        finally:
            self.tracker.stop_tracking()
        
        # Get profiling results
        memory_stats = self.tracker.get_memory_stats()
        leaks = self.tracker.get_memory_leaks()
        
        profiling_results = {
            'memory_stats': memory_stats,
            'potential_leaks': len(leaks),
            'leak_summary': [
                {
                    'source': f"{leak.source_file}:{leak.line_number}",
                    'growth_rate_mb_per_sec': leak.growth_rate_mb_per_sec,
                    'total_leaked_mb': leak.leaked_size_mb
                }
                for leak in leaks[:3]
            ],
            'recommendations': self._generate_recommendations(memory_stats, leaks)
        }
        
        return result, profiling_results
    
    def _generate_recommendations(self, stats: Dict[str, Any], leaks: List[MemoryLeak]) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        rss_stats = stats.get('rss_memory', {})
        growth_rate = rss_stats.get('growth_rate_mb_per_sec', 0)
        
        # Memory growth recommendations
        if growth_rate > 1.0:
            recommendations.append("High memory growth detected (>1MB/s). Investigate potential memory leaks.")
        elif growth_rate > 0.1:
            recommendations.append("Moderate memory growth detected. Monitor for sustained patterns.")
        
        # Peak memory recommendations
        peak_mb = rss_stats.get('peak_mb', 0)
        if peak_mb > 1000:  # >1GB
            recommendations.append("High peak memory usage (>1GB). Consider memory optimization.")
        
        # Leak-specific recommendations
        if leaks:
            recommendations.append(f"Found {len(leaks)} potential memory leaks. Review top allocations.")
            
            # Specific leak recommendations
            for leak in leaks[:3]:
                if leak.growth_rate_mb_per_sec > 0.5:
                    recommendations.append(
                        f"Critical leak in {leak.source_file}:{leak.line_number} "
                        f"({leak.growth_rate_mb_per_sec:.2f} MB/s growth)"
                    )
        
        # GC recommendations
        gc_stats = stats.get('gc_activity', {})
        if gc_stats:
            total_collections = sum(gc_stats.values())
            if total_collections > 100:
                recommendations.append(
                    f"High GC activity ({total_collections} collections). "
                    "Consider object pooling or reducing allocations."
                )
        
        # Heap recommendations
        heap_stats = stats.get('heap_memory', {})
        if heap_stats:
            heap_growth = heap_stats.get('growth_mb', 0)
            if heap_growth > 100:
                recommendations.append(
                    f"Large heap growth ({heap_growth:.1f}MB). "
                    "Review data structures and caching strategies."
                )
        
        if not recommendations:
            recommendations.append("Memory usage appears healthy.")
        
        return recommendations

class MemoryBenchmarkIntegration:
    """Integration between memory profiler and performance benchmarks"""
    
    def __init__(self, profiler: MemoryProfiler):
        self.profiler = profiler
    
    async def benchmark_with_memory_profiling(self, benchmark_func, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark with comprehensive memory profiling"""
        
        # Profile the benchmark function
        result, memory_profile = await self.profiler.profile_function(
            benchmark_func, config
        )
        
        # Enhance benchmark result with memory information
        if isinstance(result, dict):
            result['memory_profile'] = memory_profile
        else:
            # If result is not a dict, wrap it
            result = {
                'benchmark_result': result,
                'memory_profile': memory_profile
            }
        
        return result
    
    def compare_memory_profiles(self, mono_profile: Dict[str, Any], 
                               micro_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Compare memory profiles between architectures"""
        
        mono_stats = mono_profile.get('memory_stats', {}).get('rss_memory', {})
        micro_stats = micro_profile.get('memory_stats', {}).get('rss_memory', {})
        
        comparison = {}
        
        # Peak memory comparison
        mono_peak = mono_stats.get('peak_mb', 0)
        micro_peak = micro_stats.get('peak_mb', 0)
        
        if mono_peak > 0:
            peak_change = ((micro_peak - mono_peak) / mono_peak) * 100
            comparison['peak_memory_change_percent'] = round(peak_change, 2)
        
        # Memory growth rate comparison
        mono_growth = mono_stats.get('growth_rate_mb_per_sec', 0)
        micro_growth = micro_stats.get('growth_rate_mb_per_sec', 0)
        
        comparison['growth_rate_comparison'] = {
            'monolithic_mb_per_sec': mono_growth,
            'microservices_mb_per_sec': micro_growth,
            'improvement': micro_growth < mono_growth
        }
        
        # Leak comparison
        mono_leaks = mono_profile.get('potential_leaks', 0)
        micro_leaks = micro_profile.get('potential_leaks', 0)
        
        comparison['leak_comparison'] = {
            'monolithic_leaks': mono_leaks,
            'microservices_leaks': micro_leaks,
            'improvement': micro_leaks < mono_leaks
        }
        
        # Overall memory efficiency score
        efficiency_score = 0
        
        if comparison.get('peak_memory_change_percent', 0) < -10:  # >10% reduction
            efficiency_score += 3
        elif comparison.get('peak_memory_change_percent', 0) < 0:  # Any reduction
            efficiency_score += 1
        
        if comparison['growth_rate_comparison']['improvement']:
            efficiency_score += 2
        
        if comparison['leak_comparison']['improvement']:
            efficiency_score += 2
        
        comparison['efficiency_score'] = efficiency_score
        comparison['efficiency_rating'] = (
            "EXCELLENT" if efficiency_score >= 5 else
            "GOOD" if efficiency_score >= 3 else
            "FAIR" if efficiency_score >= 1 else
            "POOR"
        )
        
        return comparison

# Export main classes
__all__ = [
    'MemoryProfiler',
    'MemoryTracker', 
    'MemorySnapshot',
    'MemoryLeak',
    'MemoryBenchmarkIntegration'
]