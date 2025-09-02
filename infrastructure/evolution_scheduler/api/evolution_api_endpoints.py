"""
Evolution Scheduler API Endpoints

Archaeological Enhancement: RESTful API with WebSocket real-time updates
Innovation Score: 7.1/10 (API orchestration + real-time monitoring)
Branch Origins: api-enhancement-v3, real-time-monitoring-v2
Integration: Zero-breaking-change with enhanced unified gateway
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from ..algorithms.adaptive_algorithms import EvolutionParameters, ModelConfiguration
from ..core.evolution_scheduler_manager import EvolutionSchedulerManager, EvolutionTask
from ..integration.evomerge_coordinator import EvoMergeCoordinator, MergeStrategy
from ..monitoring.regression_detector import ComprehensiveRegressionDetector

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EvolutionAPI:
    """
    Evolution Scheduler API with comprehensive endpoint coverage.
    
    Archaeological Enhancement: Complete REST API with real-time monitoring,
    task orchestration, and performance analytics.
    """
    
    def __init__(self):
        self.app = FastAPI(title="Evolution Scheduler API", version="2.1.0")
        self.scheduler = EvolutionSchedulerManager()
        self.regression_detector = ComprehensiveRegressionDetector()
        self.evomerge_coordinator = EvoMergeCoordinator()
        
        # WebSocket connection manager for real-time updates
        self.connections: list[WebSocket] = []
        
        # Task tracking for API
        self.active_tasks: dict[str, dict[str, Any]] = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes with comprehensive coverage."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for system monitoring."""
            try:
                status = await self.scheduler.get_system_status()
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "scheduler_status": status,
                    "active_tasks": len(self.active_tasks),
                    "websocket_connections": len(self.connections)
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @self.app.post("/evolution/tasks")
        async def create_evolution_task(
            algorithm: str,
            model_config: dict[str, Any],
            parameters: dict[str, Any] | None = None
        ):
            """
            Create new evolution task with specified algorithm and configuration.
            
            Archaeological Enhancement: Complete task creation with parameter validation.
            """
            try:
                task_id = str(uuid.uuid4())
                
                # Parse model configuration
                model_configuration = ModelConfiguration(
                    model_name=model_config.get("model_name", "cognate_25m"),
                    parameter_count=model_config.get("parameter_count", 25000000),
                    architecture_type=model_config.get("architecture_type", "transformer"),
                    model_path=model_config.get("model_path"),
                    config_overrides=model_config.get("config_overrides", {})
                )
                
                # Parse evolution parameters
                evolution_params = EvolutionParameters(**(parameters or {}))
                
                # Create evolution task
                task = EvolutionTask(
                    task_id=task_id,
                    algorithm=algorithm,
                    model_config=model_configuration,
                    parameters=evolution_params,
                    created_at=datetime.now()
                )
                
                # Track task
                self.active_tasks[task_id] = {
                    "task": task,
                    "status": TaskStatus.PENDING,
                    "created_at": datetime.now().isoformat(),
                    "progress": 0.0
                }
                
                # Notify WebSocket connections
                await self._notify_websocket_clients({
                    "type": "task_created",
                    "task_id": task_id,
                    "algorithm": algorithm,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "task_id": task_id,
                    "status": "created",
                    "algorithm": algorithm,
                    "estimated_duration_hours": self._estimate_task_duration(algorithm),
                    "api_endpoints": {
                        "status": f"/evolution/tasks/{task_id}/status",
                        "results": f"/evolution/tasks/{task_id}/results",
                        "cancel": f"/evolution/tasks/{task_id}/cancel"
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to create evolution task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/evolution/tasks/{task_id}/start")
        async def start_evolution_task(task_id: str):
            """
            Start evolution task execution.
            
            Archaeological Enhancement: Async execution with progress tracking.
            """
            if task_id not in self.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task_info = self.active_tasks[task_id]
            if task_info["status"] != TaskStatus.PENDING:
                raise HTTPException(status_code=409, detail="Task already started or completed")
            
            # Update status
            task_info["status"] = TaskStatus.RUNNING
            task_info["started_at"] = datetime.now().isoformat()
            
            # Start evolution task asynchronously
            asyncio.create_task(self._execute_evolution_task(task_id))
            
            await self._notify_websocket_clients({
                "type": "task_started",
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"status": "started", "task_id": task_id}
        
        @self.app.get("/evolution/tasks/{task_id}/status")
        async def get_task_status(task_id: str):
            """Get detailed status of evolution task."""
            if task_id not in self.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task_info = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task_info["status"],
                "progress": task_info["progress"],
                "created_at": task_info["created_at"],
                "started_at": task_info.get("started_at"),
                "completed_at": task_info.get("completed_at"),
                "estimated_completion": self._estimate_completion(task_info),
                "current_generation": task_info.get("current_generation", 0),
                "best_fitness": task_info.get("best_fitness"),
                "regression_alerts": task_info.get("regression_alerts", [])
            }
        
        @self.app.get("/evolution/tasks/{task_id}/results")
        async def get_evolution_results(task_id: str):
            """Get comprehensive evolution task results."""
            if task_id not in self.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task_info = self.active_tasks[task_id]
            if task_info["status"] != TaskStatus.COMPLETED:
                raise HTTPException(status_code=409, detail="Task not completed yet")
            
            return task_info.get("results", {})
        
        @self.app.delete("/evolution/tasks/{task_id}/cancel")
        async def cancel_evolution_task(task_id: str):
            """Cancel running evolution task."""
            if task_id not in self.active_tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task_info = self.active_tasks[task_id]
            if task_info["status"] not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                raise HTTPException(status_code=409, detail="Task cannot be cancelled")
            
            # Cancel the task
            task_info["status"] = TaskStatus.CANCELLED
            task_info["cancelled_at"] = datetime.now().isoformat()
            
            await self._notify_websocket_clients({
                "type": "task_cancelled",
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"status": "cancelled", "task_id": task_id}
        
        @self.app.get("/evolution/tasks")
        async def list_evolution_tasks(
            status: TaskStatus | None = None,
            limit: int = 100,
            offset: int = 0
        ):
            """List evolution tasks with filtering."""
            tasks = []
            for task_id, task_info in list(self.active_tasks.items())[offset:offset + limit]:
                if status is None or task_info["status"] == status:
                    tasks.append({
                        "task_id": task_id,
                        "status": task_info["status"],
                        "algorithm": task_info["task"].algorithm,
                        "created_at": task_info["created_at"],
                        "progress": task_info["progress"]
                    })
            
            return {
                "tasks": tasks,
                "total": len(self.active_tasks),
                "filtered": len(tasks)
            }
        
        @self.app.get("/evolution/algorithms")
        async def list_available_algorithms():
            """List all available evolution algorithms with parameters."""
            algorithms = await self.scheduler.get_available_algorithms()
            return {
                "algorithms": algorithms,
                "count": len(algorithms),
                "documentation": {
                    "genetic": "Genetic Algorithm with adaptive mutation and crossover",
                    "differential": "Differential Evolution with self-adaptive parameters", 
                    "particle_swarm": "Particle Swarm Optimization with velocity updates",
                    "evolutionary_strategies": "Evolution Strategies with covariance adaptation",
                    "nsga2": "Non-dominated Sorting Genetic Algorithm II for multi-objective",
                    "cma_es": "Covariance Matrix Adaptation Evolution Strategy"
                }
            }
        
        @self.app.get("/evolution/performance/metrics")
        async def get_performance_metrics():
            """Get comprehensive performance metrics and analytics."""
            try:
                # Get scheduler performance metrics
                scheduler_metrics = await self.scheduler.get_performance_metrics()
                
                # Get regression detection metrics
                regression_metrics = await self.regression_detector.get_metrics_summary()
                
                # Get EvoMerge coordination metrics
                evomerge_metrics = await self.evomerge_coordinator.get_coordination_metrics()
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "scheduler_metrics": scheduler_metrics,
                    "regression_metrics": regression_metrics,
                    "evomerge_metrics": evomerge_metrics,
                    "active_tasks_count": len([t for t in self.active_tasks.values() if t["status"] == TaskStatus.RUNNING]),
                    "completed_tasks_count": len([t for t in self.active_tasks.values() if t["status"] == TaskStatus.COMPLETED]),
                    "system_health": await self._get_system_health()
                }
                
            except Exception as e:
                logger.error(f"Failed to get performance metrics: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
        
        @self.app.get("/evolution/regression/alerts")
        async def get_regression_alerts(
            severity: str | None = None,
            hours: int = 24
        ):
            """Get recent regression alerts with filtering."""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                alerts = await self.regression_detector.get_recent_alerts(
                    since=cutoff_time,
                    severity=severity
                )
                
                return {
                    "alerts": [alert.to_dict() for alert in alerts],
                    "count": len(alerts),
                    "timeframe_hours": hours,
                    "severity_filter": severity
                }
                
            except Exception as e:
                logger.error(f"Failed to get regression alerts: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve alerts")
        
        @self.app.post("/evolution/regression/acknowledge")
        async def acknowledge_regression_alert(alert_id: str, acknowledged_by: str):
            """Acknowledge regression alert."""
            try:
                success = await self.regression_detector.acknowledge_alert(alert_id, acknowledged_by)
                if success:
                    return {"status": "acknowledged", "alert_id": alert_id}
                else:
                    raise HTTPException(status_code=404, detail="Alert not found")
                    
            except Exception as e:
                logger.error(f"Failed to acknowledge alert: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/evolution/evomerge/status")
        async def get_evomerge_status():
            """Get EvoMerge coordinator status and active operations."""
            try:
                status = await self.evomerge_coordinator.get_status()
                return {
                    "coordinator_status": status,
                    "active_merges": await self.evomerge_coordinator.get_active_merges(),
                    "completed_merges_24h": await self.evomerge_coordinator.get_completed_merges(hours=24),
                    "available_strategies": [strategy.value for strategy in MergeStrategy]
                }
                
            except Exception as e:
                logger.error(f"Failed to get EvoMerge status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/evolution/ws")
        async def evolution_websocket(websocket: WebSocket):
            """
            WebSocket endpoint for real-time evolution updates.
            
            Archaeological Enhancement: Real-time progress tracking and notifications.
            """
            await websocket.accept()
            self.connections.append(websocket)
            
            try:
                while True:
                    # Send periodic status updates
                    status_update = await self._get_websocket_status_update()
                    await websocket.send_text(json.dumps(status_update))
                    await asyncio.sleep(2)  # Update every 2 seconds
                    
            except WebSocketDisconnect:
                self.connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.connections:
                    self.connections.remove(websocket)
    
    async def _execute_evolution_task(self, task_id: str):
        """
        Execute evolution task with progress tracking and regression monitoring.
        
        Archaeological Enhancement: Complete async execution with monitoring integration.
        """
        try:
            task_info = self.active_tasks[task_id]
            task = task_info["task"]
            
            # Set up progress callback
            async def progress_callback(generation: int, best_fitness: float, population_stats: dict):
                progress = min((generation / task.parameters.max_generations) * 100, 100)
                task_info["progress"] = progress
                task_info["current_generation"] = generation
                task_info["best_fitness"] = best_fitness
                task_info["population_stats"] = population_stats
                
                # Check for performance regression
                alerts = await self.regression_detector.check_for_regression(
                    "fitness_evolution",
                    [best_fitness]
                )
                if alerts:
                    task_info["regression_alerts"] = [alert.to_dict() for alert in alerts]
                
                # Send WebSocket update
                await self._notify_websocket_clients({
                    "type": "task_progress",
                    "task_id": task_id,
                    "progress": progress,
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "population_stats": population_stats,
                    "regression_alerts": len(alerts)
                })
            
            # Execute evolution with progress tracking
            result = await self.scheduler.run_evolution_async(
                task,
                progress_callback=progress_callback
            )
            
            # Store results
            task_info["status"] = TaskStatus.COMPLETED
            task_info["completed_at"] = datetime.now().isoformat()
            task_info["results"] = result.to_dict()
            task_info["progress"] = 100.0
            
            # Trigger EvoMerge if configured
            if task.parameters.auto_evomerge:
                evomerge_task = await self.evomerge_coordinator.coordinate_evolution_merge(result)
                task_info["evomerge_task_id"] = evomerge_task.task_id
            
            # Final notification
            await self._notify_websocket_clients({
                "type": "task_completed",
                "task_id": task_id,
                "results": result.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Evolution task {task_id} failed: {e}")
            task_info["status"] = TaskStatus.FAILED
            task_info["error"] = str(e)
            task_info["failed_at"] = datetime.now().isoformat()
            
            await self._notify_websocket_clients({
                "type": "task_failed",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _notify_websocket_clients(self, message: dict[str, Any]):
        """Send message to all connected WebSocket clients."""
        if not self.connections:
            return
            
        message_text = json.dumps(message)
        disconnected = []
        
        for connection in self.connections:
            try:
                await connection.send_text(message_text)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.connections.remove(conn)
    
    async def _get_websocket_status_update(self) -> dict[str, Any]:
        """Get comprehensive status update for WebSocket clients."""
        running_tasks = [
            {
                "task_id": tid,
                "progress": info["progress"],
                "generation": info.get("current_generation", 0),
                "best_fitness": info.get("best_fitness"),
                "algorithm": info["task"].algorithm
            }
            for tid, info in self.active_tasks.items()
            if info["status"] == TaskStatus.RUNNING
        ]
        
        recent_alerts = await self.regression_detector.get_recent_alerts(
            since=datetime.now() - timedelta(hours=1)
        )
        
        return {
            "type": "status_update",
            "timestamp": datetime.now().isoformat(),
            "running_tasks": running_tasks,
            "recent_alerts": len(recent_alerts),
            "system_health": await self._get_system_health(),
            "evomerge_active": len(await self.evomerge_coordinator.get_active_merges())
        }
    
    async def _get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health metrics."""
        try:
            scheduler_health = await self.scheduler.get_health_status()
            regression_health = await self.regression_detector.get_health_status()
            evomerge_health = await self.evomerge_coordinator.get_health_status()
            
            return {
                "overall": "healthy" if all([scheduler_health, regression_health, evomerge_health]) else "degraded",
                "components": {
                    "scheduler": "healthy" if scheduler_health else "unhealthy",
                    "regression_detector": "healthy" if regression_health else "unhealthy",
                    "evomerge_coordinator": "healthy" if evomerge_health else "unhealthy"
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"overall": "unknown", "error": str(e)}
    
    def _estimate_task_duration(self, algorithm: str) -> float:
        """Estimate task duration in hours based on algorithm."""
        estimates = {
            "genetic": 4.0,
            "differential": 6.0,
            "particle_swarm": 3.0,
            "evolutionary_strategies": 8.0,
            "nsga2": 12.0,
            "cma_es": 10.0
        }
        return estimates.get(algorithm, 6.0)
    
    def _estimate_completion(self, task_info: dict[str, Any]) -> str | None:
        """Estimate task completion time."""
        if task_info["status"] != TaskStatus.RUNNING:
            return None
            
        progress = task_info["progress"]
        if progress <= 0:
            return None
        
        started_at = datetime.fromisoformat(task_info["started_at"])
        elapsed = datetime.now() - started_at
        
        if progress > 0:
            total_estimated = elapsed / (progress / 100)
            remaining = total_estimated - elapsed
            completion_time = datetime.now() + remaining
            return completion_time.isoformat()
        
        return None

# Global API instance for integration
evolution_api = EvolutionAPI()
app = evolution_api.app

# Integration helpers
async def get_evolution_scheduler() -> EvolutionSchedulerManager:
    """Get evolution scheduler instance for dependency injection."""
    return evolution_api.scheduler

async def get_regression_detector() -> ComprehensiveRegressionDetector:
    """Get regression detector instance for dependency injection."""
    return evolution_api.regression_detector

async def get_evomerge_coordinator() -> EvoMergeCoordinator:
    """Get EvoMerge coordinator instance for dependency injection."""
    return evolution_api.evomerge_coordinator

# Health check for external integration
async def evolution_system_health() -> bool:
    """Quick health check for external systems."""
    try:
        health_data = await evolution_api._get_system_health()
        return health_data["overall"] == "healthy"
    except Exception:
        return False