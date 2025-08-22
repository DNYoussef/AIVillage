"""
RAG Fog Bridge

Distributes RAG (Retrieval-Augmented Generation) operations across fog nodes for scalability.
Enables distributed processing of HyperRAG components while maintaining consistency
and leveraging existing fog computing infrastructure.

Key Components:
- FogRAGCoordinator: Orchestrate distributed queries across fog nodes
- FogVectorSearch: Distribute vector search operations
- FogKnowledgeGraph: Distribute graph traversal and reasoning
- FogHippocampusRAG: Distribute episodic memory operations
- FogCognitiveNexus: Distribute multi-perspective analysis

This bridges the HyperRAG system with fog computing to enable large-scale
distributed knowledge processing while maintaining the unified interface.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import aiohttp

# Import from existing RAG infrastructure
from packages.rag.core.hyper_rag import HyperRAG, QueryMode

logger = logging.getLogger(__name__)


class FogRAGStrategy(str, Enum):
    """Strategy for distributing RAG operations"""

    LOCAL_ONLY = "local_only"  # Execute locally only
    FOG_FIRST = "fog_first"  # Prefer fog execution
    HYBRID = "hybrid"  # Mix local and fog execution
    ADAPTIVE = "adaptive"  # Adapt based on load and complexity


class FogNodeCapability(str, Enum):
    """Fog node capabilities for RAG operations"""

    VECTOR_SEARCH = "vector_search"  # Vector similarity search
    GRAPH_TRAVERSAL = "graph_traversal"  # Knowledge graph operations
    EMBEDDINGS = "embeddings"  # Embedding generation
    REASONING = "reasoning"  # Logical reasoning and inference
    MEMORY_OPS = "memory_ops"  # Episodic memory operations


@dataclass
class FogNodeInfo:
    """Information about fog node capabilities and resources"""

    node_id: str
    endpoint: str
    capabilities: list[FogNodeCapability]

    # Resource information
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu_available: bool = False

    # Performance metrics
    avg_latency_ms: float = 0.0
    current_load: float = 0.0
    availability: float = 1.0

    # Network information
    region: str = "unknown"
    network_quality: str = "good"  # good, fair, poor

    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class FogRAGTask:
    """Distributed RAG task for fog execution"""

    task_id: str
    task_type: str  # vector_search, graph_query, embedding_gen, etc.
    query: str
    parameters: dict[str, Any]

    # Fog execution details
    target_nodes: list[str] = field(default_factory=list)
    strategy: FogRAGStrategy = FogRAGStrategy.ADAPTIVE
    priority: int = 5  # 1-10 priority scale

    # Execution tracking
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class FogRAGCoordinator:
    """
    Coordinates distributed RAG operations across fog nodes

    Manages node discovery, task distribution, result aggregation,
    and fallback to local execution when needed.
    """

    def __init__(
        self,
        fog_gateway_url: str = "http://localhost:8080",
        local_rag: HyperRAG | None = None,
        default_strategy: FogRAGStrategy = FogRAGStrategy.ADAPTIVE,
    ):
        self.fog_gateway_url = fog_gateway_url.rstrip("/")
        self.local_rag = local_rag
        self.default_strategy = default_strategy

        # Node management
        self.available_nodes: dict[str, FogNodeInfo] = {}
        self.active_tasks: dict[str, FogRAGTask] = {}

        # Performance tracking
        self.fog_success_rate = 0.95
        self.avg_fog_latency = 0.0
        self.fog_cost_benefit = 1.2  # Cost/benefit ratio vs local

        # Background tasks
        self._node_discovery_task: asyncio.Task | None = None
        self._health_monitor_task: asyncio.Task | None = None

    async def initialize(self) -> bool:
        """Initialize fog RAG coordinator"""

        try:
            # Start background tasks
            self._node_discovery_task = asyncio.create_task(self._discover_fog_nodes())
            self._health_monitor_task = asyncio.create_task(self._monitor_node_health())

            # Initial node discovery
            await self._discover_fog_nodes()

            logger.info(f"FogRAGCoordinator initialized with {len(self.available_nodes)} nodes")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FogRAGCoordinator: {e}")
            return False

    async def distributed_query(
        self,
        query: str,
        mode: QueryMode = QueryMode.BALANCED,
        strategy: FogRAGStrategy | None = None,
        max_results: int = 10,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """
        Execute distributed RAG query across fog nodes

        Args:
            query: The query string
            mode: Query processing mode (fast, balanced, comprehensive)
            strategy: Distribution strategy
            max_results: Maximum results to return
            include_sources: Include source information

        Returns:
            Aggregated results from distributed processing
        """

        strategy = strategy or self.default_strategy
        task_id = str(uuid4())

        logger.info(f"Starting distributed RAG query {task_id}: {query[:50]}...")

        try:
            # Determine execution strategy
            if strategy == FogRAGStrategy.LOCAL_ONLY or not self.available_nodes:
                return await self._execute_local_query(query, mode, max_results, include_sources)

            # Create distributed tasks based on query complexity
            tasks = await self._plan_distributed_execution(query, mode, task_id)

            if not tasks:
                # Fallback to local execution
                logger.info(f"No suitable fog nodes for query {task_id}, falling back to local")
                return await self._execute_local_query(query, mode, max_results, include_sources)

            # Execute tasks across fog nodes
            results = await self._execute_distributed_tasks(tasks)

            # Aggregate and rank results
            aggregated = await self._aggregate_results(results, max_results, include_sources)

            # Update performance metrics
            self._update_performance_metrics(tasks, results)

            logger.info(f"Completed distributed RAG query {task_id} with {len(aggregated.get('results', []))} results")

            return aggregated

        except Exception as e:
            logger.error(f"Distributed query {task_id} failed: {e}")

            # Fallback to local execution
            if self.local_rag:
                logger.info(f"Falling back to local execution for query {task_id}")
                return await self._execute_local_query(query, mode, max_results, include_sources)
            else:
                return {
                    "status": "error",
                    "message": f"Distributed query failed and no local fallback: {str(e)}",
                    "results": [],
                    "query": query,
                }

    async def _discover_fog_nodes(self) -> None:
        """Discover available fog nodes with RAG capabilities"""

        try:
            async with aiohttp.ClientSession() as session:
                # Query fog gateway for available nodes
                async with session.get(
                    f"{self.fog_gateway_url}/v1/fog/nodes", params={"capabilities": "rag,vector_search,reasoning"}
                ) as response:
                    if response.status == 200:
                        nodes_data = await response.json()

                        for node_data in nodes_data.get("nodes", []):
                            node_info = FogNodeInfo(
                                node_id=node_data["node_id"],
                                endpoint=node_data["endpoint"],
                                capabilities=[
                                    FogNodeCapability(cap)
                                    for cap in node_data.get("capabilities", [])
                                    if cap in [c.value for c in FogNodeCapability]
                                ],
                                cpu_cores=node_data.get("resources", {}).get("cpu_cores", 0.0),
                                memory_gb=node_data.get("resources", {}).get("memory_gb", 0.0),
                                gpu_available=node_data.get("resources", {}).get("gpu_available", False),
                                avg_latency_ms=node_data.get("metrics", {}).get("avg_latency_ms", 0.0),
                                current_load=node_data.get("metrics", {}).get("current_load", 0.0),
                                availability=node_data.get("metrics", {}).get("availability", 1.0),
                                region=node_data.get("region", "unknown"),
                                network_quality=node_data.get("network_quality", "good"),
                            )

                            self.available_nodes[node_info.node_id] = node_info

                        logger.debug(f"Discovered {len(self.available_nodes)} RAG-capable fog nodes")

                    else:
                        logger.warning(f"Failed to discover fog nodes: {response.status}")

        except Exception as e:
            logger.error(f"Error discovering fog nodes: {e}")

    async def _monitor_node_health(self) -> None:
        """Monitor health and performance of fog nodes"""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Remove stale nodes
                current_time = datetime.now(UTC)
                stale_nodes = [
                    node_id
                    for node_id, node_info in self.available_nodes.items()
                    if (current_time - node_info.last_seen).total_seconds() > 300  # 5 minutes
                ]

                for node_id in stale_nodes:
                    del self.available_nodes[node_id]
                    logger.info(f"Removed stale fog node: {node_id}")

                # Update node health metrics
                await self._update_node_metrics()

            except Exception as e:
                logger.error(f"Error monitoring node health: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _plan_distributed_execution(self, query: str, mode: QueryMode, task_id: str) -> list[FogRAGTask]:
        """Plan how to distribute query execution across fog nodes"""

        tasks = []

        # Analyze query complexity and requirements
        query_analysis = await self._analyze_query_requirements(query, mode)

        # Vector search task
        if query_analysis["needs_vector_search"]:
            vector_nodes = [
                node_id
                for node_id, node_info in self.available_nodes.items()
                if FogNodeCapability.VECTOR_SEARCH in node_info.capabilities and node_info.current_load < 0.8
            ]

            if vector_nodes:
                tasks.append(
                    FogRAGTask(
                        task_id=f"{task_id}_vector",
                        task_type="vector_search",
                        query=query,
                        parameters={
                            "mode": mode.value,
                            "max_results": query_analysis["vector_results_needed"],
                            "similarity_threshold": 0.7,
                        },
                        target_nodes=vector_nodes[:2],  # Use top 2 nodes
                        priority=8,
                    )
                )

        # Knowledge graph traversal task
        if query_analysis["needs_graph_reasoning"]:
            graph_nodes = [
                node_id
                for node_id, node_info in self.available_nodes.items()
                if FogNodeCapability.GRAPH_TRAVERSAL in node_info.capabilities and node_info.current_load < 0.8
            ]

            if graph_nodes:
                tasks.append(
                    FogRAGTask(
                        task_id=f"{task_id}_graph",
                        task_type="graph_traversal",
                        query=query,
                        parameters={
                            "mode": mode.value,
                            "max_depth": query_analysis["graph_depth"],
                            "include_reasoning": True,
                        },
                        target_nodes=graph_nodes[:1],  # Use best node
                        priority=7,
                    )
                )

        # Episodic memory search task
        if query_analysis["needs_episodic_memory"]:
            memory_nodes = [
                node_id
                for node_id, node_info in self.available_nodes.items()
                if FogNodeCapability.MEMORY_OPS in node_info.capabilities and node_info.current_load < 0.8
            ]

            if memory_nodes:
                tasks.append(
                    FogRAGTask(
                        task_id=f"{task_id}_memory",
                        task_type="episodic_search",
                        query=query,
                        parameters={"mode": mode.value, "time_decay": True, "importance_threshold": 0.3},
                        target_nodes=memory_nodes[:1],
                        priority=6,
                    )
                )

        # Complex reasoning task
        if query_analysis["needs_complex_reasoning"] and mode in [QueryMode.COMPREHENSIVE, QueryMode.ANALYTICAL]:
            reasoning_nodes = [
                node_id
                for node_id, node_info in self.available_nodes.items()
                if FogNodeCapability.REASONING in node_info.capabilities
                and node_info.memory_gb >= 4.0  # Reasoning needs more memory
                and node_info.current_load < 0.7
            ]

            if reasoning_nodes:
                tasks.append(
                    FogRAGTask(
                        task_id=f"{task_id}_reasoning",
                        task_type="complex_reasoning",
                        query=query,
                        parameters={"mode": mode.value, "multi_perspective": True, "contradiction_detection": True},
                        target_nodes=reasoning_nodes[:1],
                        priority=9,
                    )
                )

        return tasks

    async def _analyze_query_requirements(self, query: str, mode: QueryMode) -> dict[str, Any]:
        """Analyze query to determine computational requirements"""

        # Simple heuristic analysis - production would use more sophisticated NLP
        query_lower = query.lower()

        # Check for different types of operations needed
        needs_vector = any(
            word in query_lower for word in ["similar", "related", "like", "find", "search", "documents"]
        )

        needs_graph = any(
            word in query_lower for word in ["relationship", "connection", "because", "why", "how", "explain", "reason"]
        )

        needs_memory = any(
            word in query_lower for word in ["remember", "recall", "experience", "happened", "before", "previously"]
        )

        needs_reasoning = any(
            word in query_lower for word in ["analyze", "compare", "evaluate", "assess", "complex", "contradiction"]
        ) or mode in [QueryMode.COMPREHENSIVE, QueryMode.ANALYTICAL]

        # Estimate computational requirements
        query_complexity = len(query.split())
        vector_results = min(50, max(10, query_complexity * 2))
        graph_depth = 3 if mode == QueryMode.COMPREHENSIVE else 2

        return {
            "needs_vector_search": needs_vector,
            "needs_graph_reasoning": needs_graph,
            "needs_episodic_memory": needs_memory,
            "needs_complex_reasoning": needs_reasoning,
            "query_complexity": query_complexity,
            "vector_results_needed": vector_results,
            "graph_depth": graph_depth,
        }

    async def _execute_distributed_tasks(self, tasks: list[FogRAGTask]) -> dict[str, Any]:
        """Execute tasks across fog nodes and collect results"""

        results = {}

        # Execute tasks in parallel
        async def execute_task(task: FogRAGTask) -> tuple[str, dict[str, Any]]:
            try:
                task.status = "running"
                task.started_at = datetime.now(UTC)

                # Select best node for this task
                target_node = await self._select_best_node(task)
                if not target_node:
                    raise Exception("No suitable fog node available")

                # Submit job to fog gateway
                job_spec = {
                    "namespace": "rag-distributed",
                    "runtime": "wasi",  # Use WASI for RAG operations
                    "image": f"rag-{task.task_type}:latest",
                    "args": [json.dumps(task.parameters)],
                    "env": {"QUERY": task.query, "TASK_TYPE": task.task_type, "TASK_ID": task.task_id},
                    "resources": {
                        "cpu_cores": 2.0 if task.task_type == "complex_reasoning" else 1.0,
                        "memory_gb": 4.0 if task.task_type == "complex_reasoning" else 2.0,
                        "max_duration_s": 120,
                        "network_egress": True,  # Needed for external knowledge sources
                    },
                    "metadata": {
                        "task_type": "rag_distributed",
                        "rag_task_id": task.task_id,
                        "target_node": target_node,
                    },
                }

                async with aiohttp.ClientSession() as session:
                    # Submit job
                    async with session.post(f"{self.fog_gateway_url}/v1/fog/jobs", json=job_spec) as response:
                        if response.status == 201:
                            job_data = await response.json()
                            job_id = job_data["job_id"]

                            # Wait for completion
                            result = await self._wait_for_job_completion(job_id, timeout=180)

                            task.status = "completed"
                            task.completed_at = datetime.now(UTC)
                            task.results = result

                            return task.task_id, result
                        else:
                            error_text = await response.text()
                            raise Exception(f"Job submission failed: {error_text}")

            except Exception as e:
                task.status = "failed"
                task.errors.append(str(e))
                logger.error(f"Task {task.task_id} failed: {e}")
                return task.task_id, {"status": "error", "message": str(e)}

        # Execute all tasks concurrently
        task_results = await asyncio.gather(*[execute_task(task) for task in tasks], return_exceptions=True)

        # Collect results
        for task_result in task_results:
            if isinstance(task_result, Exception):
                logger.error(f"Task execution exception: {task_result}")
            else:
                task_id, result = task_result
                results[task_id] = result

        return results

    async def _select_best_node(self, task: FogRAGTask) -> str | None:
        """Select the best fog node for a given task"""

        candidate_nodes = []

        for node_id in task.target_nodes:
            node_info = self.available_nodes.get(node_id)
            if not node_info:
                continue

            # Calculate node score based on multiple factors
            score = 0.0

            # Capability match
            required_cap = None
            if task.task_type == "vector_search":
                required_cap = FogNodeCapability.VECTOR_SEARCH
            elif task.task_type == "graph_traversal":
                required_cap = FogNodeCapability.GRAPH_TRAVERSAL
            elif task.task_type == "episodic_search":
                required_cap = FogNodeCapability.MEMORY_OPS
            elif task.task_type == "complex_reasoning":
                required_cap = FogNodeCapability.REASONING

            if required_cap and required_cap in node_info.capabilities:
                score += 40.0
            else:
                continue  # Skip nodes without required capability

            # Resource availability (lower load is better)
            score += (1.0 - node_info.current_load) * 20.0

            # Performance metrics (lower latency is better)
            if node_info.avg_latency_ms > 0:
                score += max(0, 20.0 - (node_info.avg_latency_ms / 100.0))
            else:
                score += 15.0  # Default for unknown latency

            # Availability
            score += node_info.availability * 10.0

            # Network quality
            if node_info.network_quality == "good":
                score += 5.0
            elif node_info.network_quality == "fair":
                score += 2.0

            # GPU availability for certain tasks
            if task.task_type in ["complex_reasoning", "vector_search"] and node_info.gpu_available:
                score += 5.0

            candidate_nodes.append((node_id, score))

        if not candidate_nodes:
            return None

        # Sort by score and return best node
        candidate_nodes.sort(key=lambda x: x[1], reverse=True)
        return candidate_nodes[0][0]

    async def _wait_for_job_completion(self, job_id: str, timeout: int = 180) -> dict[str, Any]:
        """Wait for fog job to complete and return results"""

        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Check job status
                    async with session.get(f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/status") as response:
                        if response.status == 200:
                            status_data = await response.json()
                            job_status = status_data.get("status")

                            if job_status == "completed":
                                # Get job results
                                async with session.get(
                                    f"{self.fog_gateway_url}/v1/fog/jobs/{job_id}/result"
                                ) as result_response:
                                    if result_response.status == 200:
                                        result_data = await result_response.json()

                                        # Parse job output as JSON
                                        try:
                                            output = json.loads(result_data.get("stdout", "{}"))
                                            return output
                                        except json.JSONDecodeError:
                                            return {
                                                "status": "completed",
                                                "raw_output": result_data.get("stdout", ""),
                                                "stderr": result_data.get("stderr", ""),
                                            }
                                    else:
                                        return {"status": "error", "message": "Failed to get job result"}

                            elif job_status in ["failed", "cancelled", "timeout"]:
                                return {"status": "error", "message": f"Job {job_status}", "job_status": job_status}

                            # Job still running, check timeout
                            if asyncio.get_event_loop().time() - start_time > timeout:
                                return {"status": "error", "message": f"Job timed out after {timeout}s"}

                            # Wait before next check
                            await asyncio.sleep(2)

                        else:
                            return {"status": "error", "message": f"Failed to check job status: {response.status}"}

            except Exception as e:
                return {"status": "error", "message": f"Error waiting for job: {str(e)}"}

    async def _aggregate_results(
        self, task_results: dict[str, Any], max_results: int, include_sources: bool
    ) -> dict[str, Any]:
        """Aggregate results from distributed tasks"""

        all_results = []
        sources = []
        metadata = {
            "distributed_execution": True,
            "task_count": len(task_results),
            "fog_nodes_used": [],
            "execution_summary": {},
        }

        # Process results from each task
        for task_id, result in task_results.items():
            if result.get("status") == "error":
                metadata["execution_summary"][task_id] = {
                    "status": "failed",
                    "error": result.get("message", "Unknown error"),
                }
                continue

            # Extract results based on task type
            task_type = task_id.split("_")[-1]  # Extract type from task_id

            if "results" in result:
                task_results_list = result["results"]

                # Add task type metadata to each result
                for res in task_results_list:
                    if isinstance(res, dict):
                        res["source_task"] = task_type
                        res["fog_executed"] = True

                all_results.extend(task_results_list)

            if include_sources and "sources" in result:
                sources.extend(result["sources"])

            # Update metadata
            metadata["execution_summary"][task_id] = {
                "status": "completed",
                "result_count": len(result.get("results", [])),
                "execution_time_ms": result.get("execution_time_ms", 0),
            }

            if "fog_node" in result:
                metadata["fog_nodes_used"].append(result["fog_node"])

        # Sort and limit results
        # Simple scoring based on relevance and source task priority
        def result_score(result):
            if not isinstance(result, dict):
                return 0.0

            base_score = result.get("score", 0.0)

            # Boost scores based on task type priority
            task_type = result.get("source_task", "")
            if task_type == "reasoning":
                base_score *= 1.2
            elif task_type == "graph":
                base_score *= 1.1
            elif task_type == "vector":
                base_score *= 1.0
            elif task_type == "memory":
                base_score *= 0.9

            return base_score

        # Sort by score and take top results
        if all_results:
            all_results.sort(key=result_score, reverse=True)
            all_results = all_results[:max_results]

        return {
            "status": "success",
            "results": all_results,
            "sources": sources if include_sources else None,
            "metadata": metadata,
            "query_mode": "distributed",
            "total_results": len(all_results),
        }

    async def _execute_local_query(
        self, query: str, mode: QueryMode, max_results: int, include_sources: bool
    ) -> dict[str, Any]:
        """Execute query locally as fallback"""

        if not self.local_rag:
            return {"status": "error", "message": "No local RAG system available for fallback", "results": []}

        try:
            result = await self.local_rag.query(
                query=query, mode=mode, max_results=max_results, include_sources=include_sources
            )

            # Add metadata indicating local execution
            if isinstance(result, dict):
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["distributed_execution"] = False
                result["metadata"]["fallback_execution"] = True

            return result

        except Exception as e:
            logger.error(f"Local query execution failed: {e}")
            return {"status": "error", "message": f"Local query execution failed: {str(e)}", "results": []}

    async def _update_node_metrics(self) -> None:
        """Update performance metrics for fog nodes"""

        for node_id, node_info in self.available_nodes.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{node_info.endpoint}/health", timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            health_data = await response.json()

                            # Update metrics
                            node_info.current_load = health_data.get("cpu_usage", 0.0)
                            node_info.avg_latency_ms = health_data.get("avg_latency_ms", 0.0)
                            node_info.availability = health_data.get("availability", 1.0)
                            node_info.last_seen = datetime.now(UTC)
                        else:
                            # Reduce availability for unhealthy nodes
                            node_info.availability *= 0.9

            except Exception as e:
                logger.debug(f"Failed to update metrics for node {node_id}: {e}")
                node_info.availability *= 0.8  # Reduce availability on error

    def _update_performance_metrics(self, tasks: list[FogRAGTask], results: dict[str, Any]) -> None:
        """Update overall fog execution performance metrics"""

        successful_tasks = sum(1 for task in tasks if task.status == "completed")
        total_tasks = len(tasks)

        if total_tasks > 0:
            # Update success rate with exponential moving average
            current_success_rate = successful_tasks / total_tasks
            self.fog_success_rate = (0.9 * self.fog_success_rate) + (0.1 * current_success_rate)

        # Calculate average latency
        completed_tasks = [
            task for task in tasks if task.status == "completed" and task.started_at and task.completed_at
        ]
        if completed_tasks:
            avg_duration = sum(
                (task.completed_at - task.started_at).total_seconds() * 1000 for task in completed_tasks
            ) / len(completed_tasks)

            self.avg_fog_latency = (0.9 * self.avg_fog_latency) + (0.1 * avg_duration)

    async def shutdown(self) -> None:
        """Shutdown fog RAG coordinator"""

        if self._node_discovery_task:
            self._node_discovery_task.cancel()
            try:
                await self._node_discovery_task
            except asyncio.CancelledError:
                pass

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("FogRAGCoordinator shutdown completed")


# Convenience functions for integration
async def create_fog_rag_coordinator(
    fog_gateway_url: str = "http://localhost:8080", local_rag: HyperRAG | None = None
) -> FogRAGCoordinator:
    """Create and initialize fog RAG coordinator"""

    coordinator = FogRAGCoordinator(fog_gateway_url, local_rag)

    if await coordinator.initialize():
        return coordinator
    else:
        raise Exception("Failed to initialize FogRAGCoordinator")


# Export main classes
__all__ = [
    "FogRAGCoordinator",
    "FogRAGStrategy",
    "FogNodeCapability",
    "FogNodeInfo",
    "FogRAGTask",
    "create_fog_rag_coordinator",
]
