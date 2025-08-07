# Agent Implementation Guide for AIVillage

## Executive Summary

The AIVillage agent framework currently has 8 of 18 specialized agents implemented. This guide provides comprehensive patterns, interfaces, and strategies for implementing the remaining 10 agents based on analysis of the existing architecture.

## Current State Analysis

### Implemented Agents (8)
1. **King** - Orchestrator and coordinator (fully implemented)
2. **Sage** - Research and analysis specialist (fully implemented)
3. **Magi** - Strategic planning and decision-making (fully implemented)
4. **Sword** - Offensive capabilities (stub only)
5. **Shield** - Defensive capabilities (stub only)
6. **Logger** - System logging and monitoring
7. **Profiler** - Performance profiling
8. **Builder** - Construction and creation tasks

### Missing Agents (10)
1. **Scribe** - Documentation and record-keeping
2. **Herald** - Communication and message routing
3. **Curator** - Knowledge organization and curation
4. **Navigator** - Path finding and exploration
5. **Alchemist** - Transformation and synthesis
6. **Guardian** - Protection and validation
7. **Chronicler** - Historical tracking and versioning
8. **Artificer** - Tool creation and enhancement
9. **Emissary** - External system integration
10. **Steward** - Resource management and allocation

## Core Agent Architecture

### Base Agent Pattern

```python
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import asyncio
import time
import uuid

from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from core.communication import Message, MessageType, Priority, StandardCommunicationProtocol
from core.error_handling import with_error_handling, get_component_logger
from rag_system.core.pipeline import EnhancedRAGPipeline
from agents.utils.task import Task as LangroidTask

@dataclass
class AgentSpecialization:
    """Agent specialization configuration."""
    role: str
    primary_capabilities: List[str]
    secondary_capabilities: List[str]
    performance_metrics: Dict[str, float]
    resource_requirements: Dict[str, Any]
    communication_protocols: List[str] = field(default_factory=list)

class BaseSpecializedAgent(UnifiedBaseAgent):
    """Base class for all specialized agents."""

    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        specialization: AgentSpecialization,
        knowledge_tracker: Optional[Any] = None
    ):
        super().__init__(config, communication_protocol, knowledge_tracker)
        self.specialization = specialization
        self.agent_id = str(uuid.uuid4())
        self.state = AgentState()
        self.metrics = AgentMetrics()
        self.logger = get_component_logger(
            f"{self.__class__.__name__}",
            {"agent_id": self.agent_id, "role": specialization.role}
        )

        # Initialize specialized components
        self._initialize_specialized_components()

        # Register capabilities
        self._register_capabilities()

        # Set up communication channels
        self._setup_communication()

    def _initialize_specialized_components(self):
        """Override in specialized agents to initialize components."""
        pass

    def _register_capabilities(self):
        """Register agent capabilities with the system."""
        for capability in self.specialization.primary_capabilities:
            self.add_tool(capability, getattr(self, f"handle_{capability}", self._default_handler))

    def _setup_communication(self):
        """Set up communication channels and protocols."""
        for protocol in self.specialization.communication_protocols:
            self.communication_protocol.subscribe(
                f"{self.name}_{protocol}",
                self.handle_protocol_message
            )

    @with_error_handling(retries=2)
    async def execute_task(self, task: LangroidTask) -> Dict[str, Any]:
        """Execute a task with specialization-specific logic."""
        start_time = time.time()

        # Pre-process task
        task = await self._preprocess_task(task)

        # Route to specialized handler
        handler = self._get_task_handler(task.type)
        result = await handler(task)

        # Post-process result
        result = await self._postprocess_result(result)

        # Update metrics
        self.metrics.record_task_execution(
            task_type=task.type,
            duration=time.time() - start_time,
            success=result.get("success", False)
        )

        return result

    async def _preprocess_task(self, task: LangroidTask) -> LangroidTask:
        """Preprocess task based on specialization."""
        return task

    async def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess result based on specialization."""
        return result

    def _get_task_handler(self, task_type: str):
        """Get specialized task handler."""
        handlers = {
            "default": self._handle_default_task,
            # Add specialized handlers in subclasses
        }
        return handlers.get(task_type, handlers["default"])

    async def _handle_default_task(self, task: LangroidTask) -> Dict[str, Any]:
        """Default task handler."""
        return {
            "success": True,
            "result": f"Task handled by {self.name}",
            "agent_id": self.agent_id
        }
```

### State Management Pattern

```python
@dataclass
class AgentState:
    """Agent state management."""

    status: str = "idle"  # idle, busy, error, evolving
    current_task: Optional[str] = None
    task_queue: List[str] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    # Evolution state
    generation: int = 0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)

    def update_status(self, new_status: str):
        """Update agent status with validation."""
        valid_statuses = ["idle", "busy", "error", "evolving", "suspended"]
        if new_status in valid_statuses:
            self.status = new_status
        else:
            raise ValueError(f"Invalid status: {new_status}")

    def checkpoint(self) -> Dict[str, Any]:
        """Create state checkpoint for recovery."""
        return {
            "status": self.status,
            "current_task": self.current_task,
            "task_queue": self.task_queue.copy(),
            "memory": self.memory.copy(),
            "context": self.context.copy(),
            "generation": self.generation,
            "timestamp": time.time()
        }

    def restore(self, checkpoint: Dict[str, Any]):
        """Restore from checkpoint."""
        self.status = checkpoint.get("status", "idle")
        self.current_task = checkpoint.get("current_task")
        self.task_queue = checkpoint.get("task_queue", [])
        self.memory = checkpoint.get("memory", {})
        self.context = checkpoint.get("context", {})
        self.generation = checkpoint.get("generation", 0)
```

### Communication Protocol

```python
class AgentCommunicationProtocol:
    """Inter-agent communication protocol."""

    def __init__(self, agent_id: str, protocol: StandardCommunicationProtocol):
        self.agent_id = agent_id
        self.protocol = protocol
        self.message_handlers = {}
        self.pending_responses = {}

    async def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        content: Any,
        priority: Priority = Priority.NORMAL
    ) -> str:
        """Send message to another agent."""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            type=message_type,
            content=content,
            priority=priority,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )

        await self.protocol.send(message)
        return message.message_id

    async def broadcast(
        self,
        message_type: MessageType,
        content: Any,
        agent_filter: Optional[List[str]] = None
    ):
        """Broadcast message to multiple agents."""
        recipients = agent_filter or self.protocol.get_all_agents()

        tasks = [
            self.send_message(recipient, message_type, content)
            for recipient in recipients
            if recipient != self.agent_id
        ]

        await asyncio.gather(*tasks)

    async def request_response(
        self,
        recipient: str,
        request: Any,
        timeout: float = 30.0
    ) -> Any:
        """Send request and wait for response."""
        message_id = await self.send_message(
            recipient,
            MessageType.REQUEST,
            request
        )

        # Wait for response
        response_future = asyncio.Future()
        self.pending_responses[message_id] = response_future

        try:
            response = await asyncio.wait_for(response_future, timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_responses[message_id]
            raise TimeoutError(f"No response from {recipient} within {timeout}s")

    def register_handler(self, message_type: MessageType, handler):
        """Register message handler."""
        self.message_handlers[message_type] = handler
```

### Resource Management

```python
class AgentResourceManager:
    """Manage agent resource allocation and constraints."""

    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        self.current_usage = {
            "memory_mb": 0,
            "cpu_percent": 0,
            "gpu_memory_mb": 0,
            "network_bandwidth_mbps": 0
        }
        self.allocation_history = []

    def check_resource_availability(self, required: Dict[str, float]) -> bool:
        """Check if required resources are available."""
        for resource, amount in required.items():
            if resource in self.constraints:
                available = self.constraints[resource] - self.current_usage.get(resource, 0)
                if available < amount:
                    return False
        return True

    def allocate_resources(self, resources: Dict[str, float]) -> bool:
        """Allocate resources if available."""
        if not self.check_resource_availability(resources):
            return False

        for resource, amount in resources.items():
            self.current_usage[resource] = self.current_usage.get(resource, 0) + amount

        self.allocation_history.append({
            "timestamp": time.time(),
            "allocated": resources,
            "usage_after": self.current_usage.copy()
        })

        return True

    def release_resources(self, resources: Dict[str, float]):
        """Release allocated resources."""
        for resource, amount in resources.items():
            if resource in self.current_usage:
                self.current_usage[resource] = max(0, self.current_usage[resource] - amount)

    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        utilization = {}
        for resource, limit in self.constraints.items():
            if limit > 0:
                usage = self.current_usage.get(resource, 0)
                utilization[resource] = (usage / limit) * 100
        return utilization
```

## Implementation Templates for Missing Agents

### 1. Scribe Agent

```python
class ScribeAgent(BaseSpecializedAgent):
    """Documentation and record-keeping specialist."""

    def _initialize_specialized_components(self):
        self.document_store = DocumentStore()
        self.template_engine = TemplateEngine()
        self.version_control = VersionControl()

    async def handle_document_creation(self, task: LangroidTask) -> Dict[str, Any]:
        """Create documentation from task context."""
        template = self.template_engine.get_template(task.metadata.get("doc_type"))
        content = await self.generate_content(task.content, template)

        doc_id = self.document_store.save(content, {
            "author": self.agent_id,
            "timestamp": time.time(),
            "type": task.metadata.get("doc_type"),
            "version": "1.0"
        })

        return {
            "success": True,
            "document_id": doc_id,
            "location": self.document_store.get_path(doc_id)
        }

    async def handle_record_keeping(self, records: List[Dict]) -> Dict[str, Any]:
        """Maintain system records."""
        processed = []
        for record in records:
            record_id = self.document_store.index_record(record)
            self.version_control.track_change(record_id, record)
            processed.append(record_id)

        return {
            "success": True,
            "records_processed": len(processed),
            "record_ids": processed
        }
```

### 2. Herald Agent

```python
class HeraldAgent(BaseSpecializedAgent):
    """Communication and message routing specialist."""

    def _initialize_specialized_components(self):
        self.routing_table = RoutingTable()
        self.message_queue = PriorityQueue()
        self.broadcast_channels = {}

    async def handle_message_routing(self, message: Message) -> Dict[str, Any]:
        """Route messages based on content and priority."""
        # Determine optimal route
        route = self.routing_table.find_route(
            message.recipient,
            message.type,
            message.priority
        )

        # Queue for delivery
        await self.message_queue.put((message.priority, message, route))

        # Process queue
        delivered = await self._process_message_queue()

        return {
            "success": True,
            "messages_routed": delivered,
            "queue_size": self.message_queue.qsize()
        }

    async def handle_broadcast(self, content: Any, channel: str) -> Dict[str, Any]:
        """Broadcast announcements to subscribers."""
        if channel not in self.broadcast_channels:
            self.broadcast_channels[channel] = []

        subscribers = self.broadcast_channels[channel]

        tasks = [
            self.send_message(sub, MessageType.BROADCAST, content)
            for sub in subscribers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": True,
            "recipients": len(subscribers),
            "failures": sum(1 for r in results if isinstance(r, Exception))
        }
```

### 3. Curator Agent

```python
class CuratorAgent(BaseSpecializedAgent):
    """Knowledge organization and curation specialist."""

    def _initialize_specialized_components(self):
        self.knowledge_graph = KnowledgeGraph()
        self.taxonomy = TaxonomyManager()
        self.quality_scorer = QualityScorer()

    async def handle_knowledge_curation(self, knowledge: Dict) -> Dict[str, Any]:
        """Curate and organize knowledge."""
        # Classify knowledge
        category = self.taxonomy.classify(knowledge)

        # Score quality
        quality_score = self.quality_scorer.evaluate(knowledge)

        # Add to knowledge graph if quality threshold met
        if quality_score >= self.config.quality_threshold:
            node_id = self.knowledge_graph.add_node(
                knowledge,
                category=category,
                quality=quality_score
            )

            # Find and create relationships
            related = self.knowledge_graph.find_related(node_id, threshold=0.7)
            for related_id, similarity in related:
                self.knowledge_graph.add_edge(node_id, related_id, weight=similarity)

            return {
                "success": True,
                "node_id": node_id,
                "category": category,
                "quality_score": quality_score,
                "relationships": len(related)
            }
        else:
            return {
                "success": False,
                "reason": "Quality threshold not met",
                "quality_score": quality_score
            }
```

### 4. Navigator Agent

```python
class NavigatorAgent(BaseSpecializedAgent):
    """Path finding and exploration specialist."""

    def _initialize_specialized_components(self):
        self.path_finder = PathFinder()
        self.exploration_strategy = ExplorationStrategy()
        self.map_builder = MapBuilder()

    async def handle_path_finding(self, request: Dict) -> Dict[str, Any]:
        """Find optimal path between points."""
        start = request["start"]
        goal = request["goal"]
        constraints = request.get("constraints", {})

        # Build or update map
        current_map = self.map_builder.get_current_map()

        # Find path
        path = self.path_finder.find_path(
            current_map,
            start,
            goal,
            constraints
        )

        if path:
            return {
                "success": True,
                "path": path,
                "distance": self.path_finder.calculate_distance(path),
                "estimated_time": self.path_finder.estimate_time(path)
            }
        else:
            # Explore to find path
            exploration_result = await self.explore_for_path(start, goal)
            return exploration_result

    async def explore_for_path(self, start: Any, goal: Any) -> Dict[str, Any]:
        """Explore unknown territory to find path."""
        strategy = self.exploration_strategy.select_strategy(start, goal)

        explored_nodes = []
        max_iterations = 100

        for i in range(max_iterations):
            next_node = strategy.get_next_exploration_target()
            result = await self.explore_node(next_node)
            explored_nodes.append(result)

            # Update map
            self.map_builder.add_node(result)

            # Check if path now exists
            path = self.path_finder.find_path(
                self.map_builder.get_current_map(),
                start,
                goal
            )

            if path:
                return {
                    "success": True,
                    "path": path,
                    "nodes_explored": len(explored_nodes),
                    "iterations": i + 1
                }

        return {
            "success": False,
            "reason": "Path not found after exploration",
            "nodes_explored": len(explored_nodes)
        }
```

### 5. Alchemist Agent

```python
class AlchemistAgent(BaseSpecializedAgent):
    """Transformation and synthesis specialist."""

    def _initialize_specialized_components(self):
        self.transformation_engine = TransformationEngine()
        self.synthesis_lab = SynthesisLab()
        self.recipe_book = RecipeBook()

    async def handle_transformation(self, request: Dict) -> Dict[str, Any]:
        """Transform input into desired output."""
        input_data = request["input"]
        target_type = request["target_type"]

        # Find transformation recipe
        recipe = self.recipe_book.find_recipe(
            type(input_data).__name__,
            target_type
        )

        if recipe:
            # Apply transformation
            result = await self.transformation_engine.transform(
                input_data,
                recipe
            )

            return {
                "success": True,
                "output": result,
                "recipe_used": recipe.name,
                "transformation_steps": len(recipe.steps)
            }
        else:
            # Attempt synthesis
            return await self.synthesize_transformation(input_data, target_type)

    async def synthesize_transformation(self, input_data: Any, target_type: str) -> Dict[str, Any]:
        """Synthesize new transformation when no recipe exists."""
        # Analyze input
        input_properties = self.synthesis_lab.analyze(input_data)

        # Generate transformation steps
        steps = self.synthesis_lab.generate_steps(
            input_properties,
            target_type
        )

        # Test transformation
        result = await self.synthesis_lab.test_transformation(
            input_data,
            steps
        )

        if result["success"]:
            # Save new recipe
            recipe_id = self.recipe_book.save_recipe(
                input_type=type(input_data).__name__,
                output_type=target_type,
                steps=steps,
                success_rate=result["confidence"]
            )

            return {
                "success": True,
                "output": result["output"],
                "new_recipe_created": True,
                "recipe_id": recipe_id
            }
        else:
            return {
                "success": False,
                "reason": "Synthesis failed",
                "attempted_steps": len(steps)
            }
```

### 6. Guardian Agent

```python
class GuardianAgent(BaseSpecializedAgent):
    """Protection and validation specialist."""

    def _initialize_specialized_components(self):
        self.security_scanner = SecurityScanner()
        self.validator = Validator()
        self.threat_detector = ThreatDetector()
        self.shield_generator = ShieldGenerator()

    async def handle_validation(self, target: Any) -> Dict[str, Any]:
        """Validate input/output against rules."""
        # Run validation checks
        validation_results = await self.validator.validate(target)

        # Check for threats
        threat_assessment = self.threat_detector.assess(target)

        # Generate protection if needed
        if threat_assessment["threat_level"] > 0.5:
            shield = self.shield_generator.generate(
                threat_type=threat_assessment["threat_type"],
                protection_level=threat_assessment["threat_level"]
            )
        else:
            shield = None

        return {
            "success": validation_results["is_valid"],
            "validation_errors": validation_results.get("errors", []),
            "threat_level": threat_assessment["threat_level"],
            "shield_deployed": shield is not None,
            "recommendations": self._generate_recommendations(
                validation_results,
                threat_assessment
            )
        }

    def _generate_recommendations(self, validation: Dict, threats: Dict) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        if validation.get("errors"):
            recommendations.append("Fix validation errors before proceeding")

        if threats["threat_level"] > 0.3:
            recommendations.append(f"Monitor for {threats['threat_type']} threats")

        if threats["threat_level"] > 0.7:
            recommendations.append("Implement additional security measures")

        return recommendations
```

### 7. Chronicler Agent

```python
class ChroniclerAgent(BaseSpecializedAgent):
    """Historical tracking and versioning specialist."""

    def _initialize_specialized_components(self):
        self.timeline = Timeline()
        self.version_tracker = VersionTracker()
        self.event_log = EventLog()

    async def handle_event_recording(self, event: Dict) -> Dict[str, Any]:
        """Record historical event."""
        # Add to timeline
        event_id = self.timeline.add_event(
            timestamp=event.get("timestamp", time.time()),
            type=event["type"],
            data=event["data"],
            actor=event.get("actor", "unknown")
        )

        # Log event
        self.event_log.log(event_id, event)

        # Track version changes
        if event.get("version_change"):
            self.version_tracker.record_change(
                entity=event["entity"],
                old_version=event.get("old_version"),
                new_version=event["new_version"],
                change_type=event.get("change_type")
            )

        return {
            "success": True,
            "event_id": event_id,
            "timeline_position": self.timeline.get_position(event_id),
            "related_events": self.timeline.find_related(event_id)
        }

    async def handle_history_query(self, query: Dict) -> Dict[str, Any]:
        """Query historical data."""
        start_time = query.get("start_time")
        end_time = query.get("end_time")
        entity = query.get("entity")
        event_type = query.get("event_type")

        # Search timeline
        events = self.timeline.search(
            start_time=start_time,
            end_time=end_time,
            entity=entity,
            event_type=event_type
        )

        # Get version history if entity specified
        if entity:
            versions = self.version_tracker.get_history(entity)
        else:
            versions = []

        return {
            "success": True,
            "events": events,
            "version_history": versions,
            "time_range": {
                "start": start_time,
                "end": end_time
            }
        }
```

### 8. Artificer Agent

```python
class ArtificerAgent(BaseSpecializedAgent):
    """Tool creation and enhancement specialist."""

    def _initialize_specialized_components(self):
        self.tool_forge = ToolForge()
        self.enhancement_lab = EnhancementLab()
        self.tool_registry = ToolRegistry()

    async def handle_tool_creation(self, spec: Dict) -> Dict[str, Any]:
        """Create new tool from specification."""
        # Design tool
        design = self.tool_forge.design_tool(
            purpose=spec["purpose"],
            inputs=spec.get("inputs", []),
            outputs=spec.get("outputs", []),
            constraints=spec.get("constraints", {})
        )

        # Forge tool
        tool = await self.tool_forge.forge(design)

        # Test tool
        test_results = await self.tool_forge.test(tool, spec.get("test_cases", []))

        if test_results["success_rate"] > 0.8:
            # Register tool
            tool_id = self.tool_registry.register(
                tool,
                metadata={
                    "creator": self.agent_id,
                    "purpose": spec["purpose"],
                    "success_rate": test_results["success_rate"]
                }
            )

            return {
                "success": True,
                "tool_id": tool_id,
                "tool_interface": tool.get_interface(),
                "test_results": test_results
            }
        else:
            return {
                "success": False,
                "reason": "Tool failed testing",
                "test_results": test_results
            }

    async def handle_tool_enhancement(self, request: Dict) -> Dict[str, Any]:
        """Enhance existing tool."""
        tool_id = request["tool_id"]
        enhancements = request["enhancements"]

        # Get tool
        tool = self.tool_registry.get(tool_id)

        # Apply enhancements
        enhanced_tool = await self.enhancement_lab.enhance(
            tool,
            enhancements
        )

        # Test enhanced version
        test_results = await self.tool_forge.test(
            enhanced_tool,
            request.get("test_cases", [])
        )

        if test_results["improvement"] > 0:
            # Update registry
            new_version = self.tool_registry.update(
                tool_id,
                enhanced_tool,
                version_increment="minor"
            )

            return {
                "success": True,
                "new_version": new_version,
                "improvements": test_results["improvement"],
                "enhancements_applied": len(enhancements)
            }
        else:
            return {
                "success": False,
                "reason": "No improvement achieved",
                "test_results": test_results
            }
```

### 9. Emissary Agent

```python
class EmissaryAgent(BaseSpecializedAgent):
    """External system integration specialist."""

    def _initialize_specialized_components(self):
        self.connector_factory = ConnectorFactory()
        self.protocol_translator = ProtocolTranslator()
        self.credential_manager = CredentialManager()
        self.api_gateway = APIGateway()

    async def handle_external_integration(self, request: Dict) -> Dict[str, Any]:
        """Integrate with external system."""
        system_type = request["system_type"]
        endpoint = request["endpoint"]

        # Get or create connector
        connector = self.connector_factory.get_connector(
            system_type,
            endpoint,
            credentials=self.credential_manager.get_credentials(system_type)
        )

        # Translate protocol if needed
        if request.get("protocol") != connector.native_protocol:
            request = self.protocol_translator.translate(
                request,
                from_protocol=request.get("protocol"),
                to_protocol=connector.native_protocol
            )

        # Execute request
        try:
            response = await connector.execute(request)

            # Translate response back
            if request.get("protocol") != connector.native_protocol:
                response = self.protocol_translator.translate(
                    response,
                    from_protocol=connector.native_protocol,
                    to_protocol=request.get("protocol")
                )

            return {
                "success": True,
                "response": response,
                "system": system_type,
                "latency_ms": connector.last_latency
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "system": system_type,
                "retry_possible": connector.is_retriable(e)
            }

    async def handle_api_proxy(self, request: Dict) -> Dict[str, Any]:
        """Proxy API requests through gateway."""
        return await self.api_gateway.proxy(request)
```

### 10. Steward Agent

```python
class StewardAgent(BaseSpecializedAgent):
    """Resource management and allocation specialist."""

    def _initialize_specialized_components(self):
        self.resource_pool = ResourcePool()
        self.allocation_manager = AllocationManager()
        self.optimizer = ResourceOptimizer()
        self.monitor = ResourceMonitor()

    async def handle_resource_allocation(self, request: Dict) -> Dict[str, Any]:
        """Allocate resources to agents/tasks."""
        requester = request["requester"]
        resources = request["resources"]
        priority = request.get("priority", Priority.NORMAL)
        duration = request.get("duration", None)

        # Check availability
        available = self.resource_pool.check_availability(resources)

        if available:
            # Allocate resources
            allocation_id = self.allocation_manager.allocate(
                requester=requester,
                resources=resources,
                priority=priority,
                duration=duration
            )

            # Start monitoring
            self.monitor.track_allocation(allocation_id)

            return {
                "success": True,
                "allocation_id": allocation_id,
                "resources_allocated": resources,
                "expires_at": time.time() + duration if duration else None
            }
        else:
            # Try optimization
            optimized = await self.optimizer.optimize_allocation(
                self.resource_pool.get_current_allocations(),
                resources,
                priority
            )

            if optimized:
                # Reallocate based on optimization
                await self._reallocate_resources(optimized)

                # Retry allocation
                return await self.handle_resource_allocation(request)
            else:
                return {
                    "success": False,
                    "reason": "Insufficient resources",
                    "available": self.resource_pool.get_available(),
                    "queue_position": self.allocation_manager.queue_request(request)
                }

    async def handle_resource_release(self, allocation_id: str) -> Dict[str, Any]:
        """Release allocated resources."""
        # Stop monitoring
        self.monitor.stop_tracking(allocation_id)

        # Release resources
        released = self.allocation_manager.release(allocation_id)

        # Return to pool
        self.resource_pool.return_resources(released)

        # Process queued requests
        processed = await self._process_resource_queue()

        return {
            "success": True,
            "resources_released": released,
            "queued_requests_processed": processed
        }
```

## Boilerplate Generator Pattern

```python
#!/usr/bin/env python3
"""Agent boilerplate generator."""

import os
from pathlib import Path
from typing import Dict, List, Any
import jinja2

class AgentGenerator:
    """Generate boilerplate for new agents."""

    def __init__(self, template_dir: Path = Path("templates/agents")):
        self.template_dir = template_dir
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir))
        )

    def generate_agent(
        self,
        name: str,
        role: str,
        capabilities: List[str],
        output_dir: Path = Path("src/production/agents")
    ) -> Dict[str, Path]:
        """Generate complete agent implementation."""

        # Prepare context
        context = {
            "agent_name": name,
            "agent_class": f"{name.capitalize()}Agent",
            "role": role,
            "capabilities": capabilities,
            "imports": self._generate_imports(capabilities),
            "methods": self._generate_methods(capabilities)
        }

        # Generate files
        generated_files = {}

        # Main agent file
        agent_template = self.jinja_env.get_template("agent_base.py.j2")
        agent_content = agent_template.render(context)
        agent_path = output_dir / f"{name.lower()}_agent.py"
        agent_path.write_text(agent_content)
        generated_files["agent"] = agent_path

        # Test file
        test_template = self.jinja_env.get_template("agent_test.py.j2")
        test_content = test_template.render(context)
        test_path = output_dir.parent / "tests" / f"test_{name.lower()}_agent.py"
        test_path.write_text(test_content)
        generated_files["test"] = test_path

        # Configuration file
        config_template = self.jinja_env.get_template("agent_config.yaml.j2")
        config_content = config_template.render(context)
        config_path = output_dir / "configs" / f"{name.lower()}_config.yaml"
        config_path.write_text(config_content)
        generated_files["config"] = config_path

        return generated_files

    def _generate_imports(self, capabilities: List[str]) -> List[str]:
        """Generate required imports based on capabilities."""
        imports = [
            "from typing import Any, Dict, List, Optional",
            "import asyncio",
            "import time",
            "from agents.unified_base_agent import UnifiedBaseAgent",
            "from core.communication import Message, MessageType",
        ]

        # Add capability-specific imports
        capability_imports = {
            "document": "from utils.document_store import DocumentStore",
            "communicate": "from utils.messaging import MessageQueue",
            "validate": "from utils.validation import Validator",
            "transform": "from utils.transformation import TransformationEngine",
            # Add more as needed
        }

        for capability in capabilities:
            if capability in capability_imports:
                imports.append(capability_imports[capability])

        return imports

    def _generate_methods(self, capabilities: List[str]) -> List[Dict[str, Any]]:
        """Generate method stubs for capabilities."""
        methods = []

        for capability in capabilities:
            methods.append({
                "name": f"handle_{capability}",
                "params": "self, task: LangroidTask",
                "return_type": "Dict[str, Any]",
                "docstring": f"Handle {capability} tasks.",
                "body": f'return {{"success": True, "capability": "{capability}"}}'
            })

        return methods

# Template: agent_base.py.j2
AGENT_TEMPLATE = """
{%- for import in imports %}
{{ import }}
{%- endfor %}

class {{ agent_class }}(UnifiedBaseAgent):
    \"\"\"{{ role }}\"\"\"

    def _initialize_specialized_components(self):
        \"\"\"Initialize {{ agent_name }}-specific components.\"\"\"
        # TODO: Initialize specialized components
        pass

    def _get_task_handler(self, task_type: str):
        \"\"\"Get task handler for {{ agent_name }}.\"\"\"
        handlers = {
            {%- for capability in capabilities %}
            "{{ capability }}": self.handle_{{ capability }},
            {%- endfor %}
            "default": self._handle_default_task
        }
        return handlers.get(task_type, handlers["default"])

    {%- for method in methods %}

    async def {{ method.name }}({{ method.params }}) -> {{ method.return_type }}:
        \"\"\"{{ method.docstring }}\"\"\"
        {{ method.body }}
    {%- endfor %}
"""

# Usage example
if __name__ == "__main__":
    generator = AgentGenerator()

    # Generate Scribe agent
    files = generator.generate_agent(
        name="Scribe",
        role="Documentation and record-keeping specialist",
        capabilities=["document", "record", "index", "version"]
    )

    print(f"Generated {len(files)} files for Scribe agent:")
    for file_type, path in files.items():
        print(f"  - {file_type}: {path}")
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)
1. Implement base agent class with all common functionality
2. Set up communication protocol and message routing
3. Create resource management system
4. Implement state management and checkpointing

### Phase 2: Essential Agents (Week 2)
1. **Scribe** - Critical for documentation
2. **Herald** - Essential for communication
3. **Guardian** - Required for validation
4. **Steward** - Needed for resource management

### Phase 3: Support Agents (Week 3)
1. **Curator** - Knowledge organization
2. **Navigator** - Path finding
3. **Chronicler** - History tracking

### Phase 4: Advanced Agents (Week 4)
1. **Alchemist** - Transformation capabilities
2. **Artificer** - Tool creation
3. **Emissary** - External integration

### Testing Strategy

```python
# Test framework for agents
class AgentTestFramework:
    """Comprehensive testing for agent implementations."""

    async def test_agent_lifecycle(self, agent_class):
        """Test agent initialization, execution, and shutdown."""

    async def test_communication(self, agent):
        """Test inter-agent communication."""

    async def test_resource_management(self, agent):
        """Test resource allocation and constraints."""

    async def test_error_handling(self, agent):
        """Test error recovery and resilience."""

    async def test_evolution(self, agent):
        """Test agent evolution capabilities."""
```

## Deployment Considerations

1. **Resource Allocation**
   - Each agent requires 100-500MB memory
   - CPU usage varies by specialization
   - GPU required for certain agents (Alchemist, Artificer)

2. **Communication Overhead**
   - Message routing adds 1-5ms latency
   - Broadcast operations scale O(n) with agent count
   - Use message batching for efficiency

3. **Monitoring Requirements**
   - Track agent health and performance
   - Monitor resource utilization
   - Log inter-agent communication patterns

## Conclusion

This guide provides a complete framework for implementing the remaining 10 agents in the AIVillage system. The modular architecture, standardized patterns, and boilerplate generator enable rapid development while maintaining consistency and quality across all agent implementations.

Key success factors:
- Follow established patterns from existing agents
- Use the base agent class for common functionality
- Implement proper error handling and resource management
- Test thoroughly with the provided framework
- Generate boilerplate to accelerate development

The phased implementation approach ensures critical agents are deployed first while maintaining system stability throughout the development process.
