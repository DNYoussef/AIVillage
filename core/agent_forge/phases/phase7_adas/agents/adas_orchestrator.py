"""
ADAS Orchestrator - Phase 7
System coordination and management for ADAS agents
"""

import asyncio
import logging
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor
import queue

class SystemMode(Enum):
    INITIALIZATION = "initialization"
    NORMAL_OPERATION = "normal_operation"
    DEGRADED_MODE = "degraded_mode"
    EMERGENCY_MODE = "emergency_mode"
    SHUTDOWN = "shutdown"

class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class AgentInfo:
    """Agent information structure"""
    agent_id: str
    agent_type: str
    status: AgentStatus
    last_heartbeat: float
    performance_metrics: Dict[str, Any]
    error_count: int
    uptime: float
    capabilities: List[str]

@dataclass
class SystemState:
    """Overall system state"""
    mode: SystemMode
    active_agents: int
    total_agents: int
    system_health: float
    processing_latency: float
    throughput: float
    error_rate: float
    timestamp: float

class AdasOrchestrator:
    """
    ADAS System Orchestrator
    Coordinates and manages all ADAS agents for optimal performance
    """

    def __init__(self, agent_id: str = "adas_orchestrator_001"):
        self.agent_id = agent_id
        self.logger = self._setup_logging()

        # System configuration
        self.system_mode = SystemMode.INITIALIZATION
        self.max_processing_time = 0.025  # 25ms orchestration target
        self.orchestration_frequency = 20  # 20Hz

        # Agent management
        self.registered_agents: Dict[str, AgentInfo] = {}
        self.agent_dependencies = {}
        self.data_flow_graph = {}

        # Communication queues
        self.input_queues = {}
        self.output_queues = {}
        self.coordination_queue = queue.Queue(maxsize=1000)

        # Processing threads
        self.processing_threads = {}
        self.is_running = False

        # Performance monitoring
        self.performance_metrics = {
            'orchestration_latency': [],
            'agent_response_times': {},
            'data_flow_efficiency': 0.0,
            'system_throughput': 0.0,
            'coordination_overhead': 0.0
        }

        # Safety monitoring
        self.safety_thresholds = {
            'max_agent_response_time': 0.100,  # 100ms
            'min_system_health': 0.7,  # 70%
            'max_error_rate': 0.05,  # 5%
            'heartbeat_timeout': 5.0   # 5 seconds
        }

        self.executor = ThreadPoolExecutor(max_workers=8)

        # Initialize agent references (will be set during registration)
        self.sensor_fusion = None
        self.perception = None
        self.prediction = None
        self.planning = None
        self.safety_monitor = None
        self.edge_deployment = None
        self.v2x_communicator = None
        self.compliance_validator = None

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"ADAS.Orchestrator.{self.agent_id}")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def initialize(self) -> bool:
        """Initialize ADAS orchestrator"""
        try:
            self.logger.info("Initializing ADAS Orchestrator")

            # Setup agent dependencies
            await self._setup_agent_dependencies()

            # Initialize data flow graph
            await self._initialize_data_flow()

            # Setup coordination protocols
            await self._setup_coordination_protocols()

            # Start orchestration threads
            self.is_running = True
            await self._start_orchestration_threads()

            self.system_mode = SystemMode.NORMAL_OPERATION
            self.logger.info("ADAS Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def _setup_agent_dependencies(self):
        """Setup agent dependency graph"""
        # Define agent dependencies
        self.agent_dependencies = {
            'sensor_fusion': [],  # No dependencies
            'perception': ['sensor_fusion'],
            'prediction': ['perception'],
            'planning': ['perception', 'prediction'],
            'safety_monitor': ['sensor_fusion', 'perception', 'prediction', 'planning'],
            'edge_deployment': [],
            'v2x_communicator': ['perception', 'prediction'],
            'compliance_validator': ['safety_monitor']
        }

        # Define data flow between agents
        self.data_flow_graph = {
            'sensor_fusion': ['perception'],
            'perception': ['prediction', 'planning', 'safety_monitor'],
            'prediction': ['planning', 'safety_monitor', 'v2x_communicator'],
            'planning': ['safety_monitor'],
            'safety_monitor': ['compliance_validator'],
            'edge_deployment': [],
            'v2x_communicator': [],
            'compliance_validator': []
        }

        self.logger.info("Agent dependencies configured")

    async def _initialize_data_flow(self):
        """Initialize data flow management"""
        # Create communication queues for each agent
        for agent_type in self.agent_dependencies.keys():
            self.input_queues[agent_type] = queue.Queue(maxsize=100)
            self.output_queues[agent_type] = queue.Queue(maxsize=100)

        # Initialize data buffers
        self.data_buffers = {
            'sensor_data': None,
            'scene_understanding': None,
            'behavior_prediction': None,
            'planning_decision': None,
            'safety_assessment': None,
            'v2x_messages': [],
            'compliance_report': None
        }

        self.logger.info("Data flow initialized")

    async def _setup_coordination_protocols(self):
        """Setup coordination protocols"""
        # Coordination strategies
        self.coordination_strategies = {
            'data_synchronization': True,
            'load_balancing': True,
            'fault_tolerance': True,
            'performance_optimization': True,
            'graceful_degradation': True
        }

        # Timing constraints
        self.timing_constraints = {
            'sensor_fusion': 0.008,   # 8ms
            'perception': 0.015,      # 15ms
            'prediction': 0.012,      # 12ms
            'planning': 0.020,        # 20ms
            'safety_monitor': 0.005,  # 5ms (highest priority)
            'v2x_communicator': 0.010,# 10ms
            'compliance_validator': 0.050  # 50ms
        }

        self.logger.info("Coordination protocols setup")

    async def _start_orchestration_threads(self):
        """Start orchestration processing threads"""
        # Main orchestration thread
        self.processing_threads['orchestrator'] = threading.Thread(
            target=self._orchestration_loop,
            daemon=True
        )
        self.processing_threads['orchestrator'].start()

        # Data flow management thread
        self.processing_threads['data_flow'] = threading.Thread(
            target=self._data_flow_loop,
            daemon=True
        )
        self.processing_threads['data_flow'].start()

        # Health monitoring thread
        self.processing_threads['health_monitor'] = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.processing_threads['health_monitor'].start()

        # Performance monitoring thread
        self.processing_threads['perf_monitor'] = threading.Thread(
            target=self._performance_monitoring_loop,
            daemon=True
        )
        self.processing_threads['perf_monitor'].start()

        self.logger.info("Orchestration threads started")

    def _orchestration_loop(self):
        """Main orchestration loop"""
        self.logger.info("Starting orchestration loop")

        while self.is_running:
            try:
                orchestration_start = time.perf_counter()

                # Check system health
                system_health = self._check_system_health()

                # Determine system mode
                new_mode = self._determine_system_mode(system_health)
                if new_mode != self.system_mode:
                    self._handle_mode_transition(new_mode)

                # Coordinate agent execution
                self._coordinate_agent_execution()

                # Handle coordination requests
                self._handle_coordination_requests()

                # Update performance metrics
                orchestration_time = time.perf_counter() - orchestration_start
                self.performance_metrics['orchestration_latency'].append(orchestration_time)
                if len(self.performance_metrics['orchestration_latency']) > 1000:
                    self.performance_metrics['orchestration_latency'].pop(0)

                # Check timing constraints
                if orchestration_time > self.max_processing_time:
                    self.logger.warning(
                        f"Orchestration exceeded time limit: {orchestration_time*1000:.2f}ms"
                    )

                # Sleep for next cycle
                sleep_time = max(0.001, 1.0/self.orchestration_frequency - orchestration_time)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")

    def _check_system_health(self) -> float:
        """Check overall system health"""
        try:
            if not self.registered_agents:
                return 0.0

            # Calculate health based on agent status
            active_agents = sum(1 for agent in self.registered_agents.values()
                              if agent.status == AgentStatus.ACTIVE)
            total_agents = len(self.registered_agents)

            if total_agents == 0:
                return 0.0

            # Base health from active agents
            health = active_agents / total_agents

            # Factor in error rates
            total_errors = sum(agent.error_count for agent in self.registered_agents.values())
            total_operations = total_agents * 100  # Assume 100 operations per agent
            error_rate = total_errors / max(1, total_operations)

            health *= (1.0 - min(0.5, error_rate))

            # Factor in response times
            avg_response_times = []
            for agent_type, times in self.performance_metrics['agent_response_times'].items():
                if times:
                    avg_response_times.append(np.mean(times))

            if avg_response_times:
                avg_response_time = np.mean(avg_response_times)
                response_factor = max(0.5, 1.0 - avg_response_time / 0.1)  # 100ms reference
                health *= response_factor

            return max(0.0, min(1.0, health))

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return 0.5

    def _determine_system_mode(self, system_health: float) -> SystemMode:
        """Determine appropriate system mode based on health"""
        if system_health < 0.3:
            return SystemMode.EMERGENCY_MODE
        elif system_health < 0.6:
            return SystemMode.DEGRADED_MODE
        else:
            return SystemMode.NORMAL_OPERATION

    def _handle_mode_transition(self, new_mode: SystemMode):
        """Handle system mode transition"""
        old_mode = self.system_mode
        self.system_mode = new_mode

        self.logger.info(f"System mode transition: {old_mode.value} -> {new_mode.value}")

        # Execute mode-specific actions
        if new_mode == SystemMode.EMERGENCY_MODE:
            self._enter_emergency_mode()
        elif new_mode == SystemMode.DEGRADED_MODE:
            self._enter_degraded_mode()
        elif new_mode == SystemMode.NORMAL_OPERATION:
            self._enter_normal_mode()

    def _enter_emergency_mode(self):
        """Enter emergency mode"""
        self.logger.critical("Entering EMERGENCY MODE")

        # Prioritize safety-critical agents
        priority_agents = ['safety_monitor', 'sensor_fusion', 'perception']

        # Reduce processing load on non-critical agents
        for agent_type in self.registered_agents:
            if agent_type not in priority_agents:
                self._reduce_agent_load(agent_type)

        # Activate emergency protocols
        if self.safety_monitor:
            asyncio.create_task(self.safety_monitor.force_safety_check())

    def _enter_degraded_mode(self):
        """Enter degraded mode"""
        self.logger.warning("Entering DEGRADED MODE")

        # Reduce system complexity
        self._optimize_processing_pipeline()

        # Increase monitoring frequency
        self.orchestration_frequency = 30  # Increase to 30Hz

    def _enter_normal_mode(self):
        """Enter normal operation mode"""
        self.logger.info("Entering NORMAL MODE")

        # Restore normal processing
        self.orchestration_frequency = 20  # Back to 20Hz

        # Reactivate all agents
        for agent_type in self.registered_agents:
            self._restore_agent_operation(agent_type)

    def _reduce_agent_load(self, agent_type: str):
        """Reduce processing load for specific agent"""
        # Implementation would reduce processing frequency or complexity
        self.logger.debug(f"Reducing load for {agent_type}")

    def _restore_agent_operation(self, agent_type: str):
        """Restore normal operation for specific agent"""
        # Implementation would restore normal processing
        self.logger.debug(f"Restoring operation for {agent_type}")

    def _optimize_processing_pipeline(self):
        """Optimize processing pipeline for degraded mode"""
        # Reduce processing complexity
        self.logger.info("Optimizing processing pipeline")

    def _coordinate_agent_execution(self):
        """Coordinate execution of all agents"""
        try:
            # Process data flow according to dependencies
            execution_order = self._calculate_execution_order()

            for agent_type in execution_order:
                if agent_type in self.registered_agents:
                    agent_info = self.registered_agents[agent_type]

                    if agent_info.status == AgentStatus.ACTIVE:
                        self._trigger_agent_processing(agent_type)

        except Exception as e:
            self.logger.error(f"Agent coordination failed: {e}")

    def _calculate_execution_order(self) -> List[str]:
        """Calculate optimal execution order based on dependencies"""
        # Topological sort of dependency graph
        execution_order = []
        visited = set()
        temp_visited = set()

        def visit(agent_type: str):
            if agent_type in temp_visited:
                return  # Cycle detected
            if agent_type in visited:
                return

            temp_visited.add(agent_type)

            # Process dependencies first
            dependencies = self.agent_dependencies.get(agent_type, [])
            for dep in dependencies:
                visit(dep)

            temp_visited.remove(agent_type)
            visited.add(agent_type)
            execution_order.append(agent_type)

        # Visit all agents
        for agent_type in self.agent_dependencies.keys():
            if agent_type not in visited:
                visit(agent_type)

        return execution_order

    def _trigger_agent_processing(self, agent_type: str):
        """Trigger processing for specific agent"""
        try:
            # Get input data for agent
            input_data = self._prepare_agent_input(agent_type)

            if input_data:
                # Send data to agent's input queue
                if agent_type in self.input_queues:
                    try:
                        self.input_queues[agent_type].put_nowait(input_data)
                    except queue.Full:
                        self.logger.warning(f"Input queue full for {agent_type}")

        except Exception as e:
            self.logger.error(f"Agent triggering failed for {agent_type}: {e}")

    def _prepare_agent_input(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Prepare input data for specific agent"""
        input_data = {'timestamp': time.time()}

        try:
            if agent_type == 'sensor_fusion':
                # Sensor fusion gets raw sensor data
                input_data['sensor_data'] = self._get_mock_sensor_data()

            elif agent_type == 'perception':
                # Perception gets fused sensor data
                input_data['fused_data'] = self.data_buffers.get('sensor_data')

            elif agent_type == 'prediction':
                # Prediction gets scene understanding
                input_data['scene_understanding'] = self.data_buffers.get('scene_understanding')

            elif agent_type == 'planning':
                # Planning gets perception and prediction data
                input_data['scene_understanding'] = self.data_buffers.get('scene_understanding')
                input_data['behavior_prediction'] = self.data_buffers.get('behavior_prediction')
                input_data['vehicle_state'] = self._get_vehicle_state()
                input_data['navigation_goal'] = self._get_navigation_goal()

            elif agent_type == 'safety_monitor':
                # Safety monitor gets all relevant data
                input_data['vehicle_state'] = self._get_vehicle_state()
                input_data['scene_understanding'] = self.data_buffers.get('scene_understanding')
                input_data['behavior_prediction'] = self.data_buffers.get('behavior_prediction')
                input_data['planning_decision'] = self.data_buffers.get('planning_decision')
                input_data['system_status'] = self._get_system_status()

            elif agent_type == 'v2x_communicator':
                # V2X gets perception and prediction for sharing
                input_data['scene_understanding'] = self.data_buffers.get('scene_understanding')
                input_data['behavior_prediction'] = self.data_buffers.get('behavior_prediction')
                input_data['vehicle_state'] = self._get_vehicle_state()

            return input_data

        except Exception as e:
            self.logger.error(f"Input preparation failed for {agent_type}: {e}")
            return None

    def _get_mock_sensor_data(self) -> Dict[str, Any]:
        """Get mock sensor data for testing"""
        return {
            'camera': {
                'detections': [
                    {'class': 'vehicle', 'bbox': [100, 200, 300, 400], 'confidence': 0.95}
                ],
                'image_stats': {'mean_brightness': 150, 'contrast': 60}
            },
            'radar': {
                'targets': [
                    {'range': 25.0, 'azimuth': 5.0, 'velocity': -10.0, 'rcs': 15.0}
                ]
            },
            'lidar': {
                'point_cloud': [[1, 2, 3]] * 1000,
                'clusters': [{'points': [[1, 2, 3]] * 10}]
            }
        }

    def _get_vehicle_state(self) -> Dict[str, Any]:
        """Get current vehicle state"""
        return {
            'position': [0.0, 0.0, 0.0],
            'velocity': [0.0, 15.0, 0.0],  # 15 m/s forward
            'acceleration': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0],
            'timestamp': time.time()
        }

    def _get_navigation_goal(self) -> Dict[str, Any]:
        """Get navigation goal"""
        return {
            'position': [0.0, 1000.0, 0.0],  # 1km ahead
            'timestamp': time.time()
        }

    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status for monitoring"""
        return {
            'sensor_fusion': {'confidence': 0.95, 'availability': 1.0, 'error_rate': 0.01},
            'perception': {'confidence': 0.92, 'availability': 1.0, 'error_rate': 0.02},
            'prediction': {'confidence': 0.88, 'availability': 1.0, 'error_rate': 0.03},
            'planning': {'confidence': 0.90, 'availability': 1.0, 'error_rate': 0.02}
        }

    def _handle_coordination_requests(self):
        """Handle coordination requests from agents"""
        try:
            while not self.coordination_queue.empty():
                try:
                    request = self.coordination_queue.get_nowait()
                    self._process_coordination_request(request)
                    self.coordination_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Coordination request handling failed: {e}")

    def _process_coordination_request(self, request: Dict[str, Any]):
        """Process specific coordination request"""
        request_type = request.get('type')
        agent_id = request.get('agent_id')

        if request_type == 'priority_boost':
            self._boost_agent_priority(agent_id)
        elif request_type == 'data_request':
            self._handle_data_request(request)
        elif request_type == 'emergency_signal':
            self._handle_emergency_signal(request)

    def _boost_agent_priority(self, agent_id: str):
        """Boost processing priority for specific agent"""
        self.logger.info(f"Boosting priority for {agent_id}")

    def _handle_data_request(self, request: Dict[str, Any]):
        """Handle data request from agent"""
        data_type = request.get('data_type')
        requesting_agent = request.get('agent_id')

        if data_type in self.data_buffers:
            # Send requested data to agent
            data = self.data_buffers[data_type]
            # Implementation would send data to requesting agent

    def _handle_emergency_signal(self, request: Dict[str, Any]):
        """Handle emergency signal from agent"""
        self.logger.critical(f"Emergency signal from {request.get('agent_id')}: {request.get('message')}")

        # Trigger emergency mode
        self._handle_mode_transition(SystemMode.EMERGENCY_MODE)

    def _data_flow_loop(self):
        """Data flow management loop"""
        self.logger.info("Starting data flow loop")

        while self.is_running:
            try:
                # Process output data from all agents
                self._process_agent_outputs()

                # Update data buffers
                self._update_data_buffers()

                # Check data freshness
                self._check_data_freshness()

                time.sleep(0.01)  # 100Hz data flow

            except Exception as e:
                self.logger.error(f"Data flow loop error: {e}")

    def _process_agent_outputs(self):
        """Process outputs from all agents"""
        for agent_type, output_queue in self.output_queues.items():
            try:
                while not output_queue.empty():
                    output_data = output_queue.get_nowait()
                    self._process_agent_output(agent_type, output_data)
                    output_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Output processing failed for {agent_type}: {e}")

    def _process_agent_output(self, agent_type: str, output_data: Any):
        """Process output from specific agent"""
        try:
            if agent_type == 'sensor_fusion':
                self.data_buffers['sensor_data'] = output_data
            elif agent_type == 'perception':
                self.data_buffers['scene_understanding'] = output_data
            elif agent_type == 'prediction':
                self.data_buffers['behavior_prediction'] = output_data
            elif agent_type == 'planning':
                self.data_buffers['planning_decision'] = output_data
            elif agent_type == 'safety_monitor':
                self.data_buffers['safety_assessment'] = output_data
                # Check for safety violations
                if hasattr(output_data, 'overall_safety_level'):
                    if output_data.overall_safety_level.value in ['critical', 'emergency']:
                        self._handle_safety_emergency(output_data)
            elif agent_type == 'compliance_validator':
                self.data_buffers['compliance_report'] = output_data

        except Exception as e:
            self.logger.error(f"Agent output processing failed for {agent_type}: {e}")

    def _handle_safety_emergency(self, safety_assessment):
        """Handle safety emergency from safety monitor"""
        self.logger.critical("Safety emergency detected!")

        # Trigger emergency actions
        emergency_actions = getattr(safety_assessment, 'emergency_actions', [])
        for action in emergency_actions:
            self._execute_emergency_action(action)

    def _execute_emergency_action(self, action: Dict[str, Any]):
        """Execute emergency action"""
        action_type = action.get('action', '')

        if action_type == 'immediate_stop':
            self.logger.critical("EXECUTING IMMEDIATE STOP")
        elif action_type == 'emergency_brake':
            self.logger.critical("EXECUTING EMERGENCY BRAKE")
        elif action_type == 'emergency_swerve':
            self.logger.critical("EXECUTING EMERGENCY SWERVE")

    def _update_data_buffers(self):
        """Update data buffers with timestamps"""
        current_time = time.time()

        # Add timestamps to all buffers
        for buffer_name, data in self.data_buffers.items():
            if data and isinstance(data, dict):
                data['buffer_updated'] = current_time

    def _check_data_freshness(self):
        """Check data freshness and alert on stale data"""
        current_time = time.time()
        max_age = 1.0  # 1 second max age

        for buffer_name, data in self.data_buffers.items():
            if data and isinstance(data, dict):
                last_update = data.get('buffer_updated', 0)
                age = current_time - last_update

                if age > max_age:
                    self.logger.warning(f"Stale data in {buffer_name}: {age:.2f}s old")

    def _health_monitoring_loop(self):
        """Health monitoring loop"""
        self.logger.info("Starting health monitoring loop")

        while self.is_running:
            try:
                current_time = time.time()

                # Check agent heartbeats
                self._check_agent_heartbeats(current_time)

                # Monitor agent performance
                self._monitor_agent_performance()

                # Check system resources
                self._check_system_resources()

                time.sleep(1.0)  # 1 Hz health monitoring

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")

    def _check_agent_heartbeats(self, current_time: float):
        """Check agent heartbeats for liveness"""
        timeout = self.safety_thresholds['heartbeat_timeout']

        for agent_id, agent_info in self.registered_agents.items():
            time_since_heartbeat = current_time - agent_info.last_heartbeat

            if time_since_heartbeat > timeout:
                if agent_info.status != AgentStatus.OFFLINE:
                    self.logger.error(f"Agent {agent_id} heartbeat timeout: {time_since_heartbeat:.1f}s")
                    agent_info.status = AgentStatus.OFFLINE
                    agent_info.error_count += 1

    def _monitor_agent_performance(self):
        """Monitor individual agent performance"""
        for agent_id, agent_info in self.registered_agents.items():
            # Check response times
            if agent_id in self.performance_metrics['agent_response_times']:
                response_times = self.performance_metrics['agent_response_times'][agent_id]
                if response_times:
                    avg_response_time = np.mean(response_times[-10:])  # Last 10 measurements

                    if avg_response_time > self.safety_thresholds['max_agent_response_time']:
                        self.logger.warning(f"High response time for {agent_id}: {avg_response_time*1000:.1f}ms")

    def _check_system_resources(self):
        """Check system resource utilization"""
        # Monitor CPU, memory, etc.
        # Implementation would check actual system resources
        pass

    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        self.logger.info("Starting performance monitoring loop")

        while self.is_running:
            try:
                # Calculate system throughput
                self._calculate_system_throughput()

                # Monitor data flow efficiency
                self._monitor_data_flow_efficiency()

                # Update coordination overhead
                self._update_coordination_overhead()

                time.sleep(5.0)  # 5 second performance monitoring

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")

    def _calculate_system_throughput(self):
        """Calculate overall system throughput"""
        # Implementation would calculate actual throughput metrics
        self.performance_metrics['system_throughput'] = 20.0  # 20 Hz

    def _monitor_data_flow_efficiency(self):
        """Monitor data flow efficiency"""
        # Implementation would monitor actual data flow
        self.performance_metrics['data_flow_efficiency'] = 0.85  # 85%

    def _update_coordination_overhead(self):
        """Update coordination overhead metrics"""
        if self.performance_metrics['orchestration_latency']:
            avg_overhead = np.mean(self.performance_metrics['orchestration_latency'][-100:])
            self.performance_metrics['coordination_overhead'] = avg_overhead

    async def register_agent(self, agent_id: str, agent_type: str, agent_instance: Any) -> bool:
        """Register ADAS agent with orchestrator"""
        try:
            # Create agent info
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                status=AgentStatus.INITIALIZING,
                last_heartbeat=time.time(),
                performance_metrics={},
                error_count=0,
                uptime=0.0,
                capabilities=[]
            )

            # Store agent reference
            if agent_type == 'sensor_fusion':
                self.sensor_fusion = agent_instance
            elif agent_type == 'perception':
                self.perception = agent_instance
            elif agent_type == 'prediction':
                self.prediction = agent_instance
            elif agent_type == 'planning':
                self.planning = agent_instance
            elif agent_type == 'safety_monitor':
                self.safety_monitor = agent_instance
            elif agent_type == 'edge_deployment':
                self.edge_deployment = agent_instance
            elif agent_type == 'v2x_communicator':
                self.v2x_communicator = agent_instance
            elif agent_type == 'compliance_validator':
                self.compliance_validator = agent_instance

            # Register agent
            self.registered_agents[agent_id] = agent_info

            # Initialize performance tracking
            self.performance_metrics['agent_response_times'][agent_id] = []

            agent_info.status = AgentStatus.ACTIVE
            self.logger.info(f"Registered agent: {agent_id} ({agent_type})")
            return True

        except Exception as e:
            self.logger.error(f"Agent registration failed: {e}")
            return False

    async def agent_heartbeat(self, agent_id: str, performance_data: Optional[Dict[str, Any]] = None):
        """Receive heartbeat from agent"""
        if agent_id in self.registered_agents:
            agent_info = self.registered_agents[agent_id]
            agent_info.last_heartbeat = time.time()

            if performance_data:
                agent_info.performance_metrics = performance_data

            # Update status if was offline
            if agent_info.status == AgentStatus.OFFLINE:
                agent_info.status = AgentStatus.ACTIVE
                self.logger.info(f"Agent {agent_id} back online")

    async def get_system_state(self) -> SystemState:
        """Get current system state"""
        active_agents = sum(1 for agent in self.registered_agents.values()
                           if agent.status == AgentStatus.ACTIVE)
        total_agents = len(self.registered_agents)

        system_health = self._check_system_health()

        avg_latency = 0.0
        if self.performance_metrics['orchestration_latency']:
            avg_latency = np.mean(self.performance_metrics['orchestration_latency'][-100:])

        return SystemState(
            mode=self.system_mode,
            active_agents=active_agents,
            total_agents=total_agents,
            system_health=system_health,
            processing_latency=avg_latency,
            throughput=self.performance_metrics['system_throughput'],
            error_rate=sum(agent.error_count for agent in self.registered_agents.values()) / max(1, total_agents),
            timestamp=time.time()
        )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = self.performance_metrics.copy()

        if metrics['orchestration_latency']:
            metrics['avg_orchestration_latency'] = np.mean(metrics['orchestration_latency'])
            metrics['max_orchestration_latency'] = np.max(metrics['orchestration_latency'])

        metrics['system_mode'] = self.system_mode.value
        metrics['registered_agents'] = len(self.registered_agents)
        metrics['agent_details'] = {
            agent_id: {
                'status': agent.status.value,
                'error_count': agent.error_count,
                'uptime': time.time() - agent.last_heartbeat
            }
            for agent_id, agent in self.registered_agents.items()
        }

        return metrics

    async def shutdown(self):
        """Shutdown ADAS orchestrator"""
        self.logger.info("Shutting down ADAS Orchestrator")

        self.system_mode = SystemMode.SHUTDOWN
        self.is_running = False

        # Wait for threads to finish
        for thread_name, thread in self.processing_threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=1.0)

        self.executor.shutdown(wait=True)

        self.logger.info("ADAS Orchestrator shutdown complete")

# Example usage
if __name__ == "__main__":
    async def test_orchestrator():
        orchestrator = AdasOrchestrator()

        if await orchestrator.initialize():
            print("ADAS Orchestrator initialized successfully")

            # Simulate agent registration
            await orchestrator.register_agent("sf_001", "sensor_fusion", None)
            await orchestrator.register_agent("perc_001", "perception", None)
            await orchestrator.register_agent("pred_001", "prediction", None)

            # Simulate some operation
            await asyncio.sleep(2.0)

            # Get system state
            state = await orchestrator.get_system_state()
            print(f"System mode: {state.mode.value}")
            print(f"Active agents: {state.active_agents}/{state.total_agents}")
            print(f"System health: {state.system_health:.2f}")

            # Get metrics
            metrics = await orchestrator.get_performance_metrics()
            print(f"Registered agents: {metrics['registered_agents']}")

            await orchestrator.shutdown()

    asyncio.run(test_orchestrator())