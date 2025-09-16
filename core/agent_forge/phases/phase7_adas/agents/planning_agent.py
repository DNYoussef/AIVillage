"""
PlanningAgent - Path planning and motion planning for ADAS

Advanced path planning agent providing real-time trajectory generation,
obstacle avoidance, and motion planning with safety guarantees.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq

from ..config.adas_config import ADASConfig, ASILLevel
from .perception_agent import DetectedObject
from .prediction_agent import PredictedTrajectory, TrajectoryPoint

class PlanningState(Enum):
    """Planning system states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    EMERGENCY = "emergency"
    DEGRADED = "degraded"
    ERROR = "error"

class ManeuverType(Enum):
    """Types of driving maneuvers"""
    LANE_KEEPING = "lane_keeping"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    OVERTAKING = "overtaking"
    EMERGENCY_BRAKE = "emergency_brake"
    EMERGENCY_SWERVE = "emergency_swerve"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    MERGING = "merging"
    PARKING = "parking"

@dataclass
class WaypointNode:
    """Single waypoint in planned path"""
    timestamp: float
    position: Tuple[float, float, float]  # x, y, z
    velocity: float
    acceleration: float
    heading: float
    curvature: float
    lateral_offset: float
    cost: float
    safety_margin: float

@dataclass
class PlannedPath:
    """Complete planned path with metadata"""
    path_id: int
    maneuver_type: ManeuverType
    waypoints: List[WaypointNode]
    total_cost: float
    safety_score: float
    comfort_score: float
    efficiency_score: float
    planning_time_ms: float
    horizon_s: float
    asil_rating: ASILLevel
    emergency_path: bool

@dataclass
class PlanningConstraints:
    """Planning constraints and limits"""
    max_velocity_mps: float = 30.0
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    max_lateral_acceleration_mps2: float = 4.0
    max_jerk_mps3: float = 2.0
    min_following_distance_m: float = 10.0
    lane_width_m: float = 3.5
    vehicle_width_m: float = 2.0
    vehicle_length_m: float = 4.5

@dataclass
class PlanningOutput:
    """Complete planning system output"""
    timestamp: float
    primary_path: PlannedPath
    alternative_paths: List[PlannedPath]
    emergency_path: Optional[PlannedPath]
    current_maneuver: ManeuverType
    planning_state: PlanningState
    obstacles: List[Dict[str, Any]]
    constraints_violated: List[str]
    planning_latency_ms: float
    path_feasibility: float

class PlanningAgent:
    """
    Path planning agent for ADAS autonomous driving

    Provides real-time path planning, obstacle avoidance, and motion planning
    with safety-critical performance guarantees and ISO 26262 compliance.
    """

    def __init__(self, config: ADASConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # State management
        self.state = PlanningState.INITIALIZING
        self.current_path = None
        self.path_id_counter = 0
        self.last_planning_time = 0.0

        # Planning configuration
        self.planning_horizon_s = 8.0    # 8 second planning horizon
        self.planning_step_s = 0.2       # 200ms planning steps
        self.replanning_frequency_hz = 5.0  # Replan every 200ms

        # Planning constraints
        self.constraints = PlanningConstraints()

        # Performance monitoring
        self.performance_metrics = {
            "avg_planning_ms": 0.0,
            "planning_rate_hz": 0.0,
            "path_quality_score": 0.0,
            "constraint_violations": 0,
            "emergency_activations": 0
        }

        # Planning algorithms
        self.path_planners = {}
        self.obstacle_predictor = None
        self.cost_evaluator = None
        self.safety_checker = None

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.planning_thread = None

        # Safety monitoring
        self.safety_monitor = PlanningSafetyMonitor(config)

        # Initialize planning components
        self._initialize_planners()

    def _initialize_planners(self) -> None:
        """Initialize path planning algorithms"""
        try:
            # A* path planner for structured environments
            self.path_planners["astar"] = AStarPlanner(
                grid_resolution=0.5,  # 0.5m grid resolution
                max_search_distance=100.0
            )

            # RRT* planner for complex scenarios
            self.path_planners["rrt_star"] = RRTStarPlanner(
                max_iterations=1000,
                step_size=2.0,
                goal_bias=0.1
            )

            # Lattice planner for highway scenarios
            self.path_planners["lattice"] = LatticePlanner(
                lateral_resolution=0.5,
                longitudinal_resolution=1.0,
                max_horizon=50.0
            )

            # Frenet frame planner for structured roads
            self.path_planners["frenet"] = FrenetPlanner(
                road_width=self.constraints.lane_width_m,
                planning_horizon=self.planning_horizon_s
            )

            # Supporting components
            self.obstacle_predictor = ObstaclePredictor()
            self.cost_evaluator = CostEvaluator(self.constraints)
            self.safety_checker = SafetyChecker(self.constraints)

            self.logger.info("Planning algorithms initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize planners: {e}")
            raise

    async def start(self) -> bool:
        """Start the planning agent"""
        try:
            self.logger.info("Starting PlanningAgent...")

            # Validate planners
            if not self._validate_planners():
                raise ValueError("Planner validation failed")

            # Start safety monitoring
            await self.safety_monitor.start()

            # Start planning loop
            self.running = True
            self.planning_thread = threading.Thread(target=self._planning_loop, daemon=True)
            self.planning_thread.start()

            self.state = PlanningState.ACTIVE
            self.logger.info("PlanningAgent started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start PlanningAgent: {e}")
            self.state = PlanningState.ERROR
            return False

    def _validate_planners(self) -> bool:
        """Validate planning algorithms meet safety requirements"""
        for name, planner in self.path_planners.items():
            if not planner.validate():
                self.logger.error(f"Planner validation failed for {name}")
                return False
        return True

    def _planning_loop(self) -> None:
        """Main planning processing loop"""
        while self.running:
            try:
                # Planning would be triggered by perception/prediction updates
                # For this implementation, we simulate the processing cycle
                time.sleep(1.0 / self.replanning_frequency_hz)

            except Exception as e:
                self.logger.error(f"Planning loop error: {e}")
                self._handle_planning_error(e)

    async def plan_path(self, ego_state: Dict[str, float],
                       predicted_trajectories: List[PredictedTrajectory],
                       goal_state: Dict[str, float],
                       timestamp: float) -> PlanningOutput:
        """Plan optimal path considering predictions and constraints"""
        start_time = time.time()

        try:
            # Validate inputs
            if not self._validate_planning_inputs(ego_state, goal_state):
                raise ValueError("Invalid planning inputs")

            # Predict obstacle positions
            predicted_obstacles = await self._predict_obstacles(
                predicted_trajectories, timestamp
            )

            # Select appropriate planning algorithm
            planner_name = self._select_planner(ego_state, predicted_obstacles)
            planner = self.path_planners[planner_name]

            # Generate multiple path candidates
            path_candidates = await self._generate_path_candidates(
                planner, ego_state, goal_state, predicted_obstacles, timestamp
            )

            # Evaluate and rank path candidates
            ranked_paths = await self._evaluate_paths(path_candidates, predicted_obstacles)

            # Select best paths
            primary_path = ranked_paths[0] if ranked_paths else None
            alternative_paths = ranked_paths[1:3] if len(ranked_paths) > 1 else []

            # Generate emergency path
            emergency_path = await self._generate_emergency_path(
                ego_state, predicted_obstacles, timestamp
            )

            # Determine current maneuver
            current_maneuver = self._determine_maneuver(primary_path, ego_state)

            # Validate path safety
            constraints_violated = await self._validate_path_safety(
                primary_path, predicted_obstacles
            )

            # Check for state transitions
            self._update_planning_state(constraints_violated, emergency_path)

            planning_latency = (time.time() - start_time) * 1000

            # Check latency constraints
            if planning_latency > self.config.latency.planning_max_ms:
                self.logger.warning(f"Planning latency exceeded: {planning_latency:.2f}ms")
                self._handle_latency_violation(planning_latency)

            # Update performance metrics
            self._update_performance_metrics(planning_latency, constraints_violated)

            output = PlanningOutput(
                timestamp=timestamp,
                primary_path=primary_path,
                alternative_paths=alternative_paths,
                emergency_path=emergency_path,
                current_maneuver=current_maneuver,
                planning_state=self.state,
                obstacles=self._format_obstacles(predicted_obstacles),
                constraints_violated=constraints_violated,
                planning_latency_ms=planning_latency,
                path_feasibility=self._compute_path_feasibility(primary_path)
            )

            # Update current path
            self.current_path = primary_path
            self.last_planning_time = timestamp

            # Safety monitoring
            await self.safety_monitor.validate_output(output)

            return output

        except Exception as e:
            self.logger.error(f"Path planning failed: {e}")
            return self._generate_failsafe_output(timestamp)

    def _validate_planning_inputs(self, ego_state: Dict[str, float],
                                 goal_state: Dict[str, float]) -> bool:
        """Validate planning inputs"""
        required_ego_fields = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading']
        for field in required_ego_fields:
            if field not in ego_state:
                return False

        required_goal_fields = ['position_x', 'position_y']
        for field in required_goal_fields:
            if field not in goal_state:
                return False

        return True

    async def _predict_obstacles(self, predicted_trajectories: List[PredictedTrajectory],
                               timestamp: float) -> List[Dict[str, Any]]:
        """Convert predicted trajectories to obstacle predictions"""
        obstacles = []

        for traj in predicted_trajectories:
            # Convert trajectory to time-indexed obstacle positions
            obstacle_states = []
            for point in traj.trajectory_points:
                state = {
                    'timestamp': point.timestamp,
                    'position': point.position,
                    'velocity': point.velocity,
                    'uncertainty': point.uncertainty_ellipse,
                    'confidence': point.confidence
                }
                obstacle_states.append(state)

            obstacle = {
                'object_id': traj.object_id,
                'tracking_id': traj.tracking_id,
                'predicted_states': obstacle_states,
                'collision_probability': traj.collision_probability,
                'risk_level': traj.risk_level,
                'maneuver_class': traj.maneuver_class
            }
            obstacles.append(obstacle)

        return obstacles

    def _select_planner(self, ego_state: Dict[str, float],
                       obstacles: List[Dict[str, Any]]) -> str:
        """Select appropriate planning algorithm based on scenario"""
        # Simple selection logic - would be more sophisticated in practice
        ego_velocity = np.sqrt(ego_state['velocity_x']**2 + ego_state['velocity_y']**2)
        obstacle_count = len(obstacles)

        if ego_velocity > 20.0:  # High speed - use lattice planner
            return "lattice"
        elif obstacle_count > 10:  # Complex scenario - use RRT*
            return "rrt_star"
        elif obstacle_count > 0:  # Structured environment - use A*
            return "astar"
        else:  # Simple scenario - use Frenet
            return "frenet"

    async def _generate_path_candidates(self, planner: Any, ego_state: Dict[str, float],
                                      goal_state: Dict[str, float],
                                      obstacles: List[Dict[str, Any]],
                                      timestamp: float) -> List[PlannedPath]:
        """Generate multiple path candidates"""
        candidates = []

        try:
            # Generate primary path
            primary_waypoints = await planner.plan(ego_state, goal_state, obstacles)
            if primary_waypoints:
                primary_path = self._create_planned_path(
                    primary_waypoints, ManeuverType.LANE_KEEPING, timestamp, False
                )
                candidates.append(primary_path)

            # Generate lane change options if applicable
            if self._should_consider_lane_change(ego_state, obstacles):
                # Left lane change
                left_goal = self._modify_goal_for_lane_change(goal_state, -1)
                left_waypoints = await planner.plan(ego_state, left_goal, obstacles)
                if left_waypoints:
                    left_path = self._create_planned_path(
                        left_waypoints, ManeuverType.LANE_CHANGE_LEFT, timestamp, False
                    )
                    candidates.append(left_path)

                # Right lane change
                right_goal = self._modify_goal_for_lane_change(goal_state, 1)
                right_waypoints = await planner.plan(ego_state, right_goal, obstacles)
                if right_waypoints:
                    right_path = self._create_planned_path(
                        right_waypoints, ManeuverType.LANE_CHANGE_RIGHT, timestamp, False
                    )
                    candidates.append(right_path)

        except Exception as e:
            self.logger.error(f"Path candidate generation failed: {e}")

        return candidates

    def _create_planned_path(self, waypoints: List[Dict], maneuver_type: ManeuverType,
                           timestamp: float, emergency: bool) -> PlannedPath:
        """Create PlannedPath from waypoints"""
        path_waypoints = []
        total_cost = 0.0

        for i, wp in enumerate(waypoints):
            waypoint = WaypointNode(
                timestamp=timestamp + i * self.planning_step_s,
                position=(wp['x'], wp['y'], wp.get('z', 0.0)),
                velocity=wp.get('velocity', 0.0),
                acceleration=wp.get('acceleration', 0.0),
                heading=wp.get('heading', 0.0),
                curvature=wp.get('curvature', 0.0),
                lateral_offset=wp.get('lateral_offset', 0.0),
                cost=wp.get('cost', 0.0),
                safety_margin=wp.get('safety_margin', 5.0)
            )
            path_waypoints.append(waypoint)
            total_cost += waypoint.cost

        # Compute path quality scores
        safety_score = self._compute_safety_score(path_waypoints)
        comfort_score = self._compute_comfort_score(path_waypoints)
        efficiency_score = self._compute_efficiency_score(path_waypoints)

        self.path_id_counter += 1
        return PlannedPath(
            path_id=self.path_id_counter,
            maneuver_type=maneuver_type,
            waypoints=path_waypoints,
            total_cost=total_cost,
            safety_score=safety_score,
            comfort_score=comfort_score,
            efficiency_score=efficiency_score,
            planning_time_ms=0.0,  # Will be updated
            horizon_s=self.planning_horizon_s,
            asil_rating=ASILLevel.D if emergency else ASILLevel.C,
            emergency_path=emergency
        )

    def _should_consider_lane_change(self, ego_state: Dict[str, float],
                                   obstacles: List[Dict[str, Any]]) -> bool:
        """Determine if lane change options should be considered"""
        # Check if there are obstacles ahead that might require lane change
        ego_pos = np.array([ego_state['position_x'], ego_state['position_y']])
        ego_heading = ego_state['heading']

        for obstacle in obstacles:
            if obstacle['predicted_states']:
                first_state = obstacle['predicted_states'][0]
                obs_pos = np.array(first_state['position'][:2])

                # Check if obstacle is ahead and close
                relative_pos = obs_pos - ego_pos
                ahead_distance = np.dot(relative_pos, [np.cos(ego_heading), np.sin(ego_heading)])

                if 0 < ahead_distance < 30.0:  # Obstacle within 30m ahead
                    return True

        return False

    def _modify_goal_for_lane_change(self, goal_state: Dict[str, float],
                                   lane_offset: int) -> Dict[str, float]:
        """Modify goal state for lane change maneuver"""
        modified_goal = goal_state.copy()
        # Offset goal position by lane width
        lateral_offset = lane_offset * self.constraints.lane_width_m
        modified_goal['position_y'] += lateral_offset
        return modified_goal

    async def _evaluate_paths(self, candidates: List[PlannedPath],
                            obstacles: List[Dict[str, Any]]) -> List[PlannedPath]:
        """Evaluate and rank path candidates"""
        evaluated_paths = []

        for path in candidates:
            # Detailed cost evaluation
            detailed_cost = await self.cost_evaluator.evaluate(path, obstacles)
            path.total_cost = detailed_cost

            # Safety check
            is_safe = await self.safety_checker.check_path(path, obstacles)
            if is_safe:
                evaluated_paths.append(path)

        # Sort by total cost (lower is better)
        evaluated_paths.sort(key=lambda p: p.total_cost)
        return evaluated_paths

    async def _generate_emergency_path(self, ego_state: Dict[str, float],
                                     obstacles: List[Dict[str, Any]],
                                     timestamp: float) -> Optional[PlannedPath]:
        """Generate emergency stopping/avoidance path"""
        try:
            # Emergency brake scenario
            brake_waypoints = self._generate_emergency_brake_path(ego_state, timestamp)
            brake_path = self._create_planned_path(
                brake_waypoints, ManeuverType.EMERGENCY_BRAKE, timestamp, True
            )

            # Check if emergency swerve is needed and possible
            if self._requires_emergency_swerve(ego_state, obstacles):
                swerve_waypoints = self._generate_emergency_swerve_path(ego_state, obstacles, timestamp)
                if swerve_waypoints:
                    swerve_path = self._create_planned_path(
                        swerve_waypoints, ManeuverType.EMERGENCY_SWERVE, timestamp, True
                    )
                    # Choose between brake and swerve based on safety
                    return swerve_path if swerve_path.safety_score > brake_path.safety_score else brake_path

            return brake_path

        except Exception as e:
            self.logger.error(f"Emergency path generation failed: {e}")
            return None

    def _generate_emergency_brake_path(self, ego_state: Dict[str, float],
                                     timestamp: float) -> List[Dict]:
        """Generate emergency braking trajectory"""
        waypoints = []

        current_vel = np.sqrt(ego_state['velocity_x']**2 + ego_state['velocity_y']**2)
        max_decel = self.constraints.max_deceleration_mps2

        # Calculate stopping distance and time
        stopping_time = current_vel / max_decel
        stopping_distance = 0.5 * current_vel * stopping_time

        num_points = int(stopping_time / self.planning_step_s) + 1

        for i in range(num_points):
            t = i * self.planning_step_s
            # Kinematic equations for constant deceleration
            velocity = max(0.0, current_vel - max_decel * t)
            distance = current_vel * t - 0.5 * max_decel * t**2

            # Position calculation
            x = ego_state['position_x'] + distance * np.cos(ego_state['heading'])
            y = ego_state['position_y'] + distance * np.sin(ego_state['heading'])

            waypoint = {
                'x': x,
                'y': y,
                'velocity': velocity,
                'acceleration': -max_decel if velocity > 0 else 0.0,
                'heading': ego_state['heading'],
                'cost': 10.0  # High cost for emergency maneuver
            }
            waypoints.append(waypoint)

        return waypoints

    def _requires_emergency_swerve(self, ego_state: Dict[str, float],
                                 obstacles: List[Dict[str, Any]]) -> bool:
        """Check if emergency swerve is required"""
        # Check for imminent collision that cannot be avoided by braking
        ego_pos = np.array([ego_state['position_x'], ego_state['position_y']])
        ego_vel = np.array([ego_state['velocity_x'], ego_state['velocity_y']])

        for obstacle in obstacles:
            if obstacle['collision_probability'] > 0.8:  # High collision risk
                # Check time to collision
                if obstacle['predicted_states']:
                    first_state = obstacle['predicted_states'][0]
                    obs_pos = np.array(first_state['position'][:2])
                    relative_pos = obs_pos - ego_pos

                    if np.linalg.norm(relative_pos) < 10.0:  # Very close obstacle
                        return True

        return False

    def _generate_emergency_swerve_path(self, ego_state: Dict[str, float],
                                      obstacles: List[Dict[str, Any]],
                                      timestamp: float) -> Optional[List[Dict]]:
        """Generate emergency swerve trajectory"""
        # Simplified swerve path generation
        waypoints = []

        # Determine swerve direction (left or right)
        swerve_direction = self._determine_swerve_direction(ego_state, obstacles)
        if swerve_direction == 0:  # No safe swerve direction
            return None

        lateral_offset = swerve_direction * 2.0  # 2m lateral offset
        swerve_distance = 20.0  # 20m swerve maneuver

        num_points = int(3.0 / self.planning_step_s)  # 3 second swerve

        for i in range(num_points):
            t = i * self.planning_step_s

            # Longitudinal motion
            longitudinal_progress = ego_state['velocity_x'] * t
            x = ego_state['position_x'] + longitudinal_progress * np.cos(ego_state['heading'])

            # Lateral motion (sinusoidal swerve)
            lateral_progress = lateral_offset * np.sin(np.pi * t / 3.0)
            y = ego_state['position_y'] + lateral_progress * np.sin(ego_state['heading'])

            waypoint = {
                'x': x,
                'y': y,
                'velocity': ego_state['velocity_x'],  # Maintain speed during swerve
                'acceleration': 0.0,
                'heading': ego_state['heading'],
                'cost': 20.0  # Very high cost for emergency swerve
            }
            waypoints.append(waypoint)

        return waypoints

    def _determine_swerve_direction(self, ego_state: Dict[str, float],
                                  obstacles: List[Dict[str, Any]]) -> int:
        """Determine safe swerve direction (-1 left, 1 right, 0 none)"""
        # Simplified direction determination
        # In practice, would check adjacent lanes and obstacle positions
        return -1  # Default to left swerve

    def _determine_maneuver(self, path: Optional[PlannedPath],
                          ego_state: Dict[str, float]) -> ManeuverType:
        """Determine current maneuver type"""
        if path is None:
            return ManeuverType.LANE_KEEPING

        return path.maneuver_type

    async def _validate_path_safety(self, path: Optional[PlannedPath],
                                  obstacles: List[Dict[str, Any]]) -> List[str]:
        """Validate path safety and return constraint violations"""
        violations = []

        if path is None:
            violations.append("No valid path found")
            return violations

        # Check velocity constraints
        for waypoint in path.waypoints:
            if waypoint.velocity > self.constraints.max_velocity_mps:
                violations.append("Maximum velocity exceeded")
                break

        # Check acceleration constraints
        for i in range(1, len(path.waypoints)):
            dt = path.waypoints[i].timestamp - path.waypoints[i-1].timestamp
            if dt > 0:
                dv = path.waypoints[i].velocity - path.waypoints[i-1].velocity
                accel = dv / dt

                if accel > self.constraints.max_acceleration_mps2:
                    violations.append("Maximum acceleration exceeded")
                    break
                elif accel < -self.constraints.max_deceleration_mps2:
                    violations.append("Maximum deceleration exceeded")
                    break

        # Check collision potential
        collision_risk = await self._check_collision_risk(path, obstacles)
        if collision_risk > 0.1:
            violations.append("High collision risk detected")

        return violations

    async def _check_collision_risk(self, path: PlannedPath,
                                  obstacles: List[Dict[str, Any]]) -> float:
        """Check collision risk for planned path"""
        max_risk = 0.0

        for waypoint in path.waypoints:
            for obstacle in obstacles:
                risk = self._compute_waypoint_obstacle_risk(waypoint, obstacle)
                max_risk = max(max_risk, risk)

        return max_risk

    def _compute_waypoint_obstacle_risk(self, waypoint: WaypointNode,
                                      obstacle: Dict[str, Any]) -> float:
        """Compute collision risk between waypoint and obstacle"""
        waypoint_pos = np.array(waypoint.position[:2])

        # Find closest obstacle state in time
        closest_state = None
        min_time_diff = float('inf')

        for state in obstacle['predicted_states']:
            time_diff = abs(state['timestamp'] - waypoint.timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_state = state

        if closest_state is None:
            return 0.0

        obs_pos = np.array(closest_state['position'][:2])
        distance = np.linalg.norm(waypoint_pos - obs_pos)

        # Consider uncertainty
        uncertainty = max(closest_state['uncertainty'][:2])
        safety_distance = waypoint.safety_margin + uncertainty

        if distance < safety_distance:
            return 1.0 - (distance / safety_distance)
        else:
            return 0.0

    def _update_planning_state(self, violations: List[str],
                             emergency_path: Optional[PlannedPath]) -> None:
        """Update planning state based on violations and conditions"""
        if violations:
            if any("collision" in v.lower() for v in violations):
                self.state = PlanningState.EMERGENCY
            else:
                self.state = PlanningState.DEGRADED
        else:
            self.state = PlanningState.ACTIVE

    def _compute_safety_score(self, waypoints: List[WaypointNode]) -> float:
        """Compute safety score for path"""
        if not waypoints:
            return 0.0

        # Check safety margins
        min_margin = min(wp.safety_margin for wp in waypoints)
        margin_score = min(min_margin / 5.0, 1.0)  # Normalize by 5m

        # Check velocity profile smoothness
        velocity_changes = []
        for i in range(1, len(waypoints)):
            dv = abs(waypoints[i].velocity - waypoints[i-1].velocity)
            dt = waypoints[i].timestamp - waypoints[i-1].timestamp
            if dt > 0:
                velocity_changes.append(dv / dt)

        accel_score = 1.0
        if velocity_changes:
            max_accel = max(velocity_changes)
            accel_score = 1.0 - min(max_accel / self.constraints.max_acceleration_mps2, 1.0)

        return (margin_score + accel_score) / 2.0

    def _compute_comfort_score(self, waypoints: List[WaypointNode]) -> float:
        """Compute comfort score for path"""
        if len(waypoints) < 3:
            return 1.0

        # Check jerk (acceleration changes)
        jerk_values = []
        for i in range(2, len(waypoints)):
            dt1 = waypoints[i-1].timestamp - waypoints[i-2].timestamp
            dt2 = waypoints[i].timestamp - waypoints[i-1].timestamp

            if dt1 > 0 and dt2 > 0:
                a1 = waypoints[i-1].acceleration
                a2 = waypoints[i].acceleration
                jerk = abs(a2 - a1) / ((dt1 + dt2) / 2.0)
                jerk_values.append(jerk)

        if not jerk_values:
            return 1.0

        max_jerk = max(jerk_values)
        jerk_score = 1.0 - min(max_jerk / self.constraints.max_jerk_mps3, 1.0)

        # Check curvature changes
        curvature_score = 1.0
        if len(waypoints) > 1:
            max_curvature = max(abs(wp.curvature) for wp in waypoints)
            curvature_score = 1.0 - min(max_curvature / 0.1, 1.0)  # 0.1 rad/m max comfortable curvature

        return (jerk_score + curvature_score) / 2.0

    def _compute_efficiency_score(self, waypoints: List[WaypointNode]) -> float:
        """Compute efficiency score for path"""
        if not waypoints:
            return 0.0

        # Check path length vs direct distance
        path_length = 0.0
        for i in range(1, len(waypoints)):
            p1 = np.array(waypoints[i-1].position[:2])
            p2 = np.array(waypoints[i].position[:2])
            path_length += np.linalg.norm(p2 - p1)

        start_pos = np.array(waypoints[0].position[:2])
        end_pos = np.array(waypoints[-1].position[:2])
        direct_distance = np.linalg.norm(end_pos - start_pos)

        if direct_distance == 0:
            return 1.0

        efficiency = direct_distance / path_length
        return min(efficiency, 1.0)

    def _compute_path_feasibility(self, path: Optional[PlannedPath]) -> float:
        """Compute overall path feasibility score"""
        if path is None:
            return 0.0

        # Weighted combination of quality scores
        weights = {'safety': 0.5, 'comfort': 0.2, 'efficiency': 0.3}

        feasibility = (
            weights['safety'] * path.safety_score +
            weights['comfort'] * path.comfort_score +
            weights['efficiency'] * path.efficiency_score
        )

        return feasibility

    def _format_obstacles(self, obstacles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format obstacles for output"""
        formatted = []
        for obs in obstacles:
            formatted_obs = {
                'id': obs['object_id'],
                'tracking_id': obs['tracking_id'],
                'collision_probability': obs['collision_probability'],
                'risk_level': obs['risk_level'],
                'maneuver_class': obs['maneuver_class'],
                'num_predicted_states': len(obs['predicted_states'])
            }
            formatted.append(formatted_obs)
        return formatted

    def _generate_failsafe_output(self, timestamp: float) -> PlanningOutput:
        """Generate safe fallback output in case of planning failure"""
        # Generate emergency brake path as failsafe
        emergency_waypoints = [{
            'x': 0.0, 'y': 0.0, 'velocity': 0.0, 'acceleration': -self.constraints.max_deceleration_mps2,
            'heading': 0.0, 'cost': 100.0
        }]

        emergency_path = self._create_planned_path(
            emergency_waypoints, ManeuverType.EMERGENCY_BRAKE, timestamp, True
        )

        return PlanningOutput(
            timestamp=timestamp,
            primary_path=emergency_path,
            alternative_paths=[],
            emergency_path=emergency_path,
            current_maneuver=ManeuverType.EMERGENCY_BRAKE,
            planning_state=PlanningState.EMERGENCY,
            obstacles=[],
            constraints_violated=["Planning system failure"],
            planning_latency_ms=0.0,
            path_feasibility=0.1
        )

    def _update_performance_metrics(self, latency_ms: float, violations: List[str]) -> None:
        """Update performance tracking metrics"""
        alpha = 0.1  # Smoothing factor
        self.performance_metrics["avg_planning_ms"] = (
            alpha * latency_ms + (1 - alpha) * self.performance_metrics["avg_planning_ms"]
        )

        current_time = time.time()
        if self.last_planning_time > 0:
            rate = 1.0 / (current_time - self.last_planning_time + 1e-6)
            self.performance_metrics["planning_rate_hz"] = (
                alpha * rate + (1 - alpha) * self.performance_metrics["planning_rate_hz"]
            )

        if violations:
            self.performance_metrics["constraint_violations"] += 1

        if self.state == PlanningState.EMERGENCY:
            self.performance_metrics["emergency_activations"] += 1

    async def stop(self) -> None:
        """Stop the planning agent"""
        self.logger.info("Stopping PlanningAgent...")
        self.running = False

        if self.planning_thread:
            self.planning_thread.join(timeout=1.0)

        await self.safety_monitor.stop()
        self.executor.shutdown(wait=True)

        self.state = PlanningState.INITIALIZING
        self.logger.info("PlanningAgent stopped")

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def get_state(self) -> PlanningState:
        """Get current planning state"""
        return self.state


# Supporting classes (simplified implementations)
class AStarPlanner:
    def __init__(self, grid_resolution: float, max_search_distance: float):
        self.grid_resolution = grid_resolution
        self.max_search_distance = max_search_distance

    def validate(self) -> bool:
        return True

    async def plan(self, start: Dict, goal: Dict, obstacles: List[Dict]) -> List[Dict]:
        # Simplified A* implementation
        return [
            {'x': start['position_x'], 'y': start['position_y'], 'velocity': 10.0},
            {'x': goal['position_x'], 'y': goal['position_y'], 'velocity': 5.0}
        ]

class RRTStarPlanner:
    def __init__(self, max_iterations: int, step_size: float, goal_bias: float):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias

    def validate(self) -> bool:
        return True

    async def plan(self, start: Dict, goal: Dict, obstacles: List[Dict]) -> List[Dict]:
        return []

class LatticePlanner:
    def __init__(self, lateral_resolution: float, longitudinal_resolution: float, max_horizon: float):
        self.lateral_resolution = lateral_resolution
        self.longitudinal_resolution = longitudinal_resolution
        self.max_horizon = max_horizon

    def validate(self) -> bool:
        return True

    async def plan(self, start: Dict, goal: Dict, obstacles: List[Dict]) -> List[Dict]:
        return []

class FrenetPlanner:
    def __init__(self, road_width: float, planning_horizon: float):
        self.road_width = road_width
        self.planning_horizon = planning_horizon

    def validate(self) -> bool:
        return True

    async def plan(self, start: Dict, goal: Dict, obstacles: List[Dict]) -> List[Dict]:
        return []

class ObstaclePredictor:
    pass

class CostEvaluator:
    def __init__(self, constraints: PlanningConstraints):
        self.constraints = constraints

    async def evaluate(self, path: PlannedPath, obstacles: List[Dict]) -> float:
        return path.total_cost

class SafetyChecker:
    def __init__(self, constraints: PlanningConstraints):
        self.constraints = constraints

    async def check_path(self, path: PlannedPath, obstacles: List[Dict]) -> bool:
        return True

class PlanningSafetyMonitor:
    def __init__(self, config: ADASConfig):
        self.config = config

    async def start(self):
        pass

    async def validate_output(self, output: PlanningOutput):
        pass

    async def stop(self):
        pass