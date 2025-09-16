"""
ADAS Phase 7 - Path Planning Module
Automotive-grade optimal path generation with A* and dynamic programming
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize


class PathPlanningMode(Enum):
    """Path planning operation modes for different driving scenarios"""
    HIGHWAY_CRUISING = "highway_cruising"
    URBAN_NAVIGATION = "urban_navigation"
    PARKING = "parking"
    EMERGENCY_AVOIDANCE = "emergency_avoidance"
    LANE_CHANGE = "lane_change"


@dataclass
class WayPoint:
    """Single waypoint in a path"""
    x: float
    y: float
    heading: float
    velocity: float
    curvature: float = 0.0
    timestamp: float = 0.0
    lane_id: int = 0
    cost: float = 0.0


@dataclass
class Path:
    """Complete path representation"""
    waypoints: List[WayPoint]
    total_cost: float
    total_time: float
    total_distance: float
    safety_score: float
    comfort_score: float
    fuel_efficiency: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class VehicleConstraints:
    """Vehicle-specific physical constraints"""
    max_speed: float = 50.0  # m/s
    max_acceleration: float = 4.0  # m/s^2
    max_deceleration: float = 8.0  # m/s^2
    max_steering_angle: float = 0.6  # radians
    max_curvature: float = 0.2  # 1/m
    wheelbase: float = 2.8  # meters
    width: float = 1.8  # meters
    length: float = 4.5  # meters
    turning_radius: float = 5.5  # meters


@dataclass
class Obstacle:
    """Dynamic or static obstacle representation"""
    center: np.ndarray
    dimensions: np.ndarray  # [width, height, length]
    velocity: np.ndarray
    heading: float
    timestamp: float
    confidence: float = 1.0
    obstacle_type: str = "unknown"  # car, pedestrian, cyclist, static


class GridNode:
    """Node for A* grid-based pathfinding"""

    def __init__(self, x: int, y: int, theta: int = 0):
        self.x = x
        self.y = y
        self.theta = theta  # Discretized heading
        self.g_cost = float('inf')
        self.h_cost = 0.0
        self.f_cost = float('inf')
        self.parent = None
        self.is_obstacle = False

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __hash__(self):
        return hash((self.x, self.y, self.theta))


class AStarPlanner:
    """A* pathfinding with automotive kinematic constraints"""

    def __init__(self,
                 grid_resolution: float = 0.5,  # meters per cell
                 theta_resolution: int = 16):    # heading discretization

        self.grid_resolution = grid_resolution
        self.theta_resolution = theta_resolution
        self.theta_step = 2 * np.pi / theta_resolution

        # Cost weights for multi-objective optimization
        self.distance_weight = 1.0
        self.curvature_weight = 2.0
        self.obstacle_weight = 10.0
        self.lane_change_weight = 3.0
        self.speed_weight = 0.5

        self.logger = logging.getLogger(__name__)

    def plan_path(self,
                  start: WayPoint,
                  goal: WayPoint,
                  obstacles: List[Obstacle],
                  vehicle_constraints: VehicleConstraints,
                  grid_bounds: Tuple[float, float, float, float] = None) -> Optional[Path]:
        """
        Plan optimal path using A* with kinematic constraints

        Args:
            start: Starting waypoint
            goal: Goal waypoint
            obstacles: List of obstacles to avoid
            vehicle_constraints: Vehicle physical limitations
            grid_bounds: (min_x, max_x, min_y, max_y) grid boundaries

        Returns:
            Optimal path or None if no path found
        """

        start_time = time.time()

        # Set default grid bounds if not provided
        if grid_bounds is None:
            grid_bounds = (
                min(start.x, goal.x) - 50,
                max(start.x, goal.x) + 50,
                min(start.y, goal.y) - 50,
                max(start.y, goal.y) + 50
            )

        # Convert to grid coordinates
        start_node = self._world_to_grid(start)
        goal_node = self._world_to_grid(goal)

        # Build obstacle grid
        obstacle_grid = self._build_obstacle_grid(obstacles, grid_bounds, vehicle_constraints)

        # A* search
        open_set = []
        closed_set = set()

        start_node.g_cost = 0
        start_node.h_cost = self._heuristic(start_node, goal_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        heapq.heappush(open_set, start_node)

        while open_set:
            current = heapq.heappop(open_set)

            if self._is_goal(current, goal_node):
                path = self._reconstruct_path(current, start, goal, vehicle_constraints)
                planning_time = time.time() - start_time
                self.logger.info(f"Path found in {planning_time:.3f}s")
                return path

            closed_set.add(current)

            # Generate successor nodes with kinematic constraints
            successors = self._get_kinematic_successors(current, vehicle_constraints)

            for successor in successors:
                if successor in closed_set:
                    continue

                # Check obstacle collision
                if self._is_collision(successor, obstacle_grid, grid_bounds):
                    continue

                # Calculate cost
                tentative_g = current.g_cost + self._motion_cost(current, successor, vehicle_constraints)

                if tentative_g < successor.g_cost:
                    successor.parent = current
                    successor.g_cost = tentative_g
                    successor.h_cost = self._heuristic(successor, goal_node)
                    successor.f_cost = successor.g_cost + successor.h_cost

                    if successor not in open_set:
                        heapq.heappush(open_set, successor)

        self.logger.warning("No path found within time limit")
        return None

    def _world_to_grid(self, waypoint: WayPoint) -> GridNode:
        """Convert world coordinates to grid node"""
        grid_x = int(waypoint.x / self.grid_resolution)
        grid_y = int(waypoint.y / self.grid_resolution)
        grid_theta = int(waypoint.heading / self.theta_step) % self.theta_resolution
        return GridNode(grid_x, grid_y, grid_theta)

    def _grid_to_world(self, node: GridNode) -> Tuple[float, float, float]:
        """Convert grid node to world coordinates"""
        world_x = node.x * self.grid_resolution
        world_y = node.y * self.grid_resolution
        world_theta = node.theta * self.theta_step
        return world_x, world_y, world_theta

    def _build_obstacle_grid(self,
                           obstacles: List[Obstacle],
                           grid_bounds: Tuple[float, float, float, float],
                           vehicle_constraints: VehicleConstraints) -> np.ndarray:
        """Build binary obstacle grid with safety margins"""

        min_x, max_x, min_y, max_y = grid_bounds
        grid_width = int((max_x - min_x) / self.grid_resolution)
        grid_height = int((max_y - min_y) / self.grid_resolution)

        obstacle_grid = np.zeros((grid_height, grid_width), dtype=bool)

        # Safety margin around obstacles
        safety_margin = max(vehicle_constraints.width, vehicle_constraints.length) / 2 + 1.0

        for obstacle in obstacles:
            # Predict future obstacle position (simple linear prediction)
            future_position = obstacle.center + obstacle.velocity * 2.0  # 2 second prediction

            # Create inflated obstacle boundary
            half_width = obstacle.dimensions[0] / 2 + safety_margin
            half_height = obstacle.dimensions[1] / 2 + safety_margin

            # Convert to grid coordinates
            center_x = int((future_position[0] - min_x) / self.grid_resolution)
            center_y = int((future_position[1] - min_y) / self.grid_resolution)

            width_cells = int(half_width / self.grid_resolution)
            height_cells = int(half_height / self.grid_resolution)

            # Mark obstacle cells
            for dx in range(-width_cells, width_cells + 1):
                for dy in range(-height_cells, height_cells + 1):
                    grid_x = center_x + dx
                    grid_y = center_y + dy

                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        obstacle_grid[grid_y, grid_x] = True

        return obstacle_grid

    def _get_kinematic_successors(self,
                                current: GridNode,
                                vehicle_constraints: VehicleConstraints) -> List[GridNode]:
        """Generate kinematically feasible successor nodes"""

        successors = []

        # Bicycle model motion primitives
        dt = 1.0  # time step
        wheelbase = vehicle_constraints.wheelbase

        # Discretized steering angles
        max_steering = vehicle_constraints.max_steering_angle
        steering_angles = np.linspace(-max_steering, max_steering, 5)

        # Discretized velocities
        velocities = [2.0, 5.0, 10.0]  # m/s

        for steering_angle in steering_angles:
            for velocity in velocities:
                # Bicycle model kinematics
                current_x, current_y, current_theta = self._grid_to_world(current)

                new_x = current_x + velocity * np.cos(current_theta) * dt
                new_y = current_y + velocity * np.sin(current_theta) * dt
                new_theta = current_theta + (velocity / wheelbase) * np.tan(steering_angle) * dt

                # Create successor node
                successor_waypoint = WayPoint(new_x, new_y, new_theta, velocity)
                successor = self._world_to_grid(successor_waypoint)

                # Check kinematic constraints
                curvature = abs(np.tan(steering_angle) / wheelbase)
                if curvature <= vehicle_constraints.max_curvature:
                    successors.append(successor)

        return successors

    def _heuristic(self, node1: GridNode, node2: GridNode) -> float:
        """Admissible heuristic function"""
        x1, y1, theta1 = self._grid_to_world(node1)
        x2, y2, theta2 = self._grid_to_world(node2)

        # Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Angular difference penalty
        angle_diff = abs(theta2 - theta1)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        return distance + 2.0 * angle_diff

    def _motion_cost(self,
                   current: GridNode,
                   successor: GridNode,
                   vehicle_constraints: VehicleConstraints) -> float:
        """Calculate motion cost between nodes"""

        x1, y1, theta1 = self._grid_to_world(current)
        x2, y2, theta2 = self._grid_to_world(successor)

        # Distance cost
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_cost = self.distance_weight * distance

        # Curvature cost (penalize sharp turns)
        angle_change = abs(theta2 - theta1)
        angle_change = min(angle_change, 2 * np.pi - angle_change)
        curvature_cost = self.curvature_weight * angle_change

        # Speed cost (encourage appropriate speeds)
        speed_cost = self.speed_weight * abs(10.0 - 8.0)  # target 8 m/s

        return distance_cost + curvature_cost + speed_cost

    def _is_collision(self,
                    node: GridNode,
                    obstacle_grid: np.ndarray,
                    grid_bounds: Tuple[float, float, float, float]) -> bool:
        """Check collision with obstacles"""

        min_x, max_x, min_y, max_y = grid_bounds
        world_x, world_y, _ = self._grid_to_world(node)

        # Convert to grid coordinates
        grid_x = int((world_x - min_x) / self.grid_resolution)
        grid_y = int((world_y - min_y) / self.grid_resolution)

        # Check bounds
        if (grid_x < 0 or grid_x >= obstacle_grid.shape[1] or
            grid_y < 0 or grid_y >= obstacle_grid.shape[0]):
            return True

        return obstacle_grid[grid_y, grid_x]

    def _is_goal(self, current: GridNode, goal: GridNode, tolerance: float = 2.0) -> bool:
        """Check if current node is close enough to goal"""
        x1, y1, _ = self._grid_to_world(current)
        x2, y2, _ = self._grid_to_world(goal)

        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance <= tolerance

    def _reconstruct_path(self,
                        goal_node: GridNode,
                        start: WayPoint,
                        goal: WayPoint,
                        vehicle_constraints: VehicleConstraints) -> Path:
        """Reconstruct path from goal to start"""

        waypoints = []
        current = goal_node

        while current is not None:
            x, y, theta = self._grid_to_world(current)
            waypoint = WayPoint(
                x=x, y=y, heading=theta,
                velocity=8.0,  # Default velocity
                timestamp=len(waypoints) * 0.1
            )
            waypoints.append(waypoint)
            current = current.parent

        waypoints.reverse()

        # Smooth path and calculate metrics
        smoothed_waypoints = self._smooth_path(waypoints, vehicle_constraints)

        return Path(
            waypoints=smoothed_waypoints,
            total_cost=goal_node.g_cost,
            total_time=len(smoothed_waypoints) * 0.1,
            total_distance=self._calculate_path_distance(smoothed_waypoints),
            safety_score=0.8,  # TODO: Calculate based on obstacles
            comfort_score=0.7,  # TODO: Calculate based on curvature
            fuel_efficiency=0.75  # TODO: Calculate based on acceleration profile
        )

    def _smooth_path(self,
                   waypoints: List[WayPoint],
                   vehicle_constraints: VehicleConstraints) -> List[WayPoint]:
        """Apply path smoothing for comfort"""

        if len(waypoints) < 3:
            return waypoints

        smoothed = [waypoints[0]]  # Keep start point

        # Simple moving average smoothing
        for i in range(1, len(waypoints) - 1):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i+1]

            # Smooth position
            smooth_x = 0.25 * prev_wp.x + 0.5 * curr_wp.x + 0.25 * next_wp.x
            smooth_y = 0.25 * prev_wp.y + 0.5 * curr_wp.y + 0.25 * next_wp.y

            # Smooth heading
            smooth_heading = curr_wp.heading  # Keep original for now

            # Calculate curvature
            curvature = self._calculate_curvature(prev_wp, curr_wp, next_wp)

            smooth_wp = WayPoint(
                x=smooth_x,
                y=smooth_y,
                heading=smooth_heading,
                velocity=min(curr_wp.velocity,
                           np.sqrt(vehicle_constraints.max_acceleration / max(abs(curvature), 0.01))),
                curvature=curvature,
                timestamp=curr_wp.timestamp
            )

            smoothed.append(smooth_wp)

        smoothed.append(waypoints[-1])  # Keep goal point
        return smoothed

    def _calculate_curvature(self, wp1: WayPoint, wp2: WayPoint, wp3: WayPoint) -> float:
        """Calculate path curvature at waypoint"""

        # Use three-point circle fitting
        x1, y1 = wp1.x, wp1.y
        x2, y2 = wp2.x, wp2.y
        x3, y3 = wp3.x, wp3.y

        # Calculate circle radius
        denominator = 2 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

        if abs(denominator) < 1e-6:
            return 0.0

        radius = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * \
                np.sqrt((x2 - x3)**2 + (y2 - y3)**2) * \
                np.sqrt((x3 - x1)**2 + (y3 - y1)**2) / abs(denominator)

        return 1.0 / max(radius, 0.1)  # Avoid division by zero

    def _calculate_path_distance(self, waypoints: List[WayPoint]) -> float:
        """Calculate total path distance"""

        total_distance = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            total_distance += np.sqrt(dx**2 + dy**2)

        return total_distance


class DynamicProgrammingPlanner:
    """Dynamic programming approach for optimal path planning"""

    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        self.logger = logging.getLogger(__name__)

    def plan_optimal_path(self,
                        start: WayPoint,
                        goal: WayPoint,
                        obstacles: List[Obstacle],
                        vehicle_constraints: VehicleConstraints,
                        time_horizon: float = 10.0) -> Optional[Path]:
        """
        Plan optimal path using dynamic programming

        Optimizes for multiple objectives:
        - Minimum travel time
        - Maximum safety
        - Maximum comfort
        - Minimum fuel consumption
        """

        # Discretize state space
        x_range = np.arange(min(start.x, goal.x) - 20,
                           max(start.x, goal.x) + 20,
                           self.resolution)
        y_range = np.arange(min(start.y, goal.y) - 20,
                           max(start.y, goal.y) + 20,
                           self.resolution)

        # Value function table
        V = np.full((len(x_range), len(y_range)), np.inf)
        policy = np.zeros((len(x_range), len(y_range), 2))

        # Goal state
        goal_i = np.argmin(np.abs(x_range - goal.x))
        goal_j = np.argmin(np.abs(y_range - goal.y))
        V[goal_i, goal_j] = 0.0

        # Value iteration
        for iteration in range(100):  # Maximum iterations
            V_old = V.copy()

            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    if i == goal_i and j == goal_j:
                        continue

                    # Current position
                    x = x_range[i]
                    y = y_range[j]

                    # Skip if obstacle
                    if self._is_obstacle_at_position(np.array([x, y]), obstacles):
                        continue

                    # Find best action
                    min_cost = np.inf
                    best_action = np.array([0.0, 0.0])

                    # Action space (velocity vectors)
                    actions = [
                        np.array([1.0, 0.0]),   # Forward
                        np.array([0.0, 1.0]),   # Right
                        np.array([-1.0, 0.0]),  # Backward
                        np.array([0.0, -1.0]),  # Left
                        np.array([0.707, 0.707]),   # Diagonal
                        np.array([-0.707, 0.707]),  # Diagonal
                        np.array([0.707, -0.707]),  # Diagonal
                        np.array([-0.707, -0.707]) # Diagonal
                    ]

                    for action in actions:
                        next_pos = np.array([x, y]) + action * self.resolution

                        # Find next state indices
                        next_i = np.argmin(np.abs(x_range - next_pos[0]))
                        next_j = np.argmin(np.abs(y_range - next_pos[1]))

                        # Check bounds
                        if (0 <= next_i < len(x_range) and
                            0 <= next_j < len(y_range)):

                            # Calculate cost
                            movement_cost = np.linalg.norm(action) * self.resolution
                            obstacle_cost = self._calculate_obstacle_cost(next_pos, obstacles)
                            comfort_cost = self._calculate_comfort_cost(action, vehicle_constraints)

                            total_cost = movement_cost + obstacle_cost + comfort_cost + V[next_i, next_j]

                            if total_cost < min_cost:
                                min_cost = total_cost
                                best_action = action

                    V[i, j] = min_cost
                    policy[i, j] = best_action

            # Check convergence
            if np.max(np.abs(V - V_old)) < 1e-3:
                self.logger.info(f"DP converged in {iteration + 1} iterations")
                break

        # Extract optimal path
        path_waypoints = self._extract_dp_path(start, goal, x_range, y_range, policy, vehicle_constraints)

        if not path_waypoints:
            return None

        return Path(
            waypoints=path_waypoints,
            total_cost=V[np.argmin(np.abs(x_range - start.x)),
                         np.argmin(np.abs(y_range - start.y))],
            total_time=len(path_waypoints) * 0.1,
            total_distance=self._calculate_path_distance(path_waypoints),
            safety_score=0.9,
            comfort_score=0.8,
            fuel_efficiency=0.85
        )

    def _is_obstacle_at_position(self, position: np.ndarray, obstacles: List[Obstacle]) -> bool:
        """Check if position collides with obstacles"""

        for obstacle in obstacles:
            distance = np.linalg.norm(position - obstacle.center[:2])
            obstacle_radius = np.max(obstacle.dimensions[:2]) / 2

            if distance < obstacle_radius + 1.0:  # Safety margin
                return True

        return False

    def _calculate_obstacle_cost(self, position: np.ndarray, obstacles: List[Obstacle]) -> float:
        """Calculate cost based on proximity to obstacles"""

        min_distance = float('inf')

        for obstacle in obstacles:
            distance = np.linalg.norm(position - obstacle.center[:2])
            obstacle_radius = np.max(obstacle.dimensions[:2]) / 2
            effective_distance = distance - obstacle_radius

            if effective_distance < min_distance:
                min_distance = effective_distance

        # Exponential penalty for close obstacles
        if min_distance < 5.0:
            return 10.0 * np.exp(-min_distance / 2.0)

        return 0.0

    def _calculate_comfort_cost(self, action: np.ndarray, vehicle_constraints: VehicleConstraints) -> float:
        """Calculate comfort cost based on acceleration"""

        acceleration = np.linalg.norm(action)
        max_comfort_accel = 2.0  # m/s^2

        if acceleration > max_comfort_accel:
            return 2.0 * (acceleration - max_comfort_accel)**2

        return 0.0

    def _extract_dp_path(self,
                       start: WayPoint,
                       goal: WayPoint,
                       x_range: np.ndarray,
                       y_range: np.ndarray,
                       policy: np.ndarray,
                       vehicle_constraints: VehicleConstraints) -> List[WayPoint]:
        """Extract optimal path from DP policy"""

        path_waypoints = []

        # Start from start position
        current_pos = np.array([start.x, start.y])
        current_time = 0.0

        for step in range(200):  # Maximum steps
            # Add current waypoint
            waypoint = WayPoint(
                x=current_pos[0],
                y=current_pos[1],
                heading=np.arctan2(current_pos[1] - start.y, current_pos[0] - start.x),
                velocity=5.0,  # Constant velocity for simplicity
                timestamp=current_time
            )
            path_waypoints.append(waypoint)

            # Check if reached goal
            if np.linalg.norm(current_pos - np.array([goal.x, goal.y])) < self.resolution:
                break

            # Get action from policy
            i = np.argmin(np.abs(x_range - current_pos[0]))
            j = np.argmin(np.abs(y_range - current_pos[1]))

            if i < 0 or i >= len(x_range) or j < 0 or j >= len(y_range):
                break

            action = policy[i, j]

            # Update position
            current_pos += action * self.resolution
            current_time += 0.1

        return path_waypoints


class PathPlanner:
    """Main path planning system for ADAS Phase 7"""

    def __init__(self,
                 planning_mode: PathPlanningMode = PathPlanningMode.HIGHWAY_CRUISING,
                 update_frequency: float = 10.0):  # Hz

        self.planning_mode = planning_mode
        self.update_frequency = update_frequency
        self.logger = logging.getLogger(__name__)

        # Initialize planners
        self.astar_planner = AStarPlanner()
        self.dp_planner = DynamicProgrammingPlanner()

        # Current path and planning state
        self.current_path: Optional[Path] = None
        self.path_timestamp = 0.0
        self.replanning_threshold = 2.0  # seconds

        # Performance metrics
        self.planning_times = []
        self.success_rate = 0.0

    def plan_path(self,
                  start: WayPoint,
                  goal: WayPoint,
                  obstacles: List[Obstacle],
                  vehicle_constraints: VehicleConstraints,
                  road_network: Optional[Dict] = None) -> Optional[Path]:
        """
        Plan optimal path based on current mode and conditions

        Args:
            start: Current vehicle position and state
            goal: Desired goal position
            obstacles: Dynamic and static obstacles
            vehicle_constraints: Vehicle physical limitations
            road_network: Optional road network information

        Returns:
            Optimal path or None if planning fails
        """

        start_time = time.time()

        try:
            # Choose planner based on mode and conditions
            if self.planning_mode == PathPlanningMode.EMERGENCY_AVOIDANCE:
                # Use faster A* for emergency situations
                path = self.astar_planner.plan_path(start, goal, obstacles, vehicle_constraints)

            elif self.planning_mode == PathPlanningMode.HIGHWAY_CRUISING:
                # Use DP for optimal highway paths
                path = self.dp_planner.plan_optimal_path(start, goal, obstacles, vehicle_constraints)

                # Fallback to A* if DP fails
                if path is None:
                    path = self.astar_planner.plan_path(start, goal, obstacles, vehicle_constraints)

            elif self.planning_mode == PathPlanningMode.PARKING:
                # Use A* with tight constraints for parking
                path = self.astar_planner.plan_path(start, goal, obstacles, vehicle_constraints)

                # Additional parking-specific validation
                if path:
                    path = self._validate_parking_path(path, vehicle_constraints)

            else:
                # Default to A* for other modes
                path = self.astar_planner.plan_path(start, goal, obstacles, vehicle_constraints)

            # Post-process path
            if path:
                path = self._post_process_path(path, vehicle_constraints)
                self.current_path = path
                self.path_timestamp = time.time()

            # Update metrics
            planning_time = time.time() - start_time
            self.planning_times.append(planning_time)

            # Keep only last 100 planning times
            if len(self.planning_times) > 100:
                self.planning_times.pop(0)

            self.logger.info(f"Path planning completed in {planning_time:.3f}s")
            return path

        except Exception as e:
            self.logger.error(f"Path planning failed: {e}")
            return None

    def update_path(self,
                   current_position: WayPoint,
                   obstacles: List[Obstacle],
                   vehicle_constraints: VehicleConstraints) -> bool:
        """
        Update current path based on new information

        Returns:
            True if replanning was triggered
        """

        if self.current_path is None:
            return False

        # Check if replanning is needed
        current_time = time.time()
        time_since_planning = current_time - self.path_timestamp

        replan_needed = (
            time_since_planning > self.replanning_threshold or
            self._path_blocked_by_obstacles(self.current_path, obstacles) or
            self._significant_deviation_from_path(current_position, self.current_path)
        )

        if replan_needed:
            self.logger.info("Replanning triggered")

            # Find current goal from existing path
            if len(self.current_path.waypoints) > 10:
                goal = self.current_path.waypoints[-1]

                # Replan from current position
                new_path = self.plan_path(current_position, goal, obstacles, vehicle_constraints)

                if new_path:
                    self.current_path = new_path
                    return True

        return False

    def _validate_parking_path(self, path: Path, vehicle_constraints: VehicleConstraints) -> Path:
        """Additional validation for parking maneuvers"""

        # Check maximum curvature for parking
        max_parking_curvature = 1.0 / vehicle_constraints.turning_radius

        valid_waypoints = []
        for waypoint in path.waypoints:
            if abs(waypoint.curvature) <= max_parking_curvature:
                valid_waypoints.append(waypoint)
            else:
                # Modify waypoint to satisfy parking constraints
                modified_waypoint = WayPoint(
                    x=waypoint.x,
                    y=waypoint.y,
                    heading=waypoint.heading,
                    velocity=min(waypoint.velocity, 2.0),  # Low speed for parking
                    curvature=np.sign(waypoint.curvature) * max_parking_curvature,
                    timestamp=waypoint.timestamp
                )
                valid_waypoints.append(modified_waypoint)

        # Update path with modified waypoints
        path.waypoints = valid_waypoints
        return path

    def _post_process_path(self, path: Path, vehicle_constraints: VehicleConstraints) -> Path:
        """Post-process path for automotive requirements"""

        # Velocity profile optimization
        optimized_waypoints = self._optimize_velocity_profile(path.waypoints, vehicle_constraints)

        # Safety checks
        safe_waypoints = self._apply_safety_checks(optimized_waypoints, vehicle_constraints)

        # Recalculate path metrics
        path.waypoints = safe_waypoints
        path.total_distance = self.astar_planner._calculate_path_distance(safe_waypoints)
        path.total_time = safe_waypoints[-1].timestamp if safe_waypoints else 0.0

        return path

    def _optimize_velocity_profile(self,
                                 waypoints: List[WayPoint],
                                 vehicle_constraints: VehicleConstraints) -> List[WayPoint]:
        """Optimize velocity profile for smooth acceleration"""

        if len(waypoints) < 2:
            return waypoints

        optimized = [waypoints[0]]  # Keep first waypoint

        for i in range(1, len(waypoints)):
            prev_wp = optimized[-1]
            curr_wp = waypoints[i]

            # Calculate time step
            dt = curr_wp.timestamp - prev_wp.timestamp
            if dt <= 0:
                dt = 0.1

            # Calculate distance
            dx = curr_wp.x - prev_wp.x
            dy = curr_wp.y - prev_wp.y
            distance = np.sqrt(dx**2 + dy**2)

            # Calculate curvature-limited velocity
            max_curve_velocity = np.sqrt(
                vehicle_constraints.max_acceleration / max(abs(curr_wp.curvature), 0.01)
            )

            # Calculate acceleration-limited velocity
            max_accel_velocity = prev_wp.velocity + vehicle_constraints.max_acceleration * dt
            min_decel_velocity = prev_wp.velocity - vehicle_constraints.max_deceleration * dt

            # Choose safe velocity
            safe_velocity = min(
                curr_wp.velocity,
                max_curve_velocity,
                max_accel_velocity,
                vehicle_constraints.max_speed
            )
            safe_velocity = max(safe_velocity, min_decel_velocity, 0.1)

            # Create optimized waypoint
            optimized_wp = WayPoint(
                x=curr_wp.x,
                y=curr_wp.y,
                heading=curr_wp.heading,
                velocity=safe_velocity,
                curvature=curr_wp.curvature,
                timestamp=curr_wp.timestamp
            )

            optimized.append(optimized_wp)

        return optimized

    def _apply_safety_checks(self,
                           waypoints: List[WayPoint],
                           vehicle_constraints: VehicleConstraints) -> List[WayPoint]:
        """Apply final safety checks to path"""

        safe_waypoints = []

        for waypoint in waypoints:
            # Clamp velocity to safe limits
            safe_velocity = max(0.1, min(waypoint.velocity, vehicle_constraints.max_speed))

            # Clamp curvature to vehicle limits
            safe_curvature = np.clip(waypoint.curvature,
                                   -vehicle_constraints.max_curvature,
                                   vehicle_constraints.max_curvature)

            safe_waypoint = WayPoint(
                x=waypoint.x,
                y=waypoint.y,
                heading=waypoint.heading,
                velocity=safe_velocity,
                curvature=safe_curvature,
                timestamp=waypoint.timestamp
            )

            safe_waypoints.append(safe_waypoint)

        return safe_waypoints

    def _path_blocked_by_obstacles(self, path: Path, obstacles: List[Obstacle]) -> bool:
        """Check if current path is blocked by obstacles"""

        # Simple check: if any obstacle is close to path waypoints
        for waypoint in path.waypoints[:10]:  # Check next 10 waypoints
            waypoint_pos = np.array([waypoint.x, waypoint.y])

            for obstacle in obstacles:
                distance = np.linalg.norm(waypoint_pos - obstacle.center[:2])
                obstacle_radius = np.max(obstacle.dimensions[:2]) / 2

                if distance < obstacle_radius + 2.0:  # Safety margin
                    return True

        return False

    def _significant_deviation_from_path(self,
                                       current_position: WayPoint,
                                       path: Path) -> bool:
        """Check if vehicle has deviated significantly from planned path"""

        if not path.waypoints:
            return True

        # Find closest waypoint on path
        current_pos = np.array([current_position.x, current_position.y])
        min_distance = float('inf')

        for waypoint in path.waypoints[:20]:  # Check next 20 waypoints
            waypoint_pos = np.array([waypoint.x, waypoint.y])
            distance = np.linalg.norm(current_pos - waypoint_pos)

            if distance < min_distance:
                min_distance = distance

        # Deviation threshold
        return min_distance > 3.0  # meters

    def get_planning_metrics(self) -> Dict[str, float]:
        """Get current planning performance metrics"""

        avg_planning_time = np.mean(self.planning_times) if self.planning_times else 0.0
        max_planning_time = np.max(self.planning_times) if self.planning_times else 0.0

        return {
            'average_planning_time_ms': avg_planning_time * 1000,
            'max_planning_time_ms': max_planning_time * 1000,
            'planning_frequency_hz': self.update_frequency,
            'current_path_length': len(self.current_path.waypoints) if self.current_path else 0,
            'path_age_seconds': time.time() - self.path_timestamp if self.path_timestamp else 0
        }


if __name__ == "__main__":
    # Example usage for automotive testing
    logging.basicConfig(level=logging.INFO)

    # Define vehicle constraints
    vehicle = VehicleConstraints(
        max_speed=25.0,  # 25 m/s (90 km/h)
        max_acceleration=3.0,
        wheelbase=2.7,
        width=1.8,
        length=4.5
    )

    # Define start and goal
    start = WayPoint(x=0.0, y=0.0, heading=0.0, velocity=10.0)
    goal = WayPoint(x=100.0, y=20.0, heading=0.0, velocity=10.0)

    # Define obstacles
    obstacles = [
        Obstacle(
            center=np.array([50.0, 5.0, 0.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),
            velocity=np.array([5.0, 0.0, 0.0]),
            heading=0.0,
            timestamp=0.0,
            obstacle_type="car"
        )
    ]

    # Initialize planner
    planner = PathPlanner(PathPlanningMode.HIGHWAY_CRUISING)

    # Plan path
    path = planner.plan_path(start, goal, obstacles, vehicle)

    if path:
        print(f"Path found with {len(path.waypoints)} waypoints")
        print(f"Total distance: {path.total_distance:.1f}m")
        print(f"Total time: {path.total_time:.1f}s")
        print(f"Safety score: {path.safety_score:.2f}")
        print(f"Comfort score: {path.comfort_score:.2f}")
    else:
        print("No path found")

    # Print planning metrics
    metrics = planner.get_planning_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")