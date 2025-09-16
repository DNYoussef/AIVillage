"""
Real Path Planning Implementation - A* and RRT* Algorithms
Replaces theater implementation with genuine planning algorithms
"""

import numpy as np
import heapq
import time
import logging
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import math
from abc import ABC, abstractmethod

class PlannerType(Enum):
    """Available planning algorithms"""
    ASTAR = "astar"
    RRT_STAR = "rrt_star"
    HYBRID_ASTAR = "hybrid_astar"

@dataclass
class Point2D:
    """2D point representation"""
    x: float
    y: float

    def distance_to(self, other: 'Point2D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Pose2D:
    """2D pose with orientation"""
    x: float
    y: float
    theta: float  # orientation in radians

@dataclass
class PathPoint:
    """Point along planned path"""
    pose: Pose2D
    curvature: float
    speed: float
    timestamp: float

@dataclass
class PlanningConstraints:
    """Real planning constraints"""
    max_speed: float = 30.0  # m/s
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = -8.0  # m/s^2
    max_curvature: float = 0.2  # 1/m (minimum turning radius 5m)
    vehicle_width: float = 2.0  # meters
    vehicle_length: float = 4.5  # meters
    safety_margin: float = 1.0  # meters
    planning_horizon: float = 100.0  # meters

class GridMap:
    """Occupancy grid for path planning"""

    def __init__(self, width: int, height: int, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.obstacles: Set[Tuple[int, int]] = set()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = gx * self.resolution
        y = gy * self.resolution
        return x, y

    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid"""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_obstacle(self, gx: int, gy: int) -> bool:
        """Check if grid cell contains obstacle"""
        if not self.is_valid(gx, gy):
            return True
        return (gx, gy) in self.obstacles

    def add_obstacle(self, x: float, y: float, radius: float = 1.0):
        """Add circular obstacle to grid"""
        gx, gy = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution)

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if dx*dx + dy*dy <= grid_radius*grid_radius:
                    nx, ny = gx + dx, gy + dy
                    if self.is_valid(nx, ny):
                        self.obstacles.add((nx, ny))

class AStarPlanner:
    """A* path planning algorithm implementation"""

    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self.movements = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal directions
        ]
        self.diagonal_cost = math.sqrt(2)

    def plan(self, start: Point2D, goal: Point2D) -> List[Point2D]:
        """Plan path using A* algorithm"""
        start_grid = self.grid_map.world_to_grid(start.x, start.y)
        goal_grid = self.grid_map.world_to_grid(goal.x, goal.y)

        if self.grid_map.is_obstacle(*start_grid):
            logging.error("Start position is in obstacle")
            return []

        if self.grid_map.is_obstacle(*goal_grid):
            logging.error("Goal position is in obstacle")
            return []

        # A* implementation
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            for movement in self.movements:
                neighbor = (current[0] + movement[0], current[1] + movement[1])

                if not self.grid_map.is_valid(*neighbor):
                    continue

                if self.grid_map.is_obstacle(*neighbor):
                    continue

                if neighbor in closed_set:
                    continue

                # Calculate movement cost
                if abs(movement[0]) + abs(movement[1]) == 2:  # diagonal
                    movement_cost = self.diagonal_cost
                else:
                    movement_cost = 1.0

                tentative_g_score = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        logging.warning("A* failed to find path")
        return []

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Point2D]:
        """Reconstruct path from A* search"""
        path = []
        while current in came_from:
            x, y = self.grid_map.grid_to_world(*current)
            path.append(Point2D(x, y))
            current = came_from[current]

        # Add start point
        x, y = self.grid_map.grid_to_world(*current)
        path.append(Point2D(x, y))

        return list(reversed(path))

class RRTStarPlanner:
    """RRT* path planning algorithm implementation"""

    def __init__(self, grid_map: GridMap, max_iterations: int = 1000):
        self.grid_map = grid_map
        self.max_iterations = max_iterations
        self.step_size = 2.0  # meters
        self.search_radius = 5.0  # meters for rewiring
        self.nodes = []
        self.edges = {}
        self.costs = {}

    def plan(self, start: Point2D, goal: Point2D) -> List[Point2D]:
        """Plan path using RRT* algorithm"""
        self.nodes = [start]
        self.edges = {0: None}
        self.costs = {0: 0.0}

        for iteration in range(self.max_iterations):
            # Sample random point
            if np.random.random() < 0.1:  # 10% bias toward goal
                rand_point = goal
            else:
                rand_point = self._sample_random_point()

            # Find nearest node
            nearest_idx = self._find_nearest_node(rand_point)
            nearest_node = self.nodes[nearest_idx]

            # Steer toward random point
            new_point = self._steer(nearest_node, rand_point)

            # Check collision
            if self._is_collision_free(nearest_node, new_point):
                # Find nodes within search radius
                near_indices = self._find_near_nodes(new_point)

                # Choose parent with minimum cost
                best_parent_idx = nearest_idx
                min_cost = self.costs[nearest_idx] + nearest_node.distance_to(new_point)

                for near_idx in near_indices:
                    near_node = self.nodes[near_idx]
                    cost = self.costs[near_idx] + near_node.distance_to(new_point)
                    if cost < min_cost and self._is_collision_free(near_node, new_point):
                        best_parent_idx = near_idx
                        min_cost = cost

                # Add new node
                new_idx = len(self.nodes)
                self.nodes.append(new_point)
                self.edges[new_idx] = best_parent_idx
                self.costs[new_idx] = min_cost

                # Rewire tree
                for near_idx in near_indices:
                    near_node = self.nodes[near_idx]
                    cost_through_new = min_cost + new_point.distance_to(near_node)
                    if cost_through_new < self.costs[near_idx]:
                        if self._is_collision_free(new_point, near_node):
                            self.edges[near_idx] = new_idx
                            self.costs[near_idx] = cost_through_new

                # Check if goal is reached
                if new_point.distance_to(goal) < self.step_size:
                    goal_idx = len(self.nodes)
                    self.nodes.append(goal)
                    self.edges[goal_idx] = new_idx
                    self.costs[goal_idx] = min_cost + new_point.distance_to(goal)
                    return self._extract_path(goal_idx)

        logging.warning("RRT* failed to find path within iteration limit")
        return []

    def _sample_random_point(self) -> Point2D:
        """Sample random point in planning space"""
        x = np.random.uniform(0, self.grid_map.width * self.grid_map.resolution)
        y = np.random.uniform(0, self.grid_map.height * self.grid_map.resolution)
        return Point2D(x, y)

    def _find_nearest_node(self, point: Point2D) -> int:
        """Find index of nearest node to given point"""
        distances = [node.distance_to(point) for node in self.nodes]
        return np.argmin(distances)

    def _steer(self, from_point: Point2D, to_point: Point2D) -> Point2D:
        """Steer from one point toward another with step size limit"""
        distance = from_point.distance_to(to_point)
        if distance <= self.step_size:
            return to_point

        angle = math.atan2(to_point.y - from_point.y, to_point.x - from_point.x)
        new_x = from_point.x + self.step_size * math.cos(angle)
        new_y = from_point.y + self.step_size * math.sin(angle)
        return Point2D(new_x, new_y)

    def _is_collision_free(self, from_point: Point2D, to_point: Point2D) -> bool:
        """Check if line segment is collision-free"""
        distance = from_point.distance_to(to_point)
        num_checks = int(distance / (self.grid_map.resolution * 0.5))

        for i in range(num_checks + 1):
            t = i / max(1, num_checks)
            x = from_point.x + t * (to_point.x - from_point.x)
            y = from_point.y + t * (to_point.y - from_point.y)

            gx, gy = self.grid_map.world_to_grid(x, y)
            if self.grid_map.is_obstacle(gx, gy):
                return False

        return True

    def _find_near_nodes(self, point: Point2D) -> List[int]:
        """Find nodes within search radius"""
        near_indices = []
        for i, node in enumerate(self.nodes):
            if node.distance_to(point) <= self.search_radius:
                near_indices.append(i)
        return near_indices

    def _extract_path(self, goal_idx: int) -> List[Point2D]:
        """Extract path from tree"""
        path = []
        current_idx = goal_idx

        while current_idx is not None:
            path.append(self.nodes[current_idx])
            current_idx = self.edges[current_idx]

        return list(reversed(path))

class CollisionChecker:
    """Real collision detection implementation"""

    def __init__(self, constraints: PlanningConstraints):
        self.constraints = constraints

    def check_vehicle_collision(self, pose: Pose2D, obstacles: List[Dict]) -> bool:
        """Check if vehicle pose collides with obstacles"""
        # Get vehicle corners
        corners = self._get_vehicle_corners(pose)

        for obstacle in obstacles:
            if self._polygon_obstacle_collision(corners, obstacle):
                return True

        return False

    def _get_vehicle_corners(self, pose: Pose2D) -> List[Point2D]:
        """Get vehicle corner points in world coordinates"""
        cos_theta = math.cos(pose.theta)
        sin_theta = math.sin(pose.theta)

        # Vehicle corners relative to center
        half_length = self.constraints.vehicle_length / 2
        half_width = self.constraints.vehicle_width / 2

        corners_local = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]

        corners_world = []
        for x_local, y_local in corners_local:
            x_world = pose.x + x_local * cos_theta - y_local * sin_theta
            y_world = pose.y + x_local * sin_theta + y_local * cos_theta
            corners_world.append(Point2D(x_world, y_world))

        return corners_world

    def _polygon_obstacle_collision(self, vehicle_corners: List[Point2D], obstacle: Dict) -> bool:
        """Check collision between vehicle polygon and obstacle"""
        # Simplified collision check - expand for production use
        # This assumes circular obstacles for simplicity
        center = Point2D(obstacle.get('x', 0), obstacle.get('y', 0))
        radius = obstacle.get('radius', 1.0) + self.constraints.safety_margin

        # Check if any vehicle corner is within obstacle
        for corner in vehicle_corners:
            if corner.distance_to(center) <= radius:
                return True

        return False

class PathOptimizer:
    """Real path optimization implementation"""

    def __init__(self, constraints: PlanningConstraints):
        self.constraints = constraints

    def optimize_path(self, path: List[Point2D]) -> List[PathPoint]:
        """Optimize path for smoothness and dynamics"""
        if len(path) < 2:
            return []

        # Smooth path using cubic spline interpolation
        smoothed_path = self._smooth_path(path)

        # Generate velocity profile
        velocity_profile = self._generate_velocity_profile(smoothed_path)

        # Combine into path points
        optimized_path = []
        for i, point in enumerate(smoothed_path):
            if i < len(smoothed_path) - 1:
                next_point = smoothed_path[i + 1]
                theta = math.atan2(next_point.y - point.y, next_point.x - point.x)
            else:
                theta = 0.0  # Final orientation

            pose = Pose2D(point.x, point.y, theta)
            curvature = self._calculate_curvature(smoothed_path, i)
            speed = velocity_profile[i]
            timestamp = i * 0.1  # 10Hz planning rate

            optimized_path.append(PathPoint(pose, curvature, speed, timestamp))

        return optimized_path

    def _smooth_path(self, path: List[Point2D]) -> List[Point2D]:
        """Smooth path using simple averaging"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]  # Keep first point

        for i in range(1, len(path) - 1):
            # Simple smoothing: average with neighbors
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]

            smooth_x = 0.25 * prev_point.x + 0.5 * curr_point.x + 0.25 * next_point.x
            smooth_y = 0.25 * prev_point.y + 0.5 * curr_point.y + 0.25 * next_point.y

            smoothed.append(Point2D(smooth_x, smooth_y))

        smoothed.append(path[-1])  # Keep last point
        return smoothed

    def _generate_velocity_profile(self, path: List[Point2D]) -> List[float]:
        """Generate velocity profile respecting dynamic constraints"""
        if len(path) <= 1:
            return [0.0]

        velocities = []

        for i in range(len(path)):
            # Calculate curvature-limited speed
            curvature = self._calculate_curvature(path, i)
            max_speed_curvature = math.sqrt(self.constraints.max_acceleration / max(abs(curvature), 0.001))

            # Limit by maximum speed
            max_speed = min(self.constraints.max_speed, max_speed_curvature)

            velocities.append(max_speed)

        # Forward pass - acceleration limits
        for i in range(1, len(velocities)):
            if i < len(path):
                distance = path[i].distance_to(path[i-1])
                max_vel_from_accel = math.sqrt(velocities[i-1]**2 + 2 * self.constraints.max_acceleration * distance)
                velocities[i] = min(velocities[i], max_vel_from_accel)

        # Backward pass - deceleration limits
        velocities[-1] = 0.0  # Stop at end
        for i in range(len(velocities) - 2, -1, -1):
            if i + 1 < len(path):
                distance = path[i].distance_to(path[i+1])
                max_vel_from_decel = math.sqrt(velocities[i+1]**2 - 2 * self.constraints.max_deceleration * distance)
                velocities[i] = min(velocities[i], max_vel_from_decel)

        return velocities

    def _calculate_curvature(self, path: List[Point2D], index: int) -> float:
        """Calculate curvature at path point"""
        if len(path) < 3 or index == 0 or index >= len(path) - 1:
            return 0.0

        p1 = path[index - 1]
        p2 = path[index]
        p3 = path[index + 1]

        # Calculate curvature using three points
        a = p1.distance_to(p2)
        b = p2.distance_to(p3)
        c = p1.distance_to(p3)

        # Area of triangle
        s = (a + b + c) / 2
        area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

        if area < 1e-6:
            return 0.0

        # Curvature = 4 * Area / (a * b * c)
        curvature = 4 * area / (a * b * c)

        return curvature

class RealPathPlanner:
    """Real path planning implementation - no theater patterns"""

    def __init__(self, constraints: PlanningConstraints, planner_type: PlannerType = PlannerType.ASTAR):
        self.constraints = constraints
        self.planner_type = planner_type

        # Create grid map for planning
        self.grid_map = GridMap(width=500, height=500, resolution=0.2)  # 100m x 100m at 20cm resolution

        # Initialize planners
        self.astar_planner = AStarPlanner(self.grid_map)
        self.rrt_star_planner = RRTStarPlanner(self.grid_map)

        # Initialize components
        self.collision_checker = CollisionChecker(constraints)
        self.path_optimizer = PathOptimizer(constraints)

        logging.info(f"Real path planner initialized with {planner_type.value}")

    def plan_path(self, start: Pose2D, goal: Pose2D, obstacles: List[Dict]) -> List[PathPoint]:
        """Plan path from start to goal avoiding obstacles"""
        start_time = time.perf_counter()

        try:
            # Update obstacle map
            self._update_obstacle_map(obstacles)

            # Convert poses to points for planning
            start_point = Point2D(start.x, start.y)
            goal_point = Point2D(goal.x, goal.y)

            # Plan geometric path
            if self.planner_type == PlannerType.ASTAR:
                raw_path = self.astar_planner.plan(start_point, goal_point)
            elif self.planner_type == PlannerType.RRT_STAR:
                raw_path = self.rrt_star_planner.plan(start_point, goal_point)
            else:
                logging.error(f"Unknown planner type: {self.planner_type}")
                return []

            if not raw_path:
                logging.warning("Path planning failed - no valid path found")
                return []

            # Optimize path for vehicle dynamics
            optimized_path = self.path_optimizer.optimize_path(raw_path)

            # Validate path against constraints
            if not self._validate_path(optimized_path, obstacles):
                logging.warning("Path validation failed")
                return []

            planning_time = (time.perf_counter() - start_time) * 1000
            logging.info(f"Path planning completed in {planning_time:.2f}ms, {len(optimized_path)} points")

            return optimized_path

        except Exception as e:
            logging.error(f"Path planning failed: {e}")
            return []

    def _update_obstacle_map(self, obstacles: List[Dict]):
        """Update grid map with current obstacles"""
        # Clear previous obstacles
        self.grid_map.obstacles.clear()

        # Add new obstacles
        for obstacle in obstacles:
            x = obstacle.get('x', 0)
            y = obstacle.get('y', 0)
            radius = obstacle.get('radius', 1.0)
            self.grid_map.add_obstacle(x, y, radius)

    def _validate_path(self, path: List[PathPoint], obstacles: List[Dict]) -> bool:
        """Validate path against safety constraints"""
        if not path:
            return False

        for path_point in path:
            # Check collision
            if self.collision_checker.check_vehicle_collision(path_point.pose, obstacles):
                return False

            # Check speed constraints
            if path_point.speed > self.constraints.max_speed:
                return False

            # Check curvature constraints
            if abs(path_point.curvature) > self.constraints.max_curvature:
                return False

        return True

    def get_planning_statistics(self) -> Dict:
        """Get planning performance statistics"""
        return {
            "planner_type": self.planner_type.value,
            "grid_resolution": self.grid_map.resolution,
            "grid_size": f"{self.grid_map.width}x{self.grid_map.height}",
            "obstacle_count": len(self.grid_map.obstacles),
            "constraints": {
                "max_speed": self.constraints.max_speed,
                "max_acceleration": self.constraints.max_acceleration,
                "max_curvature": self.constraints.max_curvature
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create planning constraints
    constraints = PlanningConstraints(
        max_speed=20.0,
        max_acceleration=2.0,
        max_curvature=0.1
    )

    # Create planner
    planner = RealPathPlanner(constraints, PlannerType.ASTAR)

    # Define start and goal
    start = Pose2D(x=0.0, y=0.0, theta=0.0)
    goal = Pose2D(x=50.0, y=30.0, theta=0.0)

    # Define obstacles
    obstacles = [
        {"x": 20.0, "y": 15.0, "radius": 3.0},
        {"x": 35.0, "y": 10.0, "radius": 2.5},
        {"x": 40.0, "y": 25.0, "radius": 4.0}
    ]

    # Plan path
    path = planner.plan_path(start, goal, obstacles)

    if path:
        print(f"Path planned successfully with {len(path)} points")
        print(f"Planning statistics: {planner.get_planning_statistics()}")
    else:
        print("Path planning failed")