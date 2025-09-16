"""
ADAS Prediction Agent - Phase 7
Trajectory and behavior prediction for autonomous driving
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
from collections import deque

class BehaviorType(Enum):
    LANE_KEEPING = "lane_keeping"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    BRAKING = "braking"
    ACCELERATING = "accelerating"
    STOPPING = "stopping"
    REVERSING = "reversing"
    UNKNOWN = "unknown"

class IntentLevel(Enum):
    CERTAIN = 0.9
    LIKELY = 0.7
    POSSIBLE = 0.5
    UNLIKELY = 0.3
    UNCERTAIN = 0.1

@dataclass
class TrajectoryPoint:
    """Single point in predicted trajectory"""
    x: float
    y: float
    z: float
    timestamp: float
    velocity: float
    acceleration: float
    heading: float
    confidence: float

@dataclass
class PredictedTrajectory:
    """Predicted trajectory for an object"""
    object_id: str
    prediction_horizon: float  # seconds
    trajectory_points: List[TrajectoryPoint]
    behavior_type: BehaviorType
    intent_confidence: float
    collision_probability: float
    safety_margin: float
    prediction_accuracy: float
    timestamp: float

@dataclass
class BehaviorPrediction:
    """Behavior prediction for multiple objects"""
    predictions: List[PredictedTrajectory]
    scene_risk_level: float
    interaction_matrix: np.ndarray
    critical_scenarios: List[Dict[str, Any]]
    prediction_quality: float
    processing_latency: float
    timestamp: float

class PredictionAgent:
    """
    Advanced prediction agent for ADAS systems
    Predicts object trajectories and behaviors for safe path planning
    """

    def __init__(self, agent_id: str = "prediction_001"):
        self.agent_id = agent_id
        self.logger = self._setup_logging()

        # Real-time constraints
        self.max_processing_time = 0.012  # 12ms target
        self.prediction_frequency = 50  # 50Hz
        self.prediction_horizon = 5.0  # 5 seconds ahead

        # Prediction models
        self.trajectory_models = {}
        self.behavior_models = {}
        self.interaction_models = {}

        # Object history for tracking
        self.object_history: Dict[str, deque] = {}
        self.max_history_length = 20

        # Processing pipeline
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=50)
        self.processing_thread = None
        self.is_running = False

        # Performance metrics
        self.performance_metrics = {
            'prediction_latency': [],
            'accuracy_scores': [],
            'false_alarms': 0,
            'missed_detections': 0,
            'prediction_rate': 0
        }

        # Safety parameters
        self.safety_thresholds = {
            'max_latency': 0.015,  # 15ms max
            'min_prediction_confidence': 0.6,
            'critical_collision_prob': 0.8,
            'safety_margin_distance': 2.0  # 2m safety margin
        }

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"ADAS.Prediction.{self.agent_id}")
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
        """Initialize prediction system"""
        try:
            self.logger.info("Initializing ADAS Prediction Agent")

            # Initialize prediction models
            await self._initialize_models()

            # Setup behavior analysis
            await self._setup_behavior_analysis()

            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._prediction_processing_loop,
                daemon=True
            )
            self.processing_thread.start()

            self.logger.info("Prediction Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def _initialize_models(self):
        """Initialize prediction models"""
        # Constant velocity model
        self.trajectory_models['constant_velocity'] = {
            'type': 'kinematic',
            'parameters': {'time_horizon': self.prediction_horizon}
        }

        # Constant acceleration model
        self.trajectory_models['constant_acceleration'] = {
            'type': 'kinematic',
            'parameters': {'time_horizon': self.prediction_horizon}
        }

        # Bicycle model for vehicles
        self.trajectory_models['bicycle_model'] = {
            'type': 'dynamic',
            'parameters': {
                'wheelbase': 2.7,  # meters
                'max_steering_angle': 0.6,  # radians
                'time_horizon': self.prediction_horizon
            }
        }

        # Lane-following model
        self.trajectory_models['lane_following'] = {
            'type': 'behavioral',
            'parameters': {
                'lane_center_weight': 0.8,
                'speed_preference': 0.6,
                'time_horizon': self.prediction_horizon
            }
        }

        # Behavior classification models
        self.behavior_models = {
            'vehicle_behavior': {
                'features': ['velocity', 'acceleration', 'steering_angle', 'position_in_lane'],
                'classes': [behavior.value for behavior in BehaviorType]
            },
            'pedestrian_behavior': {
                'features': ['velocity', 'direction_change', 'proximity_to_crosswalk'],
                'classes': ['crossing', 'walking_parallel', 'standing', 'approaching_road']
            }
        }

        self.logger.info("Prediction models initialized")

    async def _setup_behavior_analysis(self):
        """Setup behavior analysis components"""
        # Behavior pattern recognition
        self.behavior_patterns = {
            'lane_change_indicators': {
                'lateral_acceleration_threshold': 0.5,  # m/s²
                'velocity_change_threshold': 2.0,  # m/s
                'time_window': 2.0  # seconds
            },
            'braking_indicators': {
                'deceleration_threshold': -2.0,  # m/s²
                'time_to_brake': 1.5  # seconds
            },
            'turning_indicators': {
                'angular_velocity_threshold': 0.3,  # rad/s
                'lateral_displacement_threshold': 1.0  # meters
            }
        }

        # Interaction analysis
        self.interaction_thresholds = {
            'close_proximity': 10.0,  # meters
            'collision_time_threshold': 5.0,  # seconds
            'influence_radius': 20.0  # meters
        }

        self.logger.info("Behavior analysis setup complete")

    def _prediction_processing_loop(self):
        """Main prediction processing loop"""
        self.logger.info("Starting prediction processing loop")

        while self.is_running:
            try:
                # Get perception input
                try:
                    scene_understanding = self.input_queue.get(timeout=0.001)
                    processing_start = time.perf_counter()

                    # Process predictions
                    behavior_prediction = self._process_predictions(scene_understanding)

                    # Check processing time
                    processing_time = time.perf_counter() - processing_start
                    if processing_time > self.max_processing_time:
                        self.logger.warning(
                            f"Prediction processing exceeded time limit: {processing_time*1000:.2f}ms"
                        )

                    # Update performance metrics
                    self.performance_metrics['prediction_latency'].append(processing_time)
                    if len(self.performance_metrics['prediction_latency']) > 1000:
                        self.performance_metrics['prediction_latency'].pop(0)

                    # Output predictions
                    if behavior_prediction:
                        self.output_queue.put(behavior_prediction)

                    self.input_queue.task_done()

                except queue.Empty:
                    continue

            except Exception as e:
                self.logger.error(f"Prediction processing error: {e}")
                continue

    def _process_predictions(self, scene_understanding) -> Optional[BehaviorPrediction]:
        """Process trajectory and behavior predictions"""
        try:
            processing_start = time.perf_counter()

            # Update object histories
            self._update_object_histories(scene_understanding.detected_objects)

            # Generate trajectory predictions
            trajectory_predictions = self._predict_trajectories(scene_understanding.detected_objects)

            # Analyze object interactions
            interaction_matrix = self._analyze_interactions(trajectory_predictions)

            # Identify critical scenarios
            critical_scenarios = self._identify_critical_scenarios(trajectory_predictions, interaction_matrix)

            # Calculate scene risk level
            scene_risk = self._calculate_scene_risk(trajectory_predictions, critical_scenarios)

            # Assess prediction quality
            prediction_quality = self._assess_prediction_quality(trajectory_predictions)

            processing_time = time.perf_counter() - processing_start

            return BehaviorPrediction(
                predictions=trajectory_predictions,
                scene_risk_level=scene_risk,
                interaction_matrix=interaction_matrix,
                critical_scenarios=critical_scenarios,
                prediction_quality=prediction_quality,
                processing_latency=processing_time,
                timestamp=time.time()
            )

        except Exception as e:
            self.logger.error(f"Prediction processing failed: {e}")
            return None

    def _update_object_histories(self, detected_objects: List):
        """Update object history for tracking"""
        current_time = time.time()

        # Update histories for existing objects
        for obj in detected_objects:
            if obj.object_id not in self.object_history:
                self.object_history[obj.object_id] = deque(maxlen=self.max_history_length)

            # Add current state to history
            state = {
                'timestamp': obj.timestamp,
                'position': obj.position,
                'velocity': obj.velocity,
                'orientation': obj.orientation,
                'class_type': obj.class_type.value,
                'confidence': obj.confidence
            }

            self.object_history[obj.object_id].append(state)

        # Clean up old histories
        active_object_ids = {obj.object_id for obj in detected_objects}
        old_object_ids = []

        for obj_id, history in self.object_history.items():
            if obj_id not in active_object_ids:
                # Check if history is too old
                if history and current_time - history[-1]['timestamp'] > 2.0:
                    old_object_ids.append(obj_id)

        for obj_id in old_object_ids:
            del self.object_history[obj_id]

    def _predict_trajectories(self, detected_objects: List) -> List[PredictedTrajectory]:
        """Predict trajectories for all objects"""
        predictions = []

        for obj in detected_objects:
            try:
                # Get object history
                history = self.object_history.get(obj.object_id, deque())

                # Select appropriate prediction model
                model_type = self._select_prediction_model(obj, history)

                # Generate trajectory prediction
                trajectory = self._generate_trajectory_prediction(obj, history, model_type)

                # Predict behavior
                behavior_type, intent_confidence = self._predict_behavior(obj, history)

                # Calculate collision probability
                collision_prob = self._calculate_collision_probability(trajectory)

                # Calculate safety margin
                safety_margin = self._calculate_safety_margin(trajectory, detected_objects)

                # Assess prediction accuracy
                prediction_accuracy = self._assess_trajectory_accuracy(obj, history)

                prediction = PredictedTrajectory(
                    object_id=obj.object_id,
                    prediction_horizon=self.prediction_horizon,
                    trajectory_points=trajectory,
                    behavior_type=behavior_type,
                    intent_confidence=intent_confidence,
                    collision_probability=collision_prob,
                    safety_margin=safety_margin,
                    prediction_accuracy=prediction_accuracy,
                    timestamp=time.time()
                )

                predictions.append(prediction)

            except Exception as e:
                self.logger.error(f"Trajectory prediction failed for {obj.object_id}: {e}")
                continue

        return predictions

    def _select_prediction_model(self, obj, history: deque) -> str:
        """Select appropriate prediction model based on object type and behavior"""
        # Default to constant velocity
        model_type = 'constant_velocity'

        try:
            if len(history) < 3:
                return model_type

            # Analyze recent motion
            recent_states = list(history)[-3:]
            velocities = [state['velocity'] for state in recent_states]

            # Check for acceleration patterns
            if len(velocities) >= 2:
                speed_changes = [
                    np.linalg.norm(velocities[i]) - np.linalg.norm(velocities[i-1])
                    for i in range(1, len(velocities))
                ]

                if any(abs(change) > 1.0 for change in speed_changes):
                    model_type = 'constant_acceleration'

            # Vehicle-specific models
            if obj.class_type.value == 'vehicle':
                # Check for lane-following behavior
                if self._is_lane_following(obj, history):
                    model_type = 'lane_following'
                else:
                    model_type = 'bicycle_model'

        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")

        return model_type

    def _is_lane_following(self, obj, history: deque) -> bool:
        """Check if object is following lane"""
        if len(history) < 5:
            return False

        # Analyze lateral movement
        recent_positions = [state['position'] for state in list(history)[-5:]]
        lateral_positions = [pos[0] for pos in recent_positions]  # x-coordinates

        # Calculate lateral variance
        lateral_variance = np.var(lateral_positions)

        # Low lateral variance indicates lane following
        return lateral_variance < 1.0  # 1 meter variance threshold

    def _generate_trajectory_prediction(self, obj, history: deque, model_type: str) -> List[TrajectoryPoint]:
        """Generate trajectory prediction using specified model"""
        trajectory_points = []

        try:
            # Time steps for prediction
            dt = 0.1  # 100ms steps
            num_steps = int(self.prediction_horizon / dt)

            # Current state
            current_pos = np.array(obj.position)
            current_vel = np.array(obj.velocity)
            current_heading = obj.orientation

            if model_type == 'constant_velocity':
                trajectory_points = self._predict_constant_velocity(
                    current_pos, current_vel, current_heading, dt, num_steps
                )
            elif model_type == 'constant_acceleration':
                acceleration = self._estimate_acceleration(history)
                trajectory_points = self._predict_constant_acceleration(
                    current_pos, current_vel, acceleration, current_heading, dt, num_steps
                )
            elif model_type == 'bicycle_model':
                trajectory_points = self._predict_bicycle_model(
                    current_pos, current_vel, current_heading, dt, num_steps
                )
            elif model_type == 'lane_following':
                trajectory_points = self._predict_lane_following(
                    current_pos, current_vel, current_heading, dt, num_steps
                )

        except Exception as e:
            self.logger.error(f"Trajectory generation failed: {e}")
            # Fallback to simple constant velocity
            trajectory_points = self._predict_constant_velocity(
                np.array(obj.position), np.array(obj.velocity), obj.orientation, 0.1, 50
            )

        return trajectory_points

    def _predict_constant_velocity(self, pos: np.ndarray, vel: np.ndarray, heading: float,
                                  dt: float, num_steps: int) -> List[TrajectoryPoint]:
        """Predict trajectory using constant velocity model"""
        trajectory = []
        current_pos = pos.copy()
        speed = np.linalg.norm(vel)

        for i in range(num_steps):
            timestamp = time.time() + i * dt

            # Update position
            current_pos += vel * dt

            # Create trajectory point
            point = TrajectoryPoint(
                x=current_pos[0],
                y=current_pos[1],
                z=current_pos[2] if len(current_pos) > 2 else 0.0,
                timestamp=timestamp,
                velocity=speed,
                acceleration=0.0,
                heading=heading,
                confidence=max(0.1, 1.0 - i * 0.02)  # Decreasing confidence over time
            )

            trajectory.append(point)

        return trajectory

    def _predict_constant_acceleration(self, pos: np.ndarray, vel: np.ndarray, accel: np.ndarray,
                                     heading: float, dt: float, num_steps: int) -> List[TrajectoryPoint]:
        """Predict trajectory using constant acceleration model"""
        trajectory = []
        current_pos = pos.copy()
        current_vel = vel.copy()

        for i in range(num_steps):
            timestamp = time.time() + i * dt

            # Update position and velocity
            current_pos += current_vel * dt + 0.5 * accel * dt**2
            current_vel += accel * dt

            speed = np.linalg.norm(current_vel)
            acceleration_magnitude = np.linalg.norm(accel)

            point = TrajectoryPoint(
                x=current_pos[0],
                y=current_pos[1],
                z=current_pos[2] if len(current_pos) > 2 else 0.0,
                timestamp=timestamp,
                velocity=speed,
                acceleration=acceleration_magnitude,
                heading=heading,
                confidence=max(0.1, 1.0 - i * 0.02)
            )

            trajectory.append(point)

        return trajectory

    def _predict_bicycle_model(self, pos: np.ndarray, vel: np.ndarray, heading: float,
                              dt: float, num_steps: int) -> List[TrajectoryPoint]:
        """Predict trajectory using bicycle model for vehicles"""
        trajectory = []
        current_pos = pos.copy()
        current_heading = heading
        speed = np.linalg.norm(vel)

        # Assume constant steering angle (simplified)
        steering_angle = 0.0
        wheelbase = 2.7  # meters

        for i in range(num_steps):
            timestamp = time.time() + i * dt

            # Bicycle model kinematics
            angular_velocity = speed * np.tan(steering_angle) / wheelbase
            current_heading += angular_velocity * dt

            # Update position
            dx = speed * np.cos(current_heading) * dt
            dy = speed * np.sin(current_heading) * dt

            current_pos[0] += dx
            current_pos[1] += dy

            point = TrajectoryPoint(
                x=current_pos[0],
                y=current_pos[1],
                z=current_pos[2] if len(current_pos) > 2 else 0.0,
                timestamp=timestamp,
                velocity=speed,
                acceleration=0.0,
                heading=current_heading,
                confidence=max(0.1, 1.0 - i * 0.02)
            )

            trajectory.append(point)

        return trajectory

    def _predict_lane_following(self, pos: np.ndarray, vel: np.ndarray, heading: float,
                               dt: float, num_steps: int) -> List[TrajectoryPoint]:
        """Predict trajectory assuming lane-following behavior"""
        trajectory = []
        current_pos = pos.copy()
        speed = np.linalg.norm(vel)

        # Assume following lane center with slight variations
        lane_center_y = current_pos[1]  # Assume current y is near lane center

        for i in range(num_steps):
            timestamp = time.time() + i * dt

            # Move forward along lane
            current_pos[1] += speed * dt

            # Small lateral adjustments to stay in lane
            lateral_drift = 0.1 * np.sin(0.1 * i)  # Small sinusoidal drift
            current_pos[0] = pos[0] + lateral_drift

            point = TrajectoryPoint(
                x=current_pos[0],
                y=current_pos[1],
                z=current_pos[2] if len(current_pos) > 2 else 0.0,
                timestamp=timestamp,
                velocity=speed,
                acceleration=0.0,
                heading=heading,
                confidence=max(0.1, 1.0 - i * 0.015)  # Higher confidence for lane following
            )

            trajectory.append(point)

        return trajectory

    def _estimate_acceleration(self, history: deque) -> np.ndarray:
        """Estimate acceleration from object history"""
        if len(history) < 2:
            return np.zeros(3)

        # Get last two velocity measurements
        recent_states = list(history)[-2:]
        vel1 = np.array(recent_states[0]['velocity'])
        vel2 = np.array(recent_states[1]['velocity'])
        dt = recent_states[1]['timestamp'] - recent_states[0]['timestamp']

        if dt > 0:
            acceleration = (vel2 - vel1) / dt
            return acceleration
        else:
            return np.zeros(3)

    def _predict_behavior(self, obj, history: deque) -> Tuple[BehaviorType, float]:
        """Predict object behavior and intent"""
        behavior_type = BehaviorType.LANE_KEEPING
        confidence = 0.5

        try:
            if len(history) < 3:
                return behavior_type, confidence

            # Analyze motion patterns
            recent_states = list(history)[-5:]

            # Calculate motion characteristics
            velocities = [np.array(state['velocity']) for state in recent_states]
            positions = [np.array(state['position']) for state in recent_states]

            # Lateral movement analysis
            lateral_positions = [pos[0] for pos in positions]
            lateral_change = lateral_positions[-1] - lateral_positions[0]

            # Speed analysis
            speeds = [np.linalg.norm(vel) for vel in velocities]
            speed_change = speeds[-1] - speeds[0]

            # Determine behavior
            if abs(lateral_change) > 2.0:  # Significant lateral movement
                if lateral_change > 0:
                    behavior_type = BehaviorType.LANE_CHANGE_RIGHT
                else:
                    behavior_type = BehaviorType.LANE_CHANGE_LEFT
                confidence = 0.8
            elif speed_change < -2.0:  # Significant deceleration
                behavior_type = BehaviorType.BRAKING
                confidence = 0.7
            elif speed_change > 2.0:  # Significant acceleration
                behavior_type = BehaviorType.ACCELERATING
                confidence = 0.7
            elif abs(speed_change) < 0.5 and abs(lateral_change) < 0.5:  # Stable motion
                behavior_type = BehaviorType.LANE_KEEPING
                confidence = 0.9

        except Exception as e:
            self.logger.error(f"Behavior prediction failed: {e}")

        return behavior_type, confidence

    def _calculate_collision_probability(self, trajectory: List[TrajectoryPoint]) -> float:
        """Calculate collision probability for trajectory"""
        collision_prob = 0.0

        try:
            # Simple collision probability based on trajectory uncertainty
            if trajectory:
                # Higher uncertainty = higher collision probability
                avg_confidence = np.mean([point.confidence for point in trajectory])
                collision_prob = 1.0 - avg_confidence

                # Consider velocity for collision risk
                if trajectory[0].velocity > 20.0:  # High speed increases risk
                    collision_prob *= 1.5

                collision_prob = min(1.0, collision_prob)

        except Exception as e:
            self.logger.error(f"Collision probability calculation failed: {e}")

        return collision_prob

    def _calculate_safety_margin(self, trajectory: List[TrajectoryPoint], all_objects: List) -> float:
        """Calculate safety margin for trajectory"""
        min_safety_margin = float('inf')

        try:
            for point in trajectory:
                for obj in all_objects:
                    # Calculate distance to other objects
                    obj_pos = np.array(obj.position)
                    traj_pos = np.array([point.x, point.y, point.z])
                    distance = np.linalg.norm(traj_pos - obj_pos)

                    min_safety_margin = min(min_safety_margin, distance)

            if min_safety_margin == float('inf'):
                min_safety_margin = 100.0  # Large margin if no other objects

        except Exception as e:
            self.logger.error(f"Safety margin calculation failed: {e}")
            min_safety_margin = 10.0  # Default margin

        return min_safety_margin

    def _assess_trajectory_accuracy(self, obj, history: deque) -> float:
        """Assess prediction accuracy based on historical performance"""
        accuracy = 0.8  # Default accuracy

        try:
            # Simple accuracy based on object tracking stability
            if len(history) >= 5:
                positions = [np.array(state['position']) for state in list(history)[-5:]]

                # Calculate position variance
                position_variance = np.var(positions, axis=0)
                total_variance = np.sum(position_variance)

                # Lower variance = higher accuracy
                accuracy = max(0.1, 1.0 - total_variance / 10.0)

        except Exception as e:
            self.logger.error(f"Accuracy assessment failed: {e}")

        return accuracy

    def _analyze_interactions(self, predictions: List[PredictedTrajectory]) -> np.ndarray:
        """Analyze interactions between predicted trajectories"""
        num_objects = len(predictions)
        interaction_matrix = np.zeros((num_objects, num_objects))

        try:
            for i, pred1 in enumerate(predictions):
                for j, pred2 in enumerate(predictions):
                    if i != j:
                        interaction_strength = self._calculate_interaction_strength(pred1, pred2)
                        interaction_matrix[i, j] = interaction_strength

        except Exception as e:
            self.logger.error(f"Interaction analysis failed: {e}")

        return interaction_matrix

    def _calculate_interaction_strength(self, pred1: PredictedTrajectory, pred2: PredictedTrajectory) -> float:
        """Calculate interaction strength between two predictions"""
        interaction = 0.0

        try:
            # Check trajectory overlap
            min_distance = float('inf')
            min_time_diff = float('inf')

            for point1 in pred1.trajectory_points:
                for point2 in pred2.trajectory_points:
                    # Spatial distance
                    spatial_dist = np.sqrt(
                        (point1.x - point2.x)**2 +
                        (point1.y - point2.y)**2
                    )

                    # Temporal distance
                    temporal_dist = abs(point1.timestamp - point2.timestamp)

                    if spatial_dist < min_distance:
                        min_distance = spatial_dist
                        min_time_diff = temporal_dist

            # Interaction strength based on proximity and timing
            if min_distance < self.interaction_thresholds['close_proximity']:
                spatial_factor = 1.0 - (min_distance / self.interaction_thresholds['close_proximity'])
                temporal_factor = 1.0 - (min_time_diff / 5.0)  # 5 second window

                interaction = spatial_factor * temporal_factor

        except Exception as e:
            self.logger.error(f"Interaction strength calculation failed: {e}")

        return max(0.0, min(1.0, interaction))

    def _identify_critical_scenarios(self, predictions: List[PredictedTrajectory],
                                   interaction_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Identify critical scenarios requiring attention"""
        critical_scenarios = []

        try:
            # High collision probability scenarios
            for pred in predictions:
                if pred.collision_probability > self.safety_thresholds['critical_collision_prob']:
                    scenario = {
                        'type': 'high_collision_risk',
                        'object_id': pred.object_id,
                        'collision_probability': pred.collision_probability,
                        'time_to_impact': self._estimate_time_to_impact(pred),
                        'severity': 'critical'
                    }
                    critical_scenarios.append(scenario)

            # Close interaction scenarios
            high_interaction_pairs = np.where(interaction_matrix > 0.7)
            for i, j in zip(high_interaction_pairs[0], high_interaction_pairs[1]):
                if i < len(predictions) and j < len(predictions):
                    scenario = {
                        'type': 'close_interaction',
                        'object_ids': [predictions[i].object_id, predictions[j].object_id],
                        'interaction_strength': interaction_matrix[i, j],
                        'severity': 'high'
                    }
                    critical_scenarios.append(scenario)

            # Low safety margin scenarios
            for pred in predictions:
                if pred.safety_margin < self.safety_thresholds['safety_margin_distance']:
                    scenario = {
                        'type': 'low_safety_margin',
                        'object_id': pred.object_id,
                        'safety_margin': pred.safety_margin,
                        'severity': 'medium'
                    }
                    critical_scenarios.append(scenario)

        except Exception as e:
            self.logger.error(f"Critical scenario identification failed: {e}")

        return critical_scenarios

    def _estimate_time_to_impact(self, prediction: PredictedTrajectory) -> float:
        """Estimate time to potential impact"""
        if not prediction.trajectory_points:
            return float('inf')

        # Find closest approach to origin (ego vehicle)
        min_distance = float('inf')
        min_time = float('inf')

        for point in prediction.trajectory_points:
            distance = np.sqrt(point.x**2 + point.y**2)
            if distance < min_distance:
                min_distance = distance
                min_time = point.timestamp - time.time()

        return max(0.0, min_time)

    def _calculate_scene_risk(self, predictions: List[PredictedTrajectory],
                            critical_scenarios: List[Dict]) -> float:
        """Calculate overall scene risk level"""
        scene_risk = 0.0

        try:
            # Risk from individual predictions
            if predictions:
                avg_collision_prob = np.mean([pred.collision_probability for pred in predictions])
                min_safety_margin = min([pred.safety_margin for pred in predictions])

                individual_risk = avg_collision_prob * 0.7
                proximity_risk = max(0.0, 1.0 - min_safety_margin / 10.0) * 0.3

                scene_risk = individual_risk + proximity_risk

            # Risk from critical scenarios
            critical_risk = len(critical_scenarios) * 0.1
            scene_risk = min(1.0, scene_risk + critical_risk)

        except Exception as e:
            self.logger.error(f"Scene risk calculation failed: {e}")
            scene_risk = 0.5  # Default medium risk

        return scene_risk

    def _assess_prediction_quality(self, predictions: List[PredictedTrajectory]) -> float:
        """Assess overall prediction quality"""
        quality = 0.8  # Default quality

        try:
            if predictions:
                # Average prediction accuracy
                avg_accuracy = np.mean([pred.prediction_accuracy for pred in predictions])

                # Average intent confidence
                avg_intent_conf = np.mean([pred.intent_confidence for pred in predictions])

                # Combine metrics
                quality = (avg_accuracy * 0.6 + avg_intent_conf * 0.4)

        except Exception as e:
            self.logger.error(f"Prediction quality assessment failed: {e}")

        return quality

    async def process_scene_understanding(self, scene_understanding) -> bool:
        """Process scene understanding from perception agent"""
        try:
            if not scene_understanding:
                return False

            # Add to processing queue
            try:
                self.input_queue.put_nowait(scene_understanding)
                return True
            except queue.Full:
                self.logger.warning("Prediction input queue full, dropping data")
                return False

        except Exception as e:
            self.logger.error(f"Error processing scene understanding: {e}")
            return False

    async def get_behavior_prediction(self, timeout: float = 0.001) -> Optional[BehaviorPrediction]:
        """Get latest behavior prediction"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()

        if metrics['prediction_latency']:
            metrics['avg_prediction_latency'] = np.mean(metrics['prediction_latency'])
            metrics['max_prediction_latency'] = np.max(metrics['prediction_latency'])

        metrics['tracked_objects'] = len(self.object_history)
        metrics['prediction_horizon'] = self.prediction_horizon

        return metrics

    async def shutdown(self):
        """Shutdown prediction agent"""
        self.logger.info("Shutting down Prediction Agent")

        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        self.executor.shutdown(wait=True)

        self.logger.info("Prediction Agent shutdown complete")

# Example usage
if __name__ == "__main__":
    async def test_prediction():
        agent = PredictionAgent()

        if await agent.initialize():
            print("Prediction Agent initialized successfully")

            # Simulate scene understanding input
            from dataclasses import dataclass
            from enum import Enum

            @dataclass
            class MockObject:
                object_id: str
                class_type: Enum
                position: tuple
                velocity: tuple
                orientation: float
                timestamp: float

            class MockClass(Enum):
                VEHICLE = "vehicle"

            mock_scene = type('SceneUnderstanding', (), {
                'detected_objects': [
                    MockObject("car1", MockClass.VEHICLE, (10, 20, 0), (-5, 0, 0), 0.0, time.time())
                ]
            })()

            await agent.process_scene_understanding(mock_scene)
            await asyncio.sleep(0.1)

            prediction = await agent.get_behavior_prediction()
            if prediction:
                print(f"Generated {len(prediction.predictions)} predictions")
                print(f"Scene risk: {prediction.scene_risk_level:.2f}")

            metrics = await agent.get_performance_metrics()
            print(f"Metrics: {metrics}")

            await agent.shutdown()

    asyncio.run(test_prediction())