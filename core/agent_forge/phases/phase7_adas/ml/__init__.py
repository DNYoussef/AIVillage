"""
ADAS Phase 7 - Machine Learning Module
Comprehensive ML components for automotive-grade ADAS systems
"""

from .trajectory_prediction import (
    TrajectoryPredictor,
    TrajectoryState,
    PredictionResult,
    LSTMTrajectoryPredictor,
    KalmanTrajectoryFilter,
    MultiAgentInteractionModel,
    validate_trajectory_safety
)

from .path_planning import (
    PathPlanner,
    WayPoint,
    Path,
    VehicleConstraints,
    Obstacle,
    PathPlanningMode,
    AStarPlanner,
    DynamicProgrammingPlanner
)

from .scene_understanding import (
    SceneUnderstandingSystem,
    Object3D,
    SceneSegmentation,
    DepthEstimation,
    SceneContext,
    WeatherCondition,
    LightingCondition,
    EfficientNet3D,
    StereoDepthNet,
    WeatherAdaptationNet
)

from .edge_optimization import (
    EdgeOptimizer,
    HardwareSpecs,
    OptimizationConfig,
    PerformanceMetrics,
    ECUType,
    OptimizationLevel,
    QuantizationStrategy,
    ModelPruning,
    TensorRTOptimizer,
    MemoryOptimizer,
    PowerProfiler,
    create_automotive_calibration_dataset,
    analyze_inference_profile
)

__version__ = "1.0.0"

__all__ = [
    # Trajectory Prediction
    "TrajectoryPredictor",
    "TrajectoryState",
    "PredictionResult",
    "LSTMTrajectoryPredictor",
    "KalmanTrajectoryFilter",
    "MultiAgentInteractionModel",
    "validate_trajectory_safety",

    # Path Planning
    "PathPlanner",
    "WayPoint",
    "Path",
    "VehicleConstraints",
    "Obstacle",
    "PathPlanningMode",
    "AStarPlanner",
    "DynamicProgrammingPlanner",

    # Scene Understanding
    "SceneUnderstandingSystem",
    "Object3D",
    "SceneSegmentation",
    "DepthEstimation",
    "SceneContext",
    "WeatherCondition",
    "LightingCondition",
    "EfficientNet3D",
    "StereoDepthNet",
    "WeatherAdaptationNet",

    # Edge Optimization
    "EdgeOptimizer",
    "HardwareSpecs",
    "OptimizationConfig",
    "PerformanceMetrics",
    "ECUType",
    "OptimizationLevel",
    "QuantizationStrategy",
    "ModelPruning",
    "TensorRTOptimizer",
    "MemoryOptimizer",
    "PowerProfiler",
    "create_automotive_calibration_dataset",
    "analyze_inference_profile"
]


class ADASMLPipeline:
    """Integrated ML pipeline for ADAS Phase 7"""

    def __init__(self,
                 hardware_specs: HardwareSpecs,
                 optimization_config: OptimizationConfig,
                 use_stereo: bool = True):
        """
        Initialize complete ADAS ML pipeline

        Args:
            hardware_specs: Target ECU hardware specifications
            optimization_config: Edge optimization configuration
            use_stereo: Whether to use stereo depth estimation
        """

        # Initialize scene understanding
        self.scene_system = SceneUnderstandingSystem(use_stereo=use_stereo)

        # Initialize trajectory prediction
        self.trajectory_predictor = TrajectoryPredictor()

        # Initialize path planning
        self.path_planner = PathPlanner()

        # Initialize edge optimization
        self.edge_optimizer = EdgeOptimizer(hardware_specs, optimization_config)

        # Vehicle constraints (typical automotive parameters)
        self.vehicle_constraints = VehicleConstraints(
            max_speed=25.0,  # m/s (90 km/h)
            max_acceleration=4.0,  # m/s^2
            max_deceleration=8.0,  # m/s^2
            wheelbase=2.8,  # meters
            width=1.8,  # meters
            length=4.5  # meters
        )

    def process_frame(self,
                     left_image,
                     right_image=None,
                     ego_state=None,
                     goal_waypoint=None):
        """
        Process single frame through complete ADAS pipeline

        Args:
            left_image: Main camera image
            right_image: Optional stereo right image
            ego_state: Current vehicle state
            goal_waypoint: Target destination

        Returns:
            Complete ADAS processing result
        """

        # Scene understanding
        scene_context = self.scene_system.process_frame(left_image, right_image)

        # Extract detected objects for trajectory prediction
        object_states = []
        for obj in scene_context.objects_3d:
            if obj.class_name in ['car', 'pedestrian', 'cyclist']:
                state = TrajectoryState(
                    position=obj.center_3d,
                    velocity=obj.velocity,
                    acceleration=np.zeros(3),
                    heading=0.0,  # TODO: Extract from bbox
                    timestamp=scene_context.timestamp,
                    confidence=obj.confidence
                )
                object_states.append(state)

        # Trajectory prediction for detected objects
        predictions = {}
        if ego_state and len(object_states) > 0:
            ego_history = [ego_state] * 15  # Placeholder history

            for i, obj_state in enumerate(object_states):
                obj_history = [obj_state] * 15  # Placeholder history
                prediction = self.trajectory_predictor.predict_trajectory(
                    object_id=i,
                    history=obj_history,
                    other_objects=object_states
                )
                if prediction:
                    predictions[i] = prediction

        # Path planning
        planned_path = None
        if ego_state and goal_waypoint:
            # Convert scene objects to obstacles
            obstacles = []
            for obj in scene_context.objects_3d:
                obstacle = Obstacle(
                    center=obj.center_3d,
                    dimensions=np.array([2.0, 2.0, 1.5]),  # Default dimensions
                    velocity=obj.velocity,
                    heading=0.0,
                    timestamp=scene_context.timestamp,
                    obstacle_type=obj.class_name
                )
                obstacles.append(obstacle)

            # Plan path
            planned_path = self.path_planner.plan_path(
                start=ego_state,
                goal=goal_waypoint,
                obstacles=obstacles,
                vehicle_constraints=self.vehicle_constraints
            )

        return {
            'scene_context': scene_context,
            'trajectory_predictions': predictions,
            'planned_path': planned_path,
            'processing_timestamp': time.time()
        }

    def optimize_for_deployment(self, calibration_data=None):
        """
        Optimize all ML models for target hardware deployment

        Args:
            calibration_data: Data for quantization calibration

        Returns:
            Optimization results for all models
        """

        optimization_results = {}

        # Optimize scene understanding model
        if hasattr(self.scene_system, 'detection_model'):
            scene_model, scene_metrics = self.edge_optimizer.optimize_model(
                self.scene_system.detection_model,
                calibration_data
            )
            optimization_results['scene_understanding'] = {
                'model': scene_model,
                'metrics': scene_metrics
            }

        # Optimize trajectory prediction model
        if hasattr(self.trajectory_predictor, 'lstm_model'):
            traj_model, traj_metrics = self.edge_optimizer.optimize_model(
                self.trajectory_predictor.lstm_model,
                calibration_data
            )
            optimization_results['trajectory_prediction'] = {
                'model': traj_model,
                'metrics': traj_metrics
            }

        return optimization_results

    def get_system_metrics(self):
        """Get comprehensive system performance metrics"""

        metrics = {
            'scene_understanding': self.scene_system.get_performance_metrics(),
            'trajectory_prediction': self.trajectory_predictor.get_prediction_metrics(),
            'path_planning': self.path_planner.get_planning_metrics()
        }

        return metrics


# Example usage and testing
def create_test_pipeline():
    """Create test pipeline for development and validation"""

    # Define mid-range automotive ECU
    hardware = HardwareSpecs(
        ecu_type=ECUType.MID_RANGE,
        cpu_cores=4,
        cpu_frequency_mhz=1800,
        ram_mb=2048,
        gpu_present=False,
        npu_present=True,
        storage_type="emmc",
        thermal_limit_celsius=85,
        power_budget_watts=15.0
    )

    # Balanced optimization for production deployment
    opt_config = OptimizationConfig(
        target_latency_ms=50.0,
        target_fps=20,
        optimization_level=OptimizationLevel.BALANCED,
        quantization_enabled=True,
        pruning_enabled=True,
        input_resolution=(640, 384)
    )

    # Create pipeline
    pipeline = ADASMLPipeline(hardware, opt_config, use_stereo=True)

    return pipeline


# Module metadata
AUTOMOTIVE_CLASSES = [
    'road', 'vehicle', 'pedestrian', 'cyclist', 'traffic_sign',
    'traffic_light', 'lane_marking', 'barrier', 'construction'
]

PERFORMANCE_TARGETS = {
    'trajectory_prediction_fps': 20,
    'path_planning_hz': 10,
    'scene_understanding_fps': 15,
    'total_latency_ms': 100
}

SAFETY_REQUIREMENTS = {
    'trajectory_accuracy_threshold': 0.8,
    'path_safety_margin_m': 2.0,
    'object_detection_recall': 0.95,
    'false_positive_rate': 0.02
}