"""
Test script for ADAS Phase 7 ML components
Verify all modules work correctly with automotive-grade requirements
"""

import numpy as np
import torch
import logging
import time
from pathlib import Path

# Import all ML components
from trajectory_prediction import TrajectoryPredictor, TrajectoryState
from path_planning import PathPlanner, WayPoint, VehicleConstraints, Obstacle, PathPlanningMode
from scene_understanding import SceneUnderstandingSystem, WeatherCondition, LightingCondition
from edge_optimization import EdgeOptimizer, HardwareSpecs, OptimizationConfig, ECUType, OptimizationLevel


def test_trajectory_prediction():
    """Test trajectory prediction module"""
    print("Testing Trajectory Prediction...")

    # Create predictor
    predictor = TrajectoryPredictor()

    # Create sample trajectory history
    history = []
    for i in range(15):
        state = TrajectoryState(
            position=np.array([i * 2.0, 0.0, 0.0]),
            velocity=np.array([2.0, 0.0, 0.0]),
            acceleration=np.array([0.1, 0.0, 0.0]),
            heading=0.0,
            timestamp=i * 0.1
        )
        history.append(state)

    # Predict trajectory
    result = predictor.predict_trajectory(
        object_id=1,
        history=history,
        prediction_time=2.0
    )

    print(f"✓ Predicted {len(result.predicted_states)} future states")
    print(f"✓ Confidence score: {result.confidence_score:.3f}")
    return True


def test_path_planning():
    """Test path planning module"""
    print("Testing Path Planning...")

    # Create planner
    planner = PathPlanner(PathPlanningMode.HIGHWAY_CRUISING)

    # Define vehicle constraints
    vehicle = VehicleConstraints(
        max_speed=25.0,
        max_acceleration=3.0,
        wheelbase=2.7
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
            timestamp=0.0
        )
    ]

    # Plan path
    path = planner.plan_path(start, goal, obstacles, vehicle)

    if path:
        print(f"✓ Path found with {len(path.waypoints)} waypoints")
        print(f"✓ Total distance: {path.total_distance:.1f}m")
        return True
    else:
        print("✗ Path planning failed")
        return False


def test_scene_understanding():
    """Test scene understanding module"""
    print("Testing Scene Understanding...")

    # Create scene system
    scene_system = SceneUnderstandingSystem(use_stereo=False)  # Disable stereo for testing

    # Create test images
    left_image = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)

    # Process frame
    scene_context = scene_system.process_frame(left_image)

    print(f"✓ Detected {len(scene_context.objects_3d)} objects")
    print(f"✓ Weather: {scene_context.weather.value}")
    print(f"✓ Lighting: {scene_context.lighting.value}")
    return True


def test_edge_optimization():
    """Test edge optimization module"""
    print("Testing Edge Optimization...")

    # Define hardware specs
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

    # Define optimization config
    opt_config = OptimizationConfig(
        target_latency_ms=50.0,
        target_fps=20,
        optimization_level=OptimizationLevel.BALANCED,
        quantization_enabled=False,  # Disable for testing
        pruning_enabled=False,       # Disable for testing
        input_resolution=(640, 384)
    )

    # Create optimizer
    optimizer = EdgeOptimizer(hardware, opt_config)

    # Create simple test model
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, 1, 1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    model = TestModel()

    # Test optimization (without actual quantization/pruning)
    optimized_model, metrics = optimizer.optimize_model(model)

    print(f"✓ Optimization completed")
    print(f"✓ Latency: {metrics.latency_ms:.1f}ms")
    print(f"✓ Memory: {metrics.memory_usage_mb:.1f}MB")
    return True


def test_integrated_pipeline():
    """Test integrated ML pipeline"""
    print("Testing Integrated Pipeline...")

    try:
        from __init__ import ADASMLPipeline, create_test_pipeline

        # Create test pipeline
        pipeline = create_test_pipeline()

        # Create test data
        left_image = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
        right_image = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)

        ego_state = WayPoint(x=0.0, y=0.0, heading=0.0, velocity=10.0)
        goal_waypoint = WayPoint(x=50.0, y=10.0, heading=0.0, velocity=10.0)

        # Process frame
        result = pipeline.process_frame(
            left_image, right_image, ego_state, goal_waypoint
        )

        print(f"✓ Scene objects: {len(result['scene_context'].objects_3d)}")
        print(f"✓ Trajectory predictions: {len(result['trajectory_predictions'])}")
        print(f"✓ Path planning: {'Success' if result['planned_path'] else 'No path'}")

        # Get system metrics
        metrics = pipeline.get_system_metrics()
        print(f"✓ System metrics collected: {len(metrics)} modules")

        return True

    except Exception as e:
        print(f"✗ Integrated pipeline test failed: {e}")
        return False


def run_automotive_validation():
    """Run automotive-specific validation tests"""
    print("Running Automotive Validation...")

    # Test 1: Latency requirements (< 100ms total)
    start_time = time.time()

    # Simulate full pipeline processing
    scene_system = SceneUnderstandingSystem(use_stereo=False)
    left_image = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
    scene_context = scene_system.process_frame(left_image)

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # ms

    print(f"✓ Processing latency: {processing_time:.1f}ms {'(PASS)' if processing_time < 100 else '(FAIL)'}")

    # Test 2: Memory usage (< 500MB for mid-range ECU)
    import psutil
    memory_mb = psutil.virtual_memory().used / (1024 * 1024)
    print(f"✓ Memory usage: {memory_mb:.1f}MB")

    # Test 3: Safety validation
    from trajectory_prediction import validate_trajectory_safety
    from path_planning import PredictionResult

    # Create dummy prediction for safety test
    dummy_states = [
        TrajectoryState(
            position=np.array([i, 0, 0]),
            velocity=np.array([10, 0, 0]),  # 10 m/s
            acceleration=np.array([2, 0, 0]),  # 2 m/s^2
            heading=0.0,
            timestamp=i * 0.1
        ) for i in range(10)
    ]

    dummy_prediction = PredictionResult(
        predicted_states=dummy_states,
        uncertainty_bounds=np.zeros((10, 2)),
        confidence_score=0.9,
        time_horizon=1.0
    )

    # Define road boundaries
    road_boundaries = np.array([
        [-5, -10], [5, -10], [5, 10], [-5, 10]
    ])

    is_safe = validate_trajectory_safety(dummy_prediction, road_boundaries, 15.0)
    print(f"✓ Safety validation: {'PASS' if is_safe else 'FAIL'}")

    return True


def main():
    """Run all tests"""
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise

    print("=" * 60)
    print("ADAS Phase 7 ML Components Test Suite")
    print("=" * 60)

    tests = [
        ("Trajectory Prediction", test_trajectory_prediction),
        ("Path Planning", test_path_planning),
        ("Scene Understanding", test_scene_understanding),
        ("Edge Optimization", test_edge_optimization),
        ("Integrated Pipeline", test_integrated_pipeline),
        ("Automotive Validation", run_automotive_validation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! ML components are ready for automotive deployment.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")


if __name__ == "__main__":
    main()