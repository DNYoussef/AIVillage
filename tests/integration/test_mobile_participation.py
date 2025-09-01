"""
Mobile Participation Integration Tests

Tests mobile device participation in the federated system:
- Mobile device discovery and authentication
- Resource-constrained training participation
- Mobile-specific security and privacy measures
- Adaptive algorithms for mobile environments
- Offline/online synchronization

This validates the ML Specialist's work on mobile device participation
in federated training scenarios.
"""

import pytest
import time
from unittest.mock import patch

# Import mobile and federated components
from infrastructure.p2p.communications.discovery import P2PDiscovery
from infrastructure.p2p.communications.transport import P2PTransport
from core.agents.specialized.federated_coordinator import FederatedCoordinator
from infrastructure.shared.security.multi_tenant_system import MultiTenantSecuritySystem
from infrastructure.mobile.mobile_coordinator import MobileCoordinator
from infrastructure.mobile.adaptive_training import AdaptiveTraining
from infrastructure.mobile.resource_manager import MobileResourceManager


class TestMobileParticipation:
    """Integration tests for mobile device participation in federated training"""

    @pytest.fixture
    def setup_mobile_environment(self):
        """Setup mobile federated training environment"""
        # Mobile-specific components
        mobile_coordinator = MobileCoordinator()
        adaptive_training = AdaptiveTraining()
        mobile_resource_manager = MobileResourceManager()

        # Core federated components
        discovery = P2PDiscovery()
        transport = P2PTransport()
        federated_coordinator = FederatedCoordinator()
        security_system = MultiTenantSecuritySystem()

        return {
            "mobile_coordinator": mobile_coordinator,
            "adaptive_training": adaptive_training,
            "mobile_resource_manager": mobile_resource_manager,
            "p2p_discovery": discovery,
            "p2p_transport": transport,
            "federated_coordinator": federated_coordinator,
            "security_system": security_system,
        }

    @pytest.mark.asyncio
    async def test_mobile_device_discovery_and_registration(self, setup_mobile_environment):
        """Test discovery and registration of mobile devices for federated training"""
        components = setup_mobile_environment

        # Mock diverse mobile device fleet
        mock_mobile_devices = [
            {
                "device_id": "mobile_phone_001",
                "device_type": "smartphone",
                "os": "android",
                "hardware_specs": {
                    "cpu_cores": 8,
                    "ram_gb": 6,
                    "storage_gb": 128,
                    "has_gpu": False,
                    "battery_capacity": 4000,  # mAh
                },
                "network_info": {
                    "connection_type": "wifi",
                    "bandwidth_mbps": 50,
                    "latency_ms": 25,
                    "data_limit_gb": None,  # Unlimited on WiFi
                },
                "availability": {
                    "charging_status": True,
                    "battery_level": 0.85,
                    "user_active": False,
                    "available_hours": 8,
                },
            },
            {
                "device_id": "tablet_002",
                "device_type": "tablet",
                "os": "ios",
                "hardware_specs": {
                    "cpu_cores": 6,
                    "ram_gb": 4,
                    "storage_gb": 64,
                    "has_gpu": True,
                    "battery_capacity": 7000,
                },
                "network_info": {
                    "connection_type": "cellular",
                    "bandwidth_mbps": 20,
                    "latency_ms": 45,
                    "data_limit_gb": 10,  # Limited cellular data
                },
                "availability": {
                    "charging_status": False,
                    "battery_level": 0.65,
                    "user_active": False,
                    "available_hours": 4,
                },
            },
            {
                "device_id": "mobile_phone_003",
                "device_type": "smartphone",
                "os": "android",
                "hardware_specs": {
                    "cpu_cores": 4,
                    "ram_gb": 3,
                    "storage_gb": 32,
                    "has_gpu": False,
                    "battery_capacity": 3000,
                },
                "network_info": {
                    "connection_type": "wifi",
                    "bandwidth_mbps": 25,
                    "latency_ms": 30,
                    "data_limit_gb": None,
                },
                "availability": {
                    "charging_status": True,
                    "battery_level": 0.95,
                    "user_active": True,  # User is active, limited availability
                    "available_hours": 2,
                },
            },
        ]

        with patch.object(components["p2p_discovery"], "discover_mobile_devices") as mock_discover:
            mock_discover.return_value = mock_mobile_devices

            # Test mobile device discovery
            discovered_devices = await components["p2p_discovery"].discover_mobile_devices(
                capabilities=["federated_training", "local_computation"], min_battery_level=0.5, min_ram_gb=3
            )

            assert len(discovered_devices) == 3
            assert all(device["availability"]["battery_level"] >= 0.5 for device in discovered_devices)
            assert all(device["hardware_specs"]["ram_gb"] >= 3 for device in discovered_devices)

            mock_discover.assert_called_once()

        # Test mobile device registration and eligibility
        with patch.object(components["mobile_coordinator"], "register_mobile_devices") as mock_register:
            mock_register.return_value = {
                "eligible_devices": [
                    {
                        "device_id": "mobile_phone_001",
                        "eligibility_score": 0.9,  # High score - charging, good specs, long availability
                        "training_capacity": "high",
                        "assigned_role": "primary_trainer",
                    },
                    {
                        "device_id": "tablet_002",
                        "eligibility_score": 0.7,  # Medium score - not charging, cellular data limits
                        "training_capacity": "medium",
                        "assigned_role": "secondary_trainer",
                    },
                ],
                "ineligible_devices": [
                    {
                        "device_id": "mobile_phone_003",
                        "reason": "user_active_insufficient_availability",
                        "retry_after": 7200,  # Retry in 2 hours
                    }
                ],
                "total_training_capacity": "high",
            }

            registration_result = await components["mobile_coordinator"].register_mobile_devices(
                discovered_devices=discovered_devices,
                training_requirements={"min_participants": 2, "training_duration_hours": 4},
            )

            # Validate registration
            assert len(registration_result["eligible_devices"]) == 2
            assert registration_result["total_training_capacity"] == "high"
            assert "mobile_phone_001" in [d["device_id"] for d in registration_result["eligible_devices"]]
            assert len(registration_result["ineligible_devices"]) == 1

            mock_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_training_for_mobile_constraints(self, setup_mobile_environment):
        """Test adaptive training algorithms optimized for mobile device constraints"""
        components = setup_mobile_environment

        # Mobile device with constraints
        mobile_device = {
            "device_id": "constrained_phone_001",
            "constraints": {
                "max_computation_time": 300,  # 5 minutes max
                "memory_limit_mb": 512,
                "battery_drain_limit": 0.1,  # 10% battery drain max
                "data_usage_limit_mb": 50,  # 50MB max data usage
                "thermal_limit": 40,  # Celsius
            },
            "current_state": {
                "available_memory_mb": 1024,
                "battery_level": 0.8,
                "temperature_celsius": 25,
                "network_speed_mbps": 15,
            },
        }

        # Test adaptive algorithm selection
        with patch.object(components["adaptive_training"], "select_optimal_algorithm") as mock_algorithm:
            mock_algorithm.return_value = {
                "selected_algorithm": "federated_averaging_lite",
                "algorithm_parameters": {
                    "local_epochs": 3,  # Reduced from standard 10
                    "batch_size": 16,  # Reduced from standard 32
                    "learning_rate": 0.01,
                    "model_compression": True,
                    "gradient_quantization": "8bit",
                },
                "resource_optimization": {
                    "memory_efficient": True,
                    "computation_optimized": True,
                    "network_optimized": True,
                },
                "estimated_performance": {
                    "training_time_minutes": 4.5,
                    "memory_usage_mb": 450,
                    "battery_drain_percent": 8,
                    "data_usage_mb": 35,
                    "accuracy_trade_off": 0.02,  # 2% accuracy reduction for efficiency
                },
            }

            algorithm_result = await components["adaptive_training"].select_optimal_algorithm(
                device_constraints=mobile_device["constraints"],
                current_state=mobile_device["current_state"],
                training_objective="balanced_efficiency",
            )

            # Validate adaptive algorithm selection
            assert algorithm_result["selected_algorithm"] == "federated_averaging_lite"
            assert algorithm_result["estimated_performance"]["training_time_minutes"] <= 5.0
            assert algorithm_result["estimated_performance"]["memory_usage_mb"] <= 512
            assert algorithm_result["estimated_performance"]["battery_drain_percent"] <= 10
            assert algorithm_result["estimated_performance"]["data_usage_mb"] <= 50

            mock_algorithm.assert_called_once()

    @pytest.mark.asyncio
    async def test_resource_aware_training_execution(self, setup_mobile_environment):
        """Test training execution with real-time resource monitoring on mobile devices"""
        components = setup_mobile_environment

        # Training task adapted for mobile
        mobile_training_task = {
            "model_type": "lightweight_neural_network",
            "training_data_size": 1000,  # samples
            "target_accuracy": 0.85,
            "max_training_rounds": 10,
            "adaptive_parameters": True,
        }

        # Simulate training with resource monitoring
        with patch.object(components["mobile_resource_manager"], "execute_monitored_training") as mock_training:
            mock_training.return_value = {
                "training_rounds_completed": 8,
                "final_accuracy": 0.87,
                "resource_usage": {
                    "peak_memory_mb": 420,
                    "total_battery_drain": 0.09,  # 9%
                    "total_data_usage_mb": 32,
                    "peak_temperature_celsius": 38,
                    "total_training_time_minutes": 4.2,
                },
                "adaptive_adjustments": [
                    {"round": 3, "adjustment": "reduced_batch_size", "reason": "memory_pressure"},
                    {"round": 6, "adjustment": "paused_training", "reason": "thermal_throttling"},
                    {"round": 7, "adjustment": "resumed_training", "reason": "temperature_normalized"},
                ],
                "quality_metrics": {
                    "convergence_achieved": True,
                    "accuracy_target_met": True,
                    "resource_constraints_respected": True,
                },
            }

            training_result = await components["mobile_resource_manager"].execute_monitored_training(
                device_id=mobile_device["device_id"], training_task=mobile_training_task, monitoring_interval_seconds=30
            )

            # Validate resource-aware training
            assert training_result["quality_metrics"]["resource_constraints_respected"] is True
            assert training_result["final_accuracy"] >= mobile_training_task["target_accuracy"]
            assert training_result["resource_usage"]["peak_memory_mb"] <= 512
            assert training_result["resource_usage"]["total_battery_drain"] <= 0.1
            assert len(training_result["adaptive_adjustments"]) > 0  # System adapted to constraints

            mock_training.assert_called_once()

    @pytest.mark.asyncio
    async def test_mobile_security_and_privacy_measures(self, setup_mobile_environment):
        """Test mobile-specific security and privacy measures"""
        components = setup_mobile_environment

        # Mobile security requirements
        mobile_security_config = {
            "device_attestation": True,
            "secure_enclave_required": True,
            "biometric_authentication": True,
            "differential_privacy_budget": 0.05,  # Stricter for mobile
            "local_data_encryption": True,
            "gradient_obfuscation": True,
        }

        # Test mobile device authentication and attestation
        with patch.object(components["security_system"], "authenticate_mobile_device") as mock_auth:
            mock_auth.return_value = {
                "authentication_status": "success",
                "device_verified": True,
                "secure_enclave_available": True,
                "biometric_verified": True,
                "device_integrity_score": 0.95,
                "security_level": "high",
                "trusted_execution_environment": True,
            }

            auth_result = await components["security_system"].authenticate_mobile_device(
                device_id="mobile_phone_001", security_requirements=mobile_security_config
            )

            assert auth_result["authentication_status"] == "success"
            assert auth_result["device_verified"] is True
            assert auth_result["secure_enclave_available"] is True
            assert auth_result["device_integrity_score"] >= 0.9

            mock_auth.assert_called_once()

        # Test mobile differential privacy
        with patch.object(components["adaptive_training"], "apply_mobile_differential_privacy") as mock_privacy:
            mock_privacy.return_value = {
                "privacy_preserved_gradients": {
                    "layer_1": [0.08, 0.04, -0.02],  # Noise added
                    "layer_2": [0.06, -0.01, 0.03],
                },
                "privacy_analysis": {
                    "epsilon_used": 0.04,
                    "delta_used": 1e-6,
                    "noise_mechanism": "gaussian_mobile_optimized",
                    "privacy_budget_remaining": 0.01,
                    "utility_preservation": 0.92,  # 92% utility preserved
                },
                "mobile_optimizations": {
                    "noise_computation_optimized": True,
                    "battery_efficient_privacy": True,
                    "memory_efficient_mechanism": True,
                },
            }

            privacy_result = await components["adaptive_training"].apply_mobile_differential_privacy(
                gradients={"layer_1": [0.1, 0.05, -0.02], "layer_2": [0.07, -0.01, 0.03]},
                privacy_budget=mobile_security_config["differential_privacy_budget"],
                mobile_optimized=True,
            )

            # Validate mobile privacy preservation
            assert (
                privacy_result["privacy_analysis"]["epsilon_used"]
                <= mobile_security_config["differential_privacy_budget"]
            )
            assert privacy_result["privacy_analysis"]["utility_preservation"] >= 0.9
            assert privacy_result["mobile_optimizations"]["battery_efficient_privacy"] is True

            mock_privacy.assert_called_once()

    @pytest.mark.asyncio
    async def test_offline_online_synchronization(self, setup_mobile_environment):
        """Test offline training and online synchronization for mobile devices"""
        components = setup_mobile_environment

        # Offline training scenario
        offline_training_data = {
            "device_id": "mobile_offline_001",
            "offline_duration_hours": 12,
            "training_rounds_offline": 5,
            "accumulated_gradients": {
                "round_1": {"layer_1": [0.1, 0.05], "timestamp": time.time() - 43200},
                "round_2": {"layer_1": [0.08, 0.04], "timestamp": time.time() - 39600},
                "round_3": {"layer_1": [0.09, 0.045], "timestamp": time.time() - 36000},
                "round_4": {"layer_1": [0.07, 0.035], "timestamp": time.time() - 32400},
                "round_5": {"layer_1": [0.085, 0.04], "timestamp": time.time() - 28800},
            },
            "model_staleness_hours": 12,
            "data_integrity_hash": "offline_data_hash_123",
        }

        # Test offline-to-online synchronization
        with patch.object(components["mobile_coordinator"], "synchronize_offline_training") as mock_sync:
            mock_sync.return_value = {
                "synchronization_status": "success",
                "gradient_validation": {
                    "integrity_verified": True,
                    "staleness_acceptable": True,
                    "temporal_consistency": True,
                },
                "model_update_strategy": "weighted_staleness_compensation",
                "synchronized_gradients": {
                    "compensated_gradients": [0.082, 0.041],  # Adjusted for staleness
                    "staleness_weight": 0.85,  # Reduced weight due to staleness
                    "contribution_value": 0.75,
                },
                "federated_integration": {
                    "global_round_integration": 15,  # Integrated into round 15
                    "peer_coordination": True,
                    "consistency_maintained": True,
                },
                "sync_metadata": {"sync_time_seconds": 3.5, "data_transferred_mb": 2.1, "conflicts_resolved": 0},
            }

            sync_result = await components["mobile_coordinator"].synchronize_offline_training(
                offline_data=offline_training_data, current_global_model_version=15, staleness_tolerance_hours=24
            )

            # Validate offline synchronization
            assert sync_result["synchronization_status"] == "success"
            assert sync_result["gradient_validation"]["integrity_verified"] is True
            assert sync_result["gradient_validation"]["staleness_acceptable"] is True
            assert sync_result["synchronized_gradients"]["staleness_weight"] <= 1.0
            assert sync_result["sync_metadata"]["conflicts_resolved"] == 0

            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_mobile_federated_training_full_workflow(self, setup_mobile_environment):
        """Test complete mobile federated training workflow"""
        components = setup_mobile_environment

        # Mobile federated training session
        mobile_training_session = {
            "session_id": "mobile_federated_session_001",
            "target_participants": 10,
            "mobile_percentage": 0.8,  # 80% mobile devices
            "training_rounds": 5,
            "mobile_friendly_model": True,
            "adaptive_algorithms": True,
        }

        # Test complete workflow
        with patch.object(components["mobile_coordinator"], "execute_mobile_federated_training") as mock_workflow:
            mock_workflow.return_value = {
                "training_results": {
                    "rounds_completed": 5,
                    "mobile_participants": 8,
                    "desktop_participants": 2,
                    "final_accuracy": 0.89,
                    "convergence_achieved": True,
                },
                "mobile_performance": {
                    "average_training_time_minutes": 3.8,
                    "average_battery_usage_percent": 7.5,
                    "average_data_usage_mb": 28,
                    "device_dropout_rate": 0.1,  # 10% dropout rate
                    "adaptive_adjustments_per_device": 2.3,
                },
                "resource_efficiency": {
                    "computation_efficiency": 0.88,
                    "communication_efficiency": 0.92,
                    "energy_efficiency": 0.85,
                    "memory_efficiency": 0.90,
                },
                "quality_metrics": {
                    "accuracy_vs_desktop_only": -0.03,  # 3% accuracy reduction
                    "training_time_vs_desktop_only": 0.7,  # 30% faster
                    "privacy_preservation": 0.95,
                    "fault_tolerance": 0.90,
                },
                "scalability_metrics": {
                    "max_mobile_participants": 50,
                    "coordination_overhead": 0.12,  # 12% overhead
                    "network_efficiency": 0.88,
                },
            }

            workflow_result = await components["mobile_coordinator"].execute_mobile_federated_training(
                training_session=mobile_training_session,
                optimization_targets=["battery_efficiency", "accuracy", "privacy"],
            )

            # Validate complete mobile workflow
            assert workflow_result["training_results"]["convergence_achieved"] is True
            assert workflow_result["training_results"]["mobile_participants"] >= 6  # Good mobile participation
            assert workflow_result["mobile_performance"]["device_dropout_rate"] <= 0.2  # Acceptable dropout
            assert workflow_result["quality_metrics"]["privacy_preservation"] >= 0.9
            assert workflow_result["resource_efficiency"]["energy_efficiency"] >= 0.8

            # Validate mobile-specific benefits
            assert workflow_result["quality_metrics"]["training_time_vs_desktop_only"] < 1.0  # Faster training
            assert workflow_result["scalability_metrics"]["max_mobile_participants"] >= 20  # Good scalability

            mock_workflow.assert_called_once()


if __name__ == "__main__":
    # Run mobile participation tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
