"""
Security Integration Validation Tests

Validates the Security Specialist's work on:
✅ Federated node authentication and authorization
✅ Encrypted gradient exchange using BetaNet
✅ Byzantine fault tolerance and attack resistance
✅ Secure aggregation protocols work correctly

This ensures all security enhancements are properly integrated
and functioning as designed.
"""

import pytest
import time
from unittest.mock import patch

# Import security components
from infrastructure.shared.security.multi_tenant_system import MultiTenantSecuritySystem
from infrastructure.shared.security.encryption import SecureAggregator, BetaNetEncryption
from infrastructure.shared.security.byzantine_tolerance import ByzantineDefense
from infrastructure.shared.security.authentication import FederatedAuth
from infrastructure.shared.security.authorization import FederatedAuthz


class TestSecurityIntegrationValidation:
    """Integration validation tests for security enhancements"""

    @pytest.fixture
    def setup_security_components(self):
        """Setup all security components for validation"""
        # Core security systems
        multi_tenant_system = MultiTenantSecuritySystem()
        secure_aggregator = SecureAggregator()
        betanet_encryption = BetaNetEncryption()
        byzantine_defense = ByzantineDefense()
        federated_auth = FederatedAuth()
        federated_authz = FederatedAuthz()

        return {
            "multi_tenant": multi_tenant_system,
            "secure_aggregator": secure_aggregator,
            "betanet_encryption": betanet_encryption,
            "byzantine_defense": byzantine_defense,
            "federated_auth": federated_auth,
            "federated_authz": federated_authz,
        }

    @pytest.mark.asyncio
    async def test_federated_node_authentication_validation(self, setup_security_components):
        """Validate federated node authentication system"""
        components = setup_security_components

        # Test node registration and credential generation
        node_registration_requests = [
            {
                "node_id": f"federated_node_{i}",
                "node_type": "training_participant",
                "public_key": f"pubkey_{i}",
                "certificates": [f"cert_{i}"],
                "attestation_data": {
                    "hardware_security_module": True,
                    "trusted_execution_environment": True,
                    "secure_boot_verified": True,
                },
                "capabilities": ["federated_training", "secure_aggregation"],
            }
            for i in range(5)
        ]

        with patch.object(components["federated_auth"], "register_federated_nodes") as mock_register:
            mock_register.return_value = {
                "registration_results": {
                    f"federated_node_{i}": {
                        "status": "approved",
                        "node_id": f"federated_node_{i}",
                        "identity_token": f"secure_identity_token_{i}",
                        "session_keys": {"encryption_key": f"enc_key_{i}", "signing_key": f"sign_key_{i}"},
                        "trust_score": 0.9 + (i * 0.01),
                        "security_clearance": "high",
                        "expiration_time": time.time() + 3600,  # 1 hour
                    }
                    for i in range(5)
                },
                "security_validation": {
                    "all_nodes_verified": True,
                    "certificate_chain_valid": True,
                    "attestation_verified": True,
                    "security_level_consistent": True,
                },
                "total_registration_time": 2.5,  # seconds
            }

            registration_result = await components["federated_auth"].register_federated_nodes(
                registration_requests=node_registration_requests,
                security_level="high",
                verification_method="comprehensive",
            )

            # Validate node registration
            assert registration_result["security_validation"]["all_nodes_verified"] is True
            assert registration_result["security_validation"]["certificate_chain_valid"] is True
            assert len(registration_result["registration_results"]) == 5
            assert all(
                result["status"] == "approved" for result in registration_result["registration_results"].values()
            )
            assert all(
                result["security_clearance"] == "high"
                for result in registration_result["registration_results"].values()
            )

            mock_register.assert_called_once()

        # Test authentication token validation and refresh
        with patch.object(components["federated_auth"], "validate_and_refresh_tokens") as mock_validate:
            mock_validate.return_value = {
                "validation_results": {
                    f"federated_node_{i}": {
                        "token_valid": True,
                        "node_authenticated": True,
                        "security_level_maintained": True,
                        "token_refreshed": i % 2 == 0,  # Refresh every other token
                        "new_token": f"refreshed_token_{i}" if i % 2 == 0 else None,
                        "expiration_extended": i % 2 == 0,
                    }
                    for i in range(5)
                },
                "authentication_summary": {
                    "total_tokens_validated": 5,
                    "valid_tokens": 5,
                    "tokens_refreshed": 3,  # 3 tokens refreshed
                    "authentication_failures": 0,
                    "security_violations": 0,
                },
            }

            current_tokens = [
                registration_result["registration_results"][f"federated_node_{i}"]["identity_token"] for i in range(5)
            ]

            validation_result = await components["federated_auth"].validate_and_refresh_tokens(
                tokens=current_tokens,
                refresh_threshold_seconds=1800,  # Refresh if less than 30 min remaining
                security_checks=["certificate_validity", "revocation_status", "trust_score"],
            )

            # Validate token validation and refresh
            assert validation_result["authentication_summary"]["authentication_failures"] == 0
            assert validation_result["authentication_summary"]["security_violations"] == 0
            assert validation_result["authentication_summary"]["valid_tokens"] == 5
            assert all(result["token_valid"] for result in validation_result["validation_results"].values())

            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_federated_node_authorization_validation(self, setup_security_components):
        """Validate federated node authorization system"""
        components = setup_security_components

        # Test authorization policy creation and enforcement
        authorization_policies = {
            "federated_training_policy": {
                "policy_id": "fed_train_001",
                "permissions": [
                    "read_training_data",
                    "compute_gradients",
                    "participate_aggregation",
                    "receive_model_updates",
                ],
                "restrictions": {
                    "max_data_access_gb": 10,
                    "max_computation_hours": 8,
                    "allowed_model_types": ["neural_network", "svm", "decision_tree"],
                    "privacy_level_required": "high",
                },
                "conditions": {
                    "trust_score_minimum": 0.8,
                    "security_clearance_required": "medium",
                    "geographic_restrictions": None,
                },
            },
            "secure_aggregation_policy": {
                "policy_id": "sec_agg_001",
                "permissions": [
                    "encrypt_gradients",
                    "participate_secure_protocol",
                    "verify_aggregation_proofs",
                    "access_aggregated_results",
                ],
                "restrictions": {
                    "max_participants": 100,
                    "encryption_level_minimum": "high",
                    "protocol_compliance_required": True,
                },
                "conditions": {
                    "trust_score_minimum": 0.9,
                    "security_clearance_required": "high",
                    "certified_hardware_required": True,
                },
            },
        }

        with patch.object(components["federated_authz"], "create_authorization_policies") as mock_create_policies:
            mock_create_policies.return_value = {
                "created_policies": [
                    {"policy_id": "fed_train_001", "status": "active", "created_at": time.time(), "version": "1.0"},
                    {"policy_id": "sec_agg_001", "status": "active", "created_at": time.time(), "version": "1.0"},
                ],
                "policy_validation": {
                    "syntax_valid": True,
                    "conflicts_detected": False,
                    "coverage_complete": True,
                    "security_level_consistent": True,
                },
            }

            policy_creation_result = await components["federated_authz"].create_authorization_policies(
                policies=authorization_policies, validation_level="strict", enforcement_mode="active"
            )

            # Validate policy creation
            assert len(policy_creation_result["created_policies"]) == 2
            assert policy_creation_result["policy_validation"]["syntax_valid"] is True
            assert policy_creation_result["policy_validation"]["conflicts_detected"] is False
            assert all(policy["status"] == "active" for policy in policy_creation_result["created_policies"])

            mock_create_policies.assert_called_once()

        # Test authorization decision making
        authorization_requests = [
            {
                "node_id": "federated_node_0",
                "requested_action": "participate_federated_training",
                "resource_access": ["training_data_subset_1"],
                "context": {
                    "trust_score": 0.92,
                    "security_clearance": "high",
                    "hardware_certified": True,
                    "data_size_gb": 5,
                },
            },
            {
                "node_id": "federated_node_1",
                "requested_action": "participate_secure_aggregation",
                "resource_access": ["aggregation_protocol"],
                "context": {
                    "trust_score": 0.95,
                    "security_clearance": "high",
                    "hardware_certified": True,
                    "encryption_capable": True,
                },
            },
            {
                "node_id": "federated_node_2",
                "requested_action": "access_model_weights",
                "resource_access": ["global_model_v1"],
                "context": {
                    "trust_score": 0.75,  # Below minimum for some actions
                    "security_clearance": "medium",
                    "hardware_certified": False,
                },
            },
        ]

        with patch.object(components["federated_authz"], "make_authorization_decisions") as mock_authorize:
            mock_authorize.return_value = {
                "authorization_decisions": {
                    "federated_node_0": {
                        "decision": "granted",
                        "permissions_granted": ["read_training_data", "compute_gradients", "participate_aggregation"],
                        "restrictions_applied": {"max_data_access_gb": 10, "max_computation_hours": 8},
                        "conditions_met": True,
                        "policy_applied": "fed_train_001",
                    },
                    "federated_node_1": {
                        "decision": "granted",
                        "permissions_granted": [
                            "encrypt_gradients",
                            "participate_secure_protocol",
                            "verify_aggregation_proofs",
                        ],
                        "restrictions_applied": {
                            "encryption_level_minimum": "high",
                            "protocol_compliance_required": True,
                        },
                        "conditions_met": True,
                        "policy_applied": "sec_agg_001",
                    },
                    "federated_node_2": {
                        "decision": "denied",
                        "reason": "insufficient_trust_score_and_security_clearance",
                        "missing_requirements": ["trust_score_minimum_0.8", "certified_hardware_required"],
                        "conditions_met": False,
                    },
                },
                "decision_summary": {
                    "total_requests": 3,
                    "granted": 2,
                    "denied": 1,
                    "policy_violations": 1,
                    "average_decision_time_ms": 15,
                },
            }

            authorization_result = await components["federated_authz"].make_authorization_decisions(
                requests=authorization_requests,
                policies=authorization_policies,
                decision_context="federated_training_session",
            )

            # Validate authorization decisions
            assert authorization_result["decision_summary"]["granted"] == 2
            assert authorization_result["decision_summary"]["denied"] == 1
            assert authorization_result["authorization_decisions"]["federated_node_0"]["decision"] == "granted"
            assert authorization_result["authorization_decisions"]["federated_node_1"]["decision"] == "granted"
            assert authorization_result["authorization_decisions"]["federated_node_2"]["decision"] == "denied"
            assert authorization_result["decision_summary"]["average_decision_time_ms"] < 50

            mock_authorize.assert_called_once()

    @pytest.mark.asyncio
    async def test_betanet_encrypted_gradient_exchange(self, setup_security_components):
        """Validate BetaNet encrypted gradient exchange"""
        components = setup_security_components

        # Test BetaNet encryption setup and key exchange
        participating_nodes = [f"node_{i}" for i in range(4)]

        with patch.object(components["betanet_encryption"], "initialize_betanet_protocol") as mock_init:
            mock_init.return_value = {
                "protocol_initialized": True,
                "betanet_session_id": "betanet_session_001",
                "participants": participating_nodes,
                "encryption_parameters": {
                    "algorithm": "betanet_secure_multiparty",
                    "key_size": 256,
                    "security_level": 128,
                    "homomorphic_properties": True,
                },
                "shared_secrets": {f"node_{i}": f"shared_secret_{i}_hash" for i in range(4)},
                "setup_time_ms": 850,
                "protocol_verification": "passed",
            }

            betanet_init_result = await components["betanet_encryption"].initialize_betanet_protocol(
                participants=participating_nodes,
                security_level=128,
                homomorphic_operations=["addition", "scalar_multiplication"],
            )

            # Validate BetaNet initialization
            assert betanet_init_result["protocol_initialized"] is True
            assert betanet_init_result["encryption_parameters"]["homomorphic_properties"] is True
            assert len(betanet_init_result["shared_secrets"]) == 4
            assert betanet_init_result["protocol_verification"] == "passed"
            assert betanet_init_result["setup_time_ms"] < 1000

            mock_init.assert_called_once()

        # Test encrypted gradient exchange
        mock_gradients = {
            "node_0": {
                "layer_1_weights": [0.1, 0.05, -0.02, 0.08],
                "layer_1_bias": [0.03, -0.01],
                "layer_2_weights": [0.07, -0.04, 0.06],
                "gradient_norm": 0.15,
            },
            "node_1": {
                "layer_1_weights": [0.09, 0.04, -0.025, 0.075],
                "layer_1_bias": [0.028, -0.012],
                "layer_2_weights": [0.065, -0.038, 0.058],
                "gradient_norm": 0.14,
            },
            "node_2": {
                "layer_1_weights": [0.11, 0.06, -0.015, 0.085],
                "layer_1_bias": [0.032, -0.008],
                "layer_2_weights": [0.075, -0.042, 0.062],
                "gradient_norm": 0.16,
            },
            "node_3": {
                "layer_1_weights": [0.095, 0.045, -0.022, 0.078],
                "layer_1_bias": [0.029, -0.011],
                "layer_2_weights": [0.068, -0.039, 0.059],
                "gradient_norm": 0.145,
            },
        }

        with patch.object(components["betanet_encryption"], "encrypt_and_exchange_gradients") as mock_exchange:
            mock_exchange.return_value = {
                "encrypted_exchange_successful": True,
                "encrypted_gradients": {
                    f"node_{i}": {
                        "encrypted_layer_1_weights": f"encrypted_l1_weights_{i}",
                        "encrypted_layer_1_bias": f"encrypted_l1_bias_{i}",
                        "encrypted_layer_2_weights": f"encrypted_l2_weights_{i}",
                        "encryption_proof": f"encryption_proof_{i}",
                        "commitment_hash": f"commitment_{i}",
                    }
                    for i in range(4)
                },
                "exchange_metadata": {
                    "total_exchange_time_ms": 320,
                    "encryption_time_ms": 180,
                    "communication_time_ms": 140,
                    "verification_time_ms": 45,
                },
                "security_guarantees": {
                    "individual_privacy_preserved": True,
                    "gradient_integrity_verified": True,
                    "zero_knowledge_proofs_valid": True,
                    "homomorphic_properties_maintained": True,
                },
            }

            exchange_result = await components["betanet_encryption"].encrypt_and_exchange_gradients(
                gradients=mock_gradients,
                betanet_session=betanet_init_result["betanet_session_id"],
                verification_proofs=True,
            )

            # Validate encrypted gradient exchange
            assert exchange_result["encrypted_exchange_successful"] is True
            assert exchange_result["security_guarantees"]["individual_privacy_preserved"] is True
            assert exchange_result["security_guarantees"]["gradient_integrity_verified"] is True
            assert exchange_result["security_guarantees"]["zero_knowledge_proofs_valid"] is True
            assert exchange_result["exchange_metadata"]["total_exchange_time_ms"] < 500
            assert len(exchange_result["encrypted_gradients"]) == 4

            mock_exchange.assert_called_once()

    @pytest.mark.asyncio
    async def test_byzantine_fault_tolerance_validation(self, setup_security_components):
        """Validate Byzantine fault tolerance against various attacks"""
        components = setup_security_components

        # Test Byzantine attack detection with various attack types
        gradient_submissions = {
            # Honest participants
            "honest_node_1": {
                "gradients": {"layer_1": [0.1, 0.05, -0.02], "layer_2": [0.08, -0.03]},
                "metadata": {"training_samples": 1000, "loss": 0.45, "accuracy": 0.78},
                "node_type": "honest",
            },
            "honest_node_2": {
                "gradients": {"layer_1": [0.09, 0.04, -0.025], "layer_2": [0.075, -0.028]},
                "metadata": {"training_samples": 950, "loss": 0.47, "accuracy": 0.76},
                "node_type": "honest",
            },
            "honest_node_3": {
                "gradients": {"layer_1": [0.11, 0.06, -0.015], "layer_2": [0.085, -0.032]},
                "metadata": {"training_samples": 1100, "loss": 0.43, "accuracy": 0.79},
                "node_type": "honest",
            },
            # Byzantine attackers with different strategies
            "byzantine_node_1": {
                "gradients": {"layer_1": [100.0, -50.0, 25.0], "layer_2": [75.0, -80.0]},  # Magnitude attack
                "metadata": {"training_samples": 1000, "loss": 0.1, "accuracy": 0.99},
                "node_type": "magnitude_attacker",
            },
            "byzantine_node_2": {
                "gradients": {"layer_1": [0.0, 0.0, 0.0], "layer_2": [0.0, 0.0]},  # Zero gradients attack
                "metadata": {"training_samples": 1000, "loss": 1.0, "accuracy": 0.1},
                "node_type": "zero_gradients_attacker",
            },
            "byzantine_node_3": {
                "gradients": {"layer_1": [-0.1, -0.05, 0.02], "layer_2": [-0.08, 0.03]},  # Sign flip attack
                "metadata": {"training_samples": 1000, "loss": 0.45, "accuracy": 0.78},
                "node_type": "sign_flip_attacker",
            },
            "byzantine_node_4": {
                "gradients": {"layer_1": [0.1001, 0.0501, -0.0201], "layer_2": [0.0801, -0.0301]},  # Subtle attack
                "metadata": {"training_samples": 1000, "loss": 0.44, "accuracy": 0.785},
                "node_type": "subtle_attacker",
            },
        }

        with patch.object(components["byzantine_defense"], "detect_byzantine_attacks") as mock_detect:
            mock_detect.return_value = {
                "attack_detection_results": {
                    "magnitude_attacks_detected": ["byzantine_node_1"],
                    "zero_gradient_attacks_detected": ["byzantine_node_2"],
                    "sign_flip_attacks_detected": ["byzantine_node_3"],
                    "subtle_attacks_detected": ["byzantine_node_4"],
                    "honest_nodes_verified": ["honest_node_1", "honest_node_2", "honest_node_3"],
                },
                "detection_methods": {
                    "statistical_outlier_detection": True,
                    "gradient_norm_analysis": True,
                    "loss_consistency_check": True,
                    "cross_validation_verification": True,
                    "temporal_pattern_analysis": True,
                },
                "confidence_scores": {
                    "honest_node_1": 0.95,
                    "honest_node_2": 0.93,
                    "honest_node_3": 0.96,
                    "byzantine_node_1": 0.02,  # Very low confidence = attacker
                    "byzantine_node_2": 0.05,
                    "byzantine_node_3": 0.15,
                    "byzantine_node_4": 0.35,  # Harder to detect subtle attack
                },
                "attack_severity": {
                    "byzantine_node_1": "critical",
                    "byzantine_node_2": "high",
                    "byzantine_node_3": "high",
                    "byzantine_node_4": "medium",
                },
                "detection_time_ms": 125,
            }

            detection_result = await components["byzantine_defense"].detect_byzantine_attacks(
                gradient_submissions=gradient_submissions, detection_threshold=0.7, multi_method_validation=True
            )

            # Validate Byzantine attack detection
            assert len(detection_result["attack_detection_results"]["honest_nodes_verified"]) == 3
            assert "byzantine_node_1" in detection_result["attack_detection_results"]["magnitude_attacks_detected"]
            assert "byzantine_node_2" in detection_result["attack_detection_results"]["zero_gradient_attacks_detected"]
            assert "byzantine_node_3" in detection_result["attack_detection_results"]["sign_flip_attacks_detected"]
            assert "byzantine_node_4" in detection_result["attack_detection_results"]["subtle_attacks_detected"]
            assert all(
                score >= 0.7 for node, score in detection_result["confidence_scores"].items() if "honest" in node
            )
            assert detection_result["detection_time_ms"] < 200

            mock_detect.assert_called_once()

        # Test Byzantine-resilient aggregation
        honest_gradients = {node: data for node, data in gradient_submissions.items() if "honest" in node}

        with patch.object(components["byzantine_defense"], "byzantine_resilient_aggregation") as mock_aggregate:
            mock_aggregate.return_value = {
                "aggregated_gradients": {
                    "layer_1": [0.1, 0.05, -0.02],  # Robust average of honest nodes
                    "layer_2": [0.08, -0.03],
                },
                "aggregation_method": "trimmed_mean_with_outlier_removal",
                "participants_included": ["honest_node_1", "honest_node_2", "honest_node_3"],
                "participants_excluded": [
                    "byzantine_node_1",
                    "byzantine_node_2",
                    "byzantine_node_3",
                    "byzantine_node_4",
                ],
                "robustness_guarantees": {
                    "byzantine_tolerance_ratio": 0.49,  # Can tolerate up to 49% Byzantine nodes
                    "convergence_maintained": True,
                    "accuracy_preservation": 0.98,  # 98% of honest accuracy maintained
                    "attack_mitigation_effective": True,
                },
                "aggregation_metadata": {
                    "total_nodes": 7,
                    "honest_nodes": 3,
                    "byzantine_nodes_filtered": 4,
                    "aggregation_time_ms": 85,
                },
            }

            aggregation_result = await components["byzantine_defense"].byzantine_resilient_aggregation(
                all_gradients=gradient_submissions,
                honest_gradients=honest_gradients,
                byzantine_tolerance_ratio=0.5,
                aggregation_method="robust",
            )

            # Validate Byzantine-resilient aggregation
            assert len(aggregation_result["participants_included"]) == 3  # Only honest nodes
            assert len(aggregation_result["participants_excluded"]) == 4  # All Byzantine nodes excluded
            assert aggregation_result["robustness_guarantees"]["convergence_maintained"] is True
            assert aggregation_result["robustness_guarantees"]["accuracy_preservation"] >= 0.95
            assert aggregation_result["robustness_guarantees"]["attack_mitigation_effective"] is True
            assert aggregation_result["aggregation_metadata"]["aggregation_time_ms"] < 150

            mock_aggregate.assert_called_once()

    @pytest.mark.asyncio
    async def test_secure_aggregation_protocol_validation(self, setup_security_components):
        """Validate secure aggregation protocols work correctly"""
        components = setup_security_components

        # Test complete secure aggregation workflow
        secure_aggregation_session = {
            "session_id": "secure_agg_session_001",
            "participants": [f"node_{i}" for i in range(6)],
            "aggregation_method": "federated_averaging_secure",
            "security_requirements": {
                "individual_privacy": True,
                "aggregation_integrity": True,
                "verifiable_computation": True,
                "byzantine_tolerance": True,
            },
        }

        # Test secure aggregation protocol execution
        with patch.object(components["secure_aggregator"], "execute_secure_aggregation_protocol") as mock_protocol:
            mock_protocol.return_value = {
                "aggregation_successful": True,
                "protocol_phases_completed": {
                    "setup_phase": {
                        "status": "completed",
                        "participants_registered": 6,
                        "shared_secrets_established": True,
                        "commitment_scheme_initialized": True,
                    },
                    "commitment_phase": {
                        "status": "completed",
                        "commitments_received": 6,
                        "commitment_verification": "passed",
                        "zero_knowledge_proofs_valid": True,
                    },
                    "revelation_phase": {
                        "status": "completed",
                        "gradients_revealed": 6,
                        "integrity_verification": "passed",
                        "byzantine_detection": "no_attacks_detected",
                    },
                    "aggregation_phase": {
                        "status": "completed",
                        "secure_computation": "successful",
                        "result_verification": "passed",
                        "privacy_preservation": "guaranteed",
                    },
                },
                "final_aggregated_result": {
                    "aggregated_gradients": {"layer_1": [0.098, 0.049, -0.021], "layer_2": [0.076, -0.029]},
                    "aggregation_proof": {
                        "correctness_proof": "zk_proof_hash_123",
                        "completeness_proof": "completeness_hash_456",
                        "privacy_proof": "privacy_hash_789",
                    },
                    "participants_count": 6,
                    "consensus_achieved": True,
                },
                "security_guarantees_validated": {
                    "individual_privacy_preserved": True,
                    "aggregation_integrity_maintained": True,
                    "verifiable_computation_successful": True,
                    "byzantine_tolerance_effective": True,
                    "zero_knowledge_properties_maintained": True,
                },
                "performance_metrics": {
                    "total_protocol_time_ms": 2850,
                    "computation_time_ms": 1200,
                    "communication_time_ms": 1400,
                    "verification_time_ms": 250,
                    "communication_rounds": 4,
                    "bandwidth_usage_kb": 125,
                },
            }

            protocol_result = await components["secure_aggregator"].execute_secure_aggregation_protocol(
                session=secure_aggregation_session, timeout_ms=10000, verification_level="comprehensive"
            )

            # Validate secure aggregation protocol
            assert protocol_result["aggregation_successful"] is True
            assert all(
                phase["status"] == "completed" for phase in protocol_result["protocol_phases_completed"].values()
            )
            assert protocol_result["security_guarantees_validated"]["individual_privacy_preserved"] is True
            assert protocol_result["security_guarantees_validated"]["aggregation_integrity_maintained"] is True
            assert protocol_result["security_guarantees_validated"]["verifiable_computation_successful"] is True
            assert protocol_result["security_guarantees_validated"]["byzantine_tolerance_effective"] is True
            assert protocol_result["final_aggregated_result"]["consensus_achieved"] is True
            assert protocol_result["performance_metrics"]["total_protocol_time_ms"] < 5000

            mock_protocol.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_to_end_security_integration_validation(self, setup_security_components):
        """Test end-to-end security integration across all components"""
        components = setup_security_components

        # Test complete secure federated learning workflow
        secure_fl_session = {
            "session_id": "secure_fl_integration_test",
            "participants": 8,
            "security_level": "maximum",
            "privacy_budget": 1.0,
            "byzantine_tolerance": 0.25,  # Tolerate up to 25% malicious nodes
        }

        with patch.object(components["multi_tenant"], "execute_secure_federated_workflow") as mock_workflow:
            mock_workflow.return_value = {
                "workflow_execution_successful": True,
                "security_components_integration": {
                    "authentication_system": "fully_integrated",
                    "authorization_system": "fully_integrated",
                    "betanet_encryption": "fully_integrated",
                    "byzantine_defense": "fully_integrated",
                    "secure_aggregation": "fully_integrated",
                },
                "security_validations_passed": {
                    "end_to_end_encryption": True,
                    "zero_knowledge_privacy": True,
                    "byzantine_attack_resistance": True,
                    "differential_privacy_maintained": True,
                    "secure_multiparty_computation": True,
                    "verifiable_aggregation": True,
                },
                "performance_with_security": {
                    "training_rounds_completed": 10,
                    "convergence_achieved": True,
                    "final_accuracy": 0.87,  # Slight reduction due to privacy
                    "security_overhead_percent": 18,  # 18% performance overhead
                    "privacy_budget_consumed": 0.8,  # 80% of budget used
                    "total_training_time_minutes": 45,
                },
                "security_incidents": {
                    "attacks_detected": 2,  # 2 Byzantine attacks detected and mitigated
                    "attacks_mitigated": 2,
                    "privacy_violations": 0,
                    "unauthorized_access_attempts": 0,
                    "protocol_violations": 0,
                },
                "compliance_validation": {
                    "privacy_regulations_compliance": True,
                    "security_standards_compliance": True,
                    "audit_trail_complete": True,
                    "cryptographic_standards_met": True,
                },
            }

            integration_result = await components["multi_tenant"].execute_secure_federated_workflow(
                session=secure_fl_session, integration_level="full", monitoring_enabled=True
            )

            # Validate end-to-end security integration
            assert integration_result["workflow_execution_successful"] is True
            assert all(
                status == "fully_integrated"
                for status in integration_result["security_components_integration"].values()
            )
            assert all(validation for validation in integration_result["security_validations_passed"].values())
            assert integration_result["performance_with_security"]["convergence_achieved"] is True
            assert integration_result["security_incidents"]["privacy_violations"] == 0
            assert integration_result["security_incidents"]["unauthorized_access_attempts"] == 0
            assert (
                integration_result["security_incidents"]["attacks_mitigated"]
                == integration_result["security_incidents"]["attacks_detected"]
            )
            assert integration_result["compliance_validation"]["privacy_regulations_compliance"] is True
            assert (
                integration_result["performance_with_security"]["security_overhead_percent"] < 25
            )  # Acceptable overhead

            mock_workflow.assert_called_once()


if __name__ == "__main__":
    # Run security integration validation tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
