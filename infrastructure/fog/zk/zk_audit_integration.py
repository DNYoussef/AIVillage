"""
ZK Predicate Audit System Integration

Integrates zero-knowledge predicates with the existing fog computing audit system:
- Privacy-preserving audit trails
- ZK proof integration with compliance system
- Automated predicate verification workflows
- Gradual expansion framework for new predicates

This module bridges the gap between ZK predicates and existing audit infrastructure
while maintaining privacy guarantees and system integrity.
"""

from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Any

from .zk_predicates import PredicateContext, PredicateType, ProofResult, ZKPredicateEngine

logger = logging.getLogger(__name__)


class ZKAuditIntegration:
    """
    Integration layer between ZK predicates and audit systems.

    Provides:
    - Privacy-preserving audit events
    - Automated compliance verification
    - ZK proof lifecycle management
    - Integration with existing proof generation system
    """

    def __init__(
        self, zk_engine: ZKPredicateEngine, compliance_system: Any | None = None, proof_generator: Any | None = None
    ):
        self.zk_engine = zk_engine
        self.compliance_system = compliance_system
        self.proof_generator = proof_generator

        # Audit event storage
        self.audit_events: list[dict[str, Any]] = []

        # Privacy-preserving audit configuration
        self.audit_config = {
            "privacy_level": "high",  # "minimal", "standard", "high"
            "retention_hours": 168,  # 1 week
            "batch_size": 50,
            "export_format": "privacy_preserving",
        }

        logger.info("ZK Audit Integration initialized")

    async def record_zk_audit_event(
        self,
        event_type: str,
        predicate_type: PredicateType,
        entity_id: str,
        proof_result: ProofResult | None = None,
        privacy_level: str = "high",
        additional_context: dict[str, Any] | None = None,
    ):
        """
        Record privacy-preserving audit event for ZK predicate operations.

        Only records privacy-safe metadata, never sensitive data.
        """
        try:
            # Create privacy-preserving audit event
            audit_event = {
                "event_id": f"zk_audit_{int(datetime.now(timezone.utc).timestamp())}",
                "event_type": event_type,
                "predicate_type": predicate_type.value,
                "entity_id_hash": self._hash_entity_id(entity_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "privacy_level": privacy_level,
                "node_id": self.zk_engine.node_id,
            }

            # Add proof result if available (privacy-safe)
            if proof_result:
                audit_event["verification_result"] = proof_result.value
                audit_event["proof_valid"] = proof_result == ProofResult.VALID

            # Add privacy-safe context
            if additional_context:
                safe_context = self._sanitize_context(additional_context, privacy_level)
                audit_event["context"] = safe_context

            # Store audit event
            self.audit_events.append(audit_event)

            # Integrate with existing audit system if available
            if self.compliance_system:
                await self._send_to_compliance_system(audit_event)

            logger.debug(f"Recorded ZK audit event: {event_type} for {predicate_type.value}")

        except Exception as e:
            logger.error(f"Failed to record ZK audit event: {e}")

    def _hash_entity_id(self, entity_id: str) -> str:
        """Create privacy-preserving hash of entity ID."""
        import hashlib

        return hashlib.sha256(f"zk_entity_{entity_id}".encode()).hexdigest()[:16]

    def _sanitize_context(self, context: dict[str, Any], privacy_level: str) -> dict[str, Any]:
        """Sanitize context data based on privacy level."""
        if privacy_level == "minimal":
            # Only basic metadata
            return {"has_context": bool(context), "context_keys_count": len(context)}
        elif privacy_level == "standard":
            # Some aggregated information
            safe_context = {
                "context_size": len(str(context)),
                "has_sensitive_fields": any(key in ["password", "secret", "private", "key"] for key in context.keys()),
            }
            # Add safe numeric aggregates
            numeric_fields = [k for k, v in context.items() if isinstance(v, int | float)]
            if numeric_fields:
                safe_context["numeric_field_count"] = len(numeric_fields)
            return safe_context
        else:  # high privacy
            # Only existence confirmation
            return {"context_provided": bool(context)}

    async def _send_to_compliance_system(self, audit_event: dict[str, Any]):
        """Send privacy-preserving audit event to compliance system."""
        try:
            if hasattr(self.compliance_system, "_log_audit_event"):
                await self.compliance_system._log_audit_event(
                    event_type=audit_event["event_type"],
                    entity_type="zk_predicate",
                    entity_id=audit_event["entity_id_hash"],
                    framework="zk_privacy",
                    action_details={
                        "predicate_type": audit_event["predicate_type"],
                        "verification_result": audit_event.get("verification_result"),
                        "privacy_level": audit_event["privacy_level"],
                    },
                    risk_level="low",
                )
        except Exception as e:
            logger.error(f"Failed to send audit event to compliance system: {e}")

    async def verify_network_policy_compliance(
        self, network_config: dict[str, Any], policy_parameters: dict[str, Any], entity_id: str
    ) -> tuple[bool, str | None]:
        """
        Verify network policy compliance using ZK predicates.

        Returns:
            (is_compliant, proof_id or None)
        """
        try:
            # Generate commitment for network config
            context = PredicateContext(network_policies=policy_parameters)
            commitment_id = await self.zk_engine.generate_commitment(
                predicate_id="network_policy", secret_data=network_config, context=context
            )

            # Record commitment creation
            await self.record_zk_audit_event(
                event_type="zk_commitment_created",
                predicate_type=PredicateType.NETWORK_POLICY,
                entity_id=entity_id,
                additional_context={"commitment_id": commitment_id[:16]},
            )

            # Generate proof
            proof_id = await self.zk_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="network_policy",
                secret_data=network_config,
                public_parameters=policy_parameters,
            )

            # Verify proof
            verification_result = await self.zk_engine.verify_proof(
                proof_id=proof_id, public_parameters=policy_parameters, context=context
            )

            # Record verification
            await self.record_zk_audit_event(
                event_type="zk_proof_verified",
                predicate_type=PredicateType.NETWORK_POLICY,
                entity_id=entity_id,
                proof_result=verification_result,
                additional_context={"proof_id": proof_id},
            )

            is_compliant = verification_result == ProofResult.VALID

            logger.info(f"Network policy compliance check: {'PASS' if is_compliant else 'FAIL'}")
            return is_compliant, proof_id if is_compliant else None

        except Exception as e:
            logger.error(f"Network policy compliance check failed: {e}")
            await self.record_zk_audit_event(
                event_type="zk_verification_error",
                predicate_type=PredicateType.NETWORK_POLICY,
                entity_id=entity_id,
                additional_context={"error": str(e)[:100]},
            )
            return False, None

    async def verify_content_compliance(
        self, file_metadata: dict[str, Any], content_policy: dict[str, Any], entity_id: str
    ) -> tuple[bool, str | None]:
        """
        Verify content (MIME type) compliance using ZK predicates.
        """
        try:
            # Generate commitment for file metadata
            context = PredicateContext(allowed_mime_types=set(content_policy.get("allowed_types", [])))
            commitment_id = await self.zk_engine.generate_commitment(
                predicate_id="mime_type", secret_data=file_metadata, context=context
            )

            # Record commitment
            await self.record_zk_audit_event(
                event_type="zk_commitment_created", predicate_type=PredicateType.MIME_TYPE, entity_id=entity_id
            )

            # Generate and verify proof
            proof_id = await self.zk_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="mime_type",
                secret_data=file_metadata,
                public_parameters=content_policy,
            )

            verification_result = await self.zk_engine.verify_proof(
                proof_id=proof_id, public_parameters=content_policy, context=context
            )

            # Record verification
            await self.record_zk_audit_event(
                event_type="zk_proof_verified",
                predicate_type=PredicateType.MIME_TYPE,
                entity_id=entity_id,
                proof_result=verification_result,
            )

            is_compliant = verification_result == ProofResult.VALID
            return is_compliant, proof_id if is_compliant else None

        except Exception as e:
            logger.error(f"Content compliance check failed: {e}")
            return False, None

    async def verify_model_integrity(
        self, model_metadata: dict[str, Any], trusted_models: dict[str, Any], entity_id: str
    ) -> tuple[bool, str | None]:
        """
        Verify ML model integrity using ZK predicates.
        """
        try:
            # Generate commitment for model metadata
            context = PredicateContext(trusted_model_hashes=set(trusted_models.get("trusted_hashes", [])))
            commitment_id = await self.zk_engine.generate_commitment(
                predicate_id="model_hash", secret_data=model_metadata, context=context
            )

            # Record commitment
            await self.record_zk_audit_event(
                event_type="zk_commitment_created", predicate_type=PredicateType.MODEL_HASH, entity_id=entity_id
            )

            # Generate and verify proof
            proof_id = await self.zk_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="model_hash",
                secret_data=model_metadata,
                public_parameters=trusted_models,
            )

            verification_result = await self.zk_engine.verify_proof(
                proof_id=proof_id, public_parameters=trusted_models, context=context
            )

            # Record verification
            await self.record_zk_audit_event(
                event_type="zk_proof_verified",
                predicate_type=PredicateType.MODEL_HASH,
                entity_id=entity_id,
                proof_result=verification_result,
            )

            is_compliant = verification_result == ProofResult.VALID
            return is_compliant, proof_id if is_compliant else None

        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            return False, None

    async def verify_privacy_compliance(
        self, compliance_data: dict[str, Any], compliance_requirements: dict[str, Any], entity_id: str
    ) -> tuple[bool, str | None]:
        """
        Verify privacy compliance using ZK predicates.
        """
        try:
            # Generate commitment for compliance data
            context = PredicateContext(compliance_rules=compliance_requirements)
            commitment_id = await self.zk_engine.generate_commitment(
                predicate_id="compliance_check", secret_data=compliance_data, context=context
            )

            # Record commitment
            await self.record_zk_audit_event(
                event_type="zk_commitment_created", predicate_type=PredicateType.COMPLIANCE_CHECK, entity_id=entity_id
            )

            # Generate and verify proof
            proof_id = await self.zk_engine.generate_proof(
                commitment_id=commitment_id,
                predicate_id="compliance_check",
                secret_data=compliance_data,
                public_parameters=compliance_requirements,
            )

            verification_result = await self.zk_engine.verify_proof(
                proof_id=proof_id, public_parameters=compliance_requirements, context=context
            )

            # Record verification
            await self.record_zk_audit_event(
                event_type="zk_proof_verified",
                predicate_type=PredicateType.COMPLIANCE_CHECK,
                entity_id=entity_id,
                proof_result=verification_result,
            )

            is_compliant = verification_result == ProofResult.VALID
            return is_compliant, proof_id if is_compliant else None

        except Exception as e:
            logger.error(f"Privacy compliance check failed: {e}")
            return False, None

    async def batch_verify_compliance(
        self, verification_requests: list[dict[str, Any]]
    ) -> dict[str, tuple[bool, str | None]]:
        """
        Batch verify multiple compliance requirements using ZK predicates.

        Args:
            verification_requests: List of verification requests with format:
                {
                    "type": "network_policy|mime_type|model_hash|compliance_check",
                    "secret_data": dict,
                    "public_parameters": dict,
                    "entity_id": str
                }

        Returns:
            Dict mapping entity_id to (is_compliant, proof_id)
        """
        results = {}

        for request in verification_requests:
            try:
                verification_type = request["type"]
                secret_data = request["secret_data"]
                public_parameters = request["public_parameters"]
                entity_id = request["entity_id"]

                if verification_type == "network_policy":
                    result = await self.verify_network_policy_compliance(secret_data, public_parameters, entity_id)
                elif verification_type == "mime_type":
                    result = await self.verify_content_compliance(secret_data, public_parameters, entity_id)
                elif verification_type == "model_hash":
                    result = await self.verify_model_integrity(secret_data, public_parameters, entity_id)
                elif verification_type == "compliance_check":
                    result = await self.verify_privacy_compliance(secret_data, public_parameters, entity_id)
                else:
                    logger.warning(f"Unknown verification type: {verification_type}")
                    result = (False, None)

                results[entity_id] = result

            except Exception as e:
                logger.error(f"Batch verification failed for entity {request.get('entity_id', 'unknown')}: {e}")
                results[request.get("entity_id", "unknown")] = (False, None)

        logger.info(f"Batch verified {len(verification_requests)} requests")
        return results

    async def generate_compliance_report(
        self, start_time: datetime, end_time: datetime, predicate_types: list[PredicateType] | None = None
    ) -> dict[str, Any]:
        """
        Generate privacy-preserving compliance report for ZK predicate usage.
        """
        try:
            # Filter audit events by time range and predicate types
            filtered_events = []
            for event in self.audit_events:
                event_time = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                if start_time <= event_time <= end_time:
                    if predicate_types is None or PredicateType(event["predicate_type"]) in predicate_types:
                        filtered_events.append(event)

            # Generate aggregated statistics (privacy-preserving)
            stats = {
                "reporting_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600,
                },
                "total_zk_operations": len(filtered_events),
                "operations_by_type": {},
                "verification_results": {},
                "privacy_levels": {},
                "node_id": self.zk_engine.node_id,
            }

            for event in filtered_events:
                # Count by predicate type
                predicate_type = event["predicate_type"]
                stats["operations_by_type"][predicate_type] = stats["operations_by_type"].get(predicate_type, 0) + 1

                # Count by verification result
                if "verification_result" in event:
                    result = event["verification_result"]
                    stats["verification_results"][result] = stats["verification_results"].get(result, 0) + 1

                # Count by privacy level
                privacy_level = event.get("privacy_level", "unknown")
                stats["privacy_levels"][privacy_level] = stats["privacy_levels"].get(privacy_level, 0) + 1

            # Add ZK engine statistics
            zk_stats = await self.zk_engine.get_proof_stats()
            stats["zk_engine_stats"] = {
                "total_commitments": zk_stats["total_commitments"],
                "total_proofs": zk_stats["total_proofs"],
                "verification_rate": zk_stats["verification_rate"],
                "validity_rate": zk_stats["validity_rate"],
            }

            logger.info(f"Generated ZK compliance report with {len(filtered_events)} events")
            return stats

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise

    async def cleanup_audit_data(self, retention_hours: int | None = None) -> int:
        """Clean up old audit events based on retention policy."""
        if retention_hours is None:
            retention_hours = self.audit_config["retention_hours"]

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)

        # Filter out old events
        original_count = len(self.audit_events)
        self.audit_events = [
            event
            for event in self.audit_events
            if datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")) > cutoff_time
        ]

        cleaned_count = original_count - len(self.audit_events)

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old ZK audit events")

        return cleaned_count

    async def export_privacy_preserving_audit_log(self, output_path: str) -> int:
        """Export privacy-preserving audit log."""
        try:
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "node_id": self.zk_engine.node_id,
                "privacy_level": self.audit_config["privacy_level"],
                "event_count": len(self.audit_events),
                "events": self.audit_events,
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, sort_keys=True)

            logger.info(f"Exported {len(self.audit_events)} ZK audit events to {output_path}")
            return len(self.audit_events)

        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
            raise


class ZKPredicateWorkflow:
    """
    Workflow orchestrator for ZK predicate operations.

    Provides higher-level workflows that combine multiple ZK predicates
    for common fog computing scenarios.
    """

    def __init__(self, audit_integration: ZKAuditIntegration):
        self.audit_integration = audit_integration
        self.workflows: dict[str, dict[str, Any]] = {}

        # Initialize common workflows
        self._initialize_workflows()

        logger.info("ZK Predicate Workflow orchestrator initialized")

    def _initialize_workflows(self):
        """Initialize common ZK predicate workflows."""

        # Comprehensive fog node onboarding
        self.workflows["fog_node_onboarding"] = {
            "name": "Fog Node Onboarding Verification",
            "description": "Complete privacy-preserving verification for new fog nodes",
            "steps": [
                {"type": "network_policy", "required": True},
                {"type": "compliance_check", "required": True},
                {"type": "model_hash", "required": False},
            ],
        }

        # Content processing verification
        self.workflows["content_processing"] = {
            "name": "Content Processing Verification",
            "description": "Verify content compliance before processing",
            "steps": [{"type": "mime_type", "required": True}, {"type": "compliance_check", "required": True}],
        }

        # Model deployment verification
        self.workflows["model_deployment"] = {
            "name": "ML Model Deployment Verification",
            "description": "Comprehensive model integrity and compliance verification",
            "steps": [
                {"type": "model_hash", "required": True},
                {"type": "compliance_check", "required": True},
                {"type": "network_policy", "required": False},
            ],
        }

    async def execute_workflow(
        self, workflow_name: str, entity_id: str, verification_data: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Execute a predefined ZK predicate workflow.

        Args:
            workflow_name: Name of workflow to execute
            entity_id: Entity being verified
            verification_data: Data for each verification step:
                {
                    "network_policy": {"secret_data": {...}, "public_parameters": {...}},
                    "mime_type": {"secret_data": {...}, "public_parameters": {...}},
                    ...
                }

        Returns:
            Workflow execution results
        """
        try:
            if workflow_name not in self.workflows:
                raise ValueError(f"Unknown workflow: {workflow_name}")

            workflow = self.workflows[workflow_name]
            results = {
                "workflow_name": workflow_name,
                "entity_id": entity_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "steps": [],
                "overall_success": True,
                "proof_ids": [],
            }

            logger.info(f"Executing workflow '{workflow_name}' for entity {entity_id}")

            # Execute each step
            for step in workflow["steps"]:
                step_type = step["type"]
                is_required = step["required"]

                if step_type not in verification_data:
                    if is_required:
                        results["overall_success"] = False
                        results["steps"].append(
                            {"type": step_type, "success": False, "error": f"Required step data missing: {step_type}"}
                        )
                        continue
                    else:
                        results["steps"].append(
                            {
                                "type": step_type,
                                "success": True,
                                "skipped": True,
                                "reason": "Optional step data not provided",
                            }
                        )
                        continue

                # Execute verification step
                step_data = verification_data[step_type]
                step_success, proof_id = await self._execute_verification_step(step_type, entity_id, step_data)

                results["steps"].append(
                    {"type": step_type, "success": step_success, "proof_id": proof_id, "required": is_required}
                )

                if proof_id:
                    results["proof_ids"].append(proof_id)

                if not step_success and is_required:
                    results["overall_success"] = False

            results["completed_at"] = datetime.now(timezone.utc).isoformat()

            # Record workflow completion
            await self.audit_integration.record_zk_audit_event(
                event_type="zk_workflow_completed",
                predicate_type=PredicateType.COMPLIANCE_CHECK,  # Generic type for workflows
                entity_id=entity_id,
                additional_context={
                    "workflow_name": workflow_name,
                    "overall_success": results["overall_success"],
                    "steps_count": len(results["steps"]),
                    "proof_count": len(results["proof_ids"]),
                },
            )

            logger.info(
                f"Workflow '{workflow_name}' completed: {'SUCCESS' if results['overall_success'] else 'FAILED'}"
            )
            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    async def _execute_verification_step(
        self, step_type: str, entity_id: str, step_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Execute a single verification step."""
        try:
            secret_data = step_data["secret_data"]
            public_parameters = step_data["public_parameters"]

            if step_type == "network_policy":
                return await self.audit_integration.verify_network_policy_compliance(
                    secret_data, public_parameters, entity_id
                )
            elif step_type == "mime_type":
                return await self.audit_integration.verify_content_compliance(secret_data, public_parameters, entity_id)
            elif step_type == "model_hash":
                return await self.audit_integration.verify_model_integrity(secret_data, public_parameters, entity_id)
            elif step_type == "compliance_check":
                return await self.audit_integration.verify_privacy_compliance(secret_data, public_parameters, entity_id)
            else:
                logger.error(f"Unknown verification step type: {step_type}")
                return False, None

        except Exception as e:
            logger.error(f"Verification step failed for {step_type}: {e}")
            return False, None

    def register_workflow(self, workflow_name: str, workflow_definition: dict[str, Any]):
        """Register a custom ZK predicate workflow."""
        required_fields = ["name", "description", "steps"]
        if not all(field in workflow_definition for field in required_fields):
            raise ValueError(f"Workflow definition must include: {required_fields}")

        self.workflows[workflow_name] = workflow_definition
        logger.info(f"Registered custom ZK workflow: {workflow_name}")

    def get_available_workflows(self) -> list[str]:
        """Get list of available workflows."""
        return list(self.workflows.keys())

    def get_workflow_definition(self, workflow_name: str) -> dict[str, Any] | None:
        """Get definition of a specific workflow."""
        return self.workflows.get(workflow_name)
