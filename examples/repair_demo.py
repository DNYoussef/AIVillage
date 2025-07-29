#!/usr/bin/env python3
"""HypeRAG Innovator Repair Agent Demo

Demonstrates the repair proposal system for GDC violations.
Shows template encoding, LLM integration, and repair operation generation.
"""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.hyperag.repair import InnovatorAgent, RepairOperation, TemplateEncoder
from mcp_servers.hyperag.repair.llm_driver import LLMDriver, ModelBackend, ModelConfig


async def demo_template_encoding():
    """Demonstrate violation template encoding"""
    print("🔧 Template Encoding Demo")
    print("=" * 50)

    # Sample violation data (similar to what GDC extractor would produce)
    violation_data = {
        "violation_id": "VIO_001",
        "rule_name": "allergy_prescription_conflict",
        "violated_pattern": "Patient with allergy prescribed contraindicated medication",
        "confidence_score": 0.95,
        "severity": "critical",
        "detected_at": datetime.now().isoformat(),
        "rule_description": "Patients must not be prescribed medications they are allergic to",
        "subgraph": {
            "nodes": [
                {
                    "id": "P123",
                    "labels": ["Patient"],
                    "properties": {
                        "name": "John Doe",
                        "age": 45,
                        "allergy": ["penicillin", "sulfa"],
                        "patient_id": "P123",
                    },
                },
                {
                    "id": "M456",
                    "labels": ["Medication"],
                    "properties": {
                        "name": "amoxicillin",
                        "drug_class": "penicillin",
                        "dosage_form": "tablet",
                    },
                },
            ],
            "edges": [
                {
                    "id": "PRESC_001",
                    "startNode": "P123",
                    "endNode": "M456",
                    "type": "PRESCRIBES",
                    "properties": {
                        "dosage": "500mg",
                        "frequency": "twice daily",
                        "prescribed_date": "2024-01-15",
                        "duration": "7 days",
                    },
                }
            ],
        },
    }

    # Create template encoder
    encoder = TemplateEncoder(
        domain_config={
            "field_mappings": {
                "Patient": {"allergy", "condition"},
                "Medication": {"dosage", "medication", "allergy"},
                "PRESCRIBES": {"dosage", "date", "medication"},
            }
        }
    )

    # Encode violation
    violation_template = encoder.encode_violation(violation_data)

    print("📝 Violation Template:")
    print(violation_template.to_description())

    # Create repair context
    context = encoder.create_repair_context(violation_template)
    print("\n🔍 Repair Context:")
    print(json.dumps(context, indent=2))

    # Extract conflicts
    conflicts = encoder.extract_critical_conflicts(violation_template)
    print(f"\n⚠️  Critical Conflicts Found: {len(conflicts)}")
    for conflict in conflicts:
        print(f"  - {conflict['description']}")


async def demo_llm_driver():
    """Demonstrate enhanced LLM driver functionality"""
    print("\n\n🤖 Enhanced LLM Driver Demo")
    print("=" * 50)

    # Create model config with rate limiting and audit features
    config = ModelConfig(
        model_name="llama3.2:3b",
        backend=ModelBackend.OLLAMA,
        temperature=0.1,
        max_tokens=1024,
        requests_per_minute=30,  # Rate limiting
        max_concurrent_requests=2,
        timeout_seconds=30,
    )

    print("📋 Configuration:")
    print(f"   Backend: {config.backend.value}")
    print(f"   Rate limit: {config.requests_per_minute} req/min")
    print(f"   Max concurrent: {config.max_concurrent_requests}")
    print(f"   Timeout: {config.timeout_seconds}s")

    driver = LLMDriver(config)

    # Check if model is available
    if await driver.is_ready():
        print("✅ LLM is ready")

        # Get model info
        model_info = await driver.get_model_info()
        print(f"📊 Model: {model_info.get('model_name', 'unknown')}")

        # Test simple generation
        test_prompt = """Analyze this medical scenario and suggest a repair operation:

Patient P123 is allergic to penicillin but has been prescribed amoxicillin (which contains penicillin).

Respond with a single JSON repair operation:"""

        system_prompt = "You are a medical knowledge graph repair assistant. Always respond with valid JSON."

        try:
            response = await driver.generate(
                prompt=test_prompt, system_prompt=system_prompt, max_tokens=256
            )

            print("\n🎯 Test Generation:")
            print(f"   Latency: {response.latency_ms:.1f}ms")
            print(f"   Tokens: {response.usage.get('total_tokens', 'unknown')}")
            print(f"   Response: {response.text[:200]}...")

            # Test confidence parsing
            confidence = driver.parse_confidence_from_response(response.text)
            if confidence:
                print(f"   Parsed Confidence: {confidence:.2f}")

            # Show audit log and usage stats
            audit_log = driver.get_audit_log()
            print(f"   Audit entries: {len(audit_log)}")

            usage_stats = driver.get_usage_stats()
            if "total_requests" in usage_stats:
                print(f"   Total requests: {usage_stats['total_requests']}")
                print(f"   Avg latency: {usage_stats['average_latency_ms']:.1f}ms")

        except Exception as e:
            print(f"❌ Generation failed: {e}")

        # Test LMStudio backend availability
        print("\n🏗️  Testing LMStudio Backend:")
        lmstudio_config = ModelConfig(
            model_name="any-model",
            backend=ModelBackend.LMSTUDIO,
            api_endpoint="http://localhost:1234",
        )
        lmstudio_driver = LLMDriver(lmstudio_config)

        if await lmstudio_driver.is_ready():
            print("✅ LMStudio is available")
        else:
            print(
                "❌ LMStudio not available (make sure LMStudio is running on port 1234)"
            )
    else:
        print("❌ Ollama not available (make sure Ollama is running with llama3.2:3b)")
        print("   To install: ollama pull llama3.2:3b")


async def demo_repair_proposals():
    """Demonstrate repair proposal generation"""
    print("\n\n🔧 Repair Proposal Demo")
    print("=" * 50)

    # Sample violation with medical safety issue
    violation_data = {
        "violation_id": "VIO_002",
        "rule_name": "dosage_validation_failure",
        "violated_pattern": "Prescription edge missing required dosage information",
        "subgraph": {
            "nodes": [
                {
                    "id": "P456",
                    "labels": ["Patient"],
                    "properties": {
                        "name": "Jane Smith",
                        "age": 67,
                        "weight": "65kg",
                        "condition": "hypertension",
                    },
                },
                {
                    "id": "M789",
                    "labels": ["Medication"],
                    "properties": {
                        "name": "lisinopril",
                        "drug_class": "ACE inhibitor",
                        "strength": "10mg",
                    },
                },
            ],
            "edges": [
                {
                    "id": "PRESC_002",
                    "startNode": "P456",
                    "endNode": "M789",
                    "type": "PRESCRIBES",
                    "properties": {
                        "prescribed_date": "2024-01-20",
                        "prescriber": "Dr. Johnson",
                        # Missing: dosage, frequency
                    },
                }
            ],
        },
    }

    try:
        # Create Innovator Agent
        agent = await InnovatorAgent.create_default(
            model_name="llama3.2:3b", domain="medical"
        )

        # Check if ready
        if await agent.llm_driver.is_ready():
            print("🏥 Medical Repair Agent Ready")

            # Generate repair proposals (now returns RepairProposalSet)
            proposal_set = await agent.generate_repair_proposals(
                violation_data=violation_data,
                max_operations=5,
                confidence_threshold=0.3,
            )

            print(f"\n📋 Repair Proposal Set {proposal_set.proposal_set_id}")
            print(f"   Violation: {proposal_set.violation_id}")
            print(f"   Rule: {proposal_set.gdc_rule}")
            print(f"   Proposals: {len(proposal_set.proposals)}")
            print(f"   Overall Confidence: {proposal_set.overall_confidence:.3f}")
            print(f"   Safety Score: {proposal_set.safety_score:.3f}")
            print(f"   Generation Time: {proposal_set.generation_time_ms:.1f}ms")
            print(f"   Valid: {proposal_set.is_valid}")

            if proposal_set.proposals:
                print("\n🔧 Proposed Operations:")
                for i, op in enumerate(proposal_set.proposals, 1):
                    print(f"   {i}. {op.operation_type.value}")
                    print(f"      Target: {op.target_id}")
                    print(f"      Confidence: {op.confidence:.3f}")
                    print(f"      Rationale: {op.rationale}")
                    if op.safety_critical:
                        print("      ⚠️  SAFETY CRITICAL")
                    print()

                print("📄 JSON Array Format:")
                print(proposal_set.to_json_array())

            # Show validation results (automatically performed)
            print("\n✅ Automatic Validation:")
            print(f"   Valid: {proposal_set.is_valid}")
            if proposal_set.validation_errors:
                print(f"   Errors: {len(proposal_set.validation_errors)}")
                for error in proposal_set.validation_errors:
                    print(f"     - {error}")
            if proposal_set.validation_warnings:
                print(f"   Warnings: {len(proposal_set.validation_warnings)}")
                for warning in proposal_set.validation_warnings:
                    print(f"     - {warning}")

            # Show high confidence proposals
            high_conf_ops = proposal_set.get_high_confidence_proposals(0.8)
            if high_conf_ops:
                print(f"\n🏆 High Confidence Operations (≥0.8): {len(high_conf_ops)}")
                for op in high_conf_ops:
                    print(
                        f"   - {op.operation_type.value} on {op.target_id} ({op.confidence:.2f})"
                    )

        else:
            print("❌ LLM not available for proposal generation")

    except Exception as e:
        print(f"❌ Repair proposal demo failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_operation_types():
    """Demonstrate different repair operation types"""
    print("\n\n⚙️  Operation Types Demo")
    print("=" * 50)

    # Create sample operations
    operations = [
        RepairOperation(
            operation_type=RepairOperationType.DELETE_EDGE,
            target_id="PRESC_BAD_001",
            rationale="Remove unsafe prescription - patient allergic to penicillin",
            confidence=0.95,
            safety_critical=True,
            estimated_impact="high",
        ),
        RepairOperation(
            operation_type=RepairOperationType.ADD_EDGE,
            target_id="PRESC_NEW_001",
            source_id="P123",
            relationship_type="PRESCRIBES",
            rationale="Add safe alternative medication prescription",
            confidence=0.8,
            properties={"dosage": "250mg", "frequency": "twice daily"},
        ),
        RepairOperation(
            operation_type=RepairOperationType.UPDATE_ATTR,
            target_id="PRESC_002",
            property_name="dosage",
            property_value="500mg twice daily",
            rationale="Add missing dosage information for complete prescription",
            confidence=0.85,
        ),
        RepairOperation(
            operation_type=RepairOperationType.MERGE_NODES,
            target_id="MED_001",
            merge_target_id="MED_002",
            rationale="Merge duplicate medication entries with same active ingredient",
            confidence=0.7,
        ),
    ]

    print("🔧 Operation Examples:")
    for i, op in enumerate(operations, 1):
        print(f"\n{i}. {op.operation_type.value.upper()}")
        print(f"   Target: {op.target_id}")
        print(f"   Confidence: {op.confidence:.2f}")
        print(f"   Impact: {op.estimated_impact}")
        print(f"   Safety Critical: {'Yes' if op.safety_critical else 'No'}")
        print(f"   Rationale: {op.rationale}")

        # Show operation-specific details
        if op.source_id:
            print(f"   Source: {op.source_id}")
        if op.relationship_type:
            print(f"   Relationship: {op.relationship_type}")
        if op.property_name:
            print(f"   Property: {op.property_name} = {op.property_value}")
        if op.merge_target_id:
            print(f"   Merge Target: {op.merge_target_id}")

        print(f"   JSON: {op.to_jsonl()}")


async def demo_domain_specialization():
    """Demonstrate domain-specific repair logic"""
    print("\n\n🏥 Domain Specialization Demo")
    print("=" * 50)

    # Medical domain encoder
    medical_encoder = TemplateEncoder(
        domain_config={
            "field_mappings": {
                "Patient": {"allergy", "condition", "age"},
                "Medication": {"dosage", "drug_class", "contraindications"},
                "PRESCRIBES": {"dosage", "date", "frequency"},
                "ALLERGIC_TO": {"severity", "allergy"},
            },
            "repair_guidelines": {
                "safety_first": True,
                "require_dosage_validation": True,
                "preserve_allergy_data": True,
            },
        }
    )

    print("🏥 Medical Domain Configuration:")
    print("   Critical fields: allergy, dosage, contraindications")
    print("   Safety guidelines: preserve_allergy_data, require_dosage_validation")

    # General domain encoder
    general_encoder = TemplateEncoder()

    print("\n🌐 General Domain Configuration:")
    print("   Critical fields: confidence, timestamp, source")
    print("   Safety guidelines: minimal intervention")

    # Sample node for comparison
    medical_node = {
        "id": "P123",
        "labels": ["Patient"],
        "properties": {
            "name": "John Doe",
            "allergy": ["penicillin"],
            "condition": "pneumonia",
            "age": 45,
            "insurance": "Blue Cross",
        },
    }

    print("\n📊 Field Analysis Comparison:")

    # Medical encoding
    medical_template = medical_encoder.encode_node(medical_node)
    print(
        f"   Medical Domain Critical Fields: {[f.value for f in medical_template.critical_fields]}"
    )

    # General encoding
    general_template = general_encoder.encode_node(medical_node)
    print(
        f"   General Domain Critical Fields: {[f.value for f in general_template.critical_fields]}"
    )

    print(
        f"\n   Medical template focuses on: {', '.join(f.value for f in medical_template.critical_fields)}"
    )
    print(
        f"   General template focuses on: {', '.join(f.value for f in general_template.critical_fields)}"
    )


async def main():
    """Main demo function"""
    print("🚀 HypeRAG Innovator Repair Agent Demo")
    print("=" * 60)

    try:
        await demo_template_encoding()
        await demo_llm_driver()
        await demo_repair_proposals()
        await demo_operation_types()
        await demo_domain_specialization()

        print("\n\n✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Enhanced template encoding with natural language sentences")
        print("  • LLM driver with rate limiting and audit logging")
        print("  • LMStudio backend support for local 7B-14B models")
        print("  • RepairProposalSet with automatic validation")
        print("  • JSON array output format with few-shot examples")
        print("  • Enhanced confidence parsing and extraction")
        print("  • Safety-critical operation identification")
        print("  • Domain-specific repair logic (medical vs general)")
        print("  • No auto-apply policy - proposals only")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(main())
