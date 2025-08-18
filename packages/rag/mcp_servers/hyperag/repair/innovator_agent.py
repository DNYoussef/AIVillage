"""HypeRAG Innovator Repair Agent.

Given a GDC Violation subgraph, produces structured repair proposal lists
without auto-applying changes. Supports pluggable local LLM models.
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..guardian.gate import GuardianGate
from .llm_driver import LLMDriver, ModelConfig
from .templates import TemplateEncoder, ViolationTemplate


class RepairOperationType(Enum):
    """Types of repair operations."""

    ADD_EDGE = "add_edge"
    DELETE_EDGE = "delete_edge"
    UPDATE_ATTR = "update_attr"
    MERGE_NODES = "merge_nodes"
    ADD_NODE = "add_node"
    DELETE_NODE = "delete_node"


@dataclass
class RepairOperation:
    """Single repair operation with rationale."""

    operation_type: RepairOperationType
    target_id: str
    rationale: str
    confidence: float

    # Operation-specific parameters
    source_id: str | None = None
    relationship_type: str | None = None
    property_name: str | None = None
    property_value: Any | None = None
    merge_target_id: str | None = None
    node_type: str | None = None
    properties: dict[str, Any] | None = None

    # Metadata
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    safety_critical: bool = False
    estimated_impact: str = "low"

    def to_dict(self) -> dict[str, Any]:
        """Convert operation to dictionary format."""
        result = {
            "op": self.operation_type.value,
            "target_id": self.target_id,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "operation_id": self.operation_id,
            "safety_critical": self.safety_critical,
            "estimated_impact": self.estimated_impact,
        }

        # Add operation-specific fields
        if self.source_id:
            result["source_id"] = self.source_id
        if self.relationship_type:
            result["relationship_type"] = self.relationship_type
        if self.property_name:
            result["property_name"] = self.property_name
        if self.property_value is not None:
            result["property_value"] = self.property_value
        if self.merge_target_id:
            result["merge_target_id"] = self.merge_target_id
        if self.node_type:
            result["node_type"] = self.node_type
        if self.properties:
            result["properties"] = self.properties

        return result

    def to_jsonl(self) -> str:
        """Convert operation to JSONL format."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairOperation":
        """Create operation from dictionary."""
        op_type = RepairOperationType(data["op"])

        return cls(
            operation_type=op_type,
            target_id=data["target_id"],
            rationale=data["rationale"],
            confidence=data["confidence"],
            source_id=data.get("source_id"),
            relationship_type=data.get("relationship_type"),
            property_name=data.get("property_name"),
            property_value=data.get("property_value"),
            merge_target_id=data.get("merge_target_id"),
            node_type=data.get("node_type"),
            properties=data.get("properties"),
            operation_id=data.get("operation_id", str(uuid.uuid4())),
            safety_critical=data.get("safety_critical", False),
            estimated_impact=data.get("estimated_impact", "low"),
        )


@dataclass
class RepairProposalSet:
    """Set of repair proposals with validation and metadata."""

    proposals: list[RepairOperation]
    violation_id: str
    gdc_rule: str

    # Generation metadata
    proposal_set_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    generation_time_ms: float = 0.0

    # Validation results
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    # Quality metrics
    overall_confidence: float = 0.0
    safety_score: float = 0.0
    completeness_score: float = 0.0

    # Analysis
    repair_summary: str = ""
    potential_risks: list[str] = field(default_factory=list)
    validation_notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.proposals:
            self.overall_confidence = sum(op.confidence for op in self.proposals) / len(self.proposals)
            self.safety_score = 1.0 - (sum(1 for op in self.proposals if op.safety_critical) / len(self.proposals))
            self.completeness_score = min(1.0, len(self.proposals) / 3.0)  # Assume ~3 ops for complete repair

    def to_dict(self) -> dict[str, Any]:
        """Convert proposal set to dictionary."""
        return {
            "proposal_set_id": self.proposal_set_id,
            "violation_id": self.violation_id,
            "gdc_rule": self.gdc_rule,
            "created_at": self.created_at.isoformat(),
            "model_used": self.model_used,
            "generation_time_ms": self.generation_time_ms,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "overall_confidence": self.overall_confidence,
            "safety_score": self.safety_score,
            "completeness_score": self.completeness_score,
            "repair_summary": self.repair_summary,
            "potential_risks": self.potential_risks,
            "validation_notes": self.validation_notes,
            "proposals": [op.to_dict() for op in self.proposals],
        }

    def to_json_array(self) -> str:
        """Convert proposals to JSON array format."""
        return json.dumps([op.to_dict() for op in self.proposals], indent=2)

    def get_high_confidence_proposals(self, threshold: float = 0.8) -> list[RepairOperation]:
        """Get proposals with confidence above threshold."""
        return [op for op in self.proposals if op.confidence >= threshold]

    def get_safety_critical_proposals(self) -> list[RepairOperation]:
        """Get safety-critical proposals that need extra review."""
        return [op for op in self.proposals if op.safety_critical]

    def validate(self) -> bool:
        """Validate the proposal set and update validation status."""
        self.validation_errors.clear()
        self.validation_warnings.clear()

        # Check for empty proposals
        if not self.proposals:
            self.validation_warnings.append("No repair proposals generated")

        # Validate individual proposals
        for i, proposal in enumerate(self.proposals):
            if not proposal.target_id:
                self.validation_errors.append(f"Proposal {i + 1} missing target_id")

            if not proposal.rationale:
                self.validation_warnings.append(f"Proposal {i + 1} missing rationale")

            if proposal.confidence < 0.0 or proposal.confidence > 1.0:
                self.validation_errors.append(f"Proposal {i + 1} has invalid confidence: {proposal.confidence}")

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid


@dataclass
class RepairProposal:
    """Complete repair proposal for a GDC violation (legacy compatibility)."""

    violation_id: str
    gdc_rule: str
    operations: list[RepairOperation]

    # Metadata
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    generation_time_ms: float = 0.0

    # Quality metrics
    overall_confidence: float = 0.0
    safety_score: float = 0.0
    completeness_score: float = 0.0

    # Analysis
    repair_summary: str = ""
    potential_risks: list[str] = field(default_factory=list)
    validation_notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.operations:
            self.overall_confidence = sum(op.confidence for op in self.operations) / len(self.operations)
            self.safety_score = 1.0 - (sum(1 for op in self.operations if op.safety_critical) / len(self.operations))
            self.completeness_score = min(1.0, len(self.operations) / 3.0)  # Assume ~3 ops for complete repair

    def to_dict(self) -> dict[str, Any]:
        """Convert proposal to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "violation_id": self.violation_id,
            "gdc_rule": self.gdc_rule,
            "created_at": self.created_at.isoformat(),
            "model_used": self.model_used,
            "generation_time_ms": self.generation_time_ms,
            "overall_confidence": self.overall_confidence,
            "safety_score": self.safety_score,
            "completeness_score": self.completeness_score,
            "repair_summary": self.repair_summary,
            "potential_risks": self.potential_risks,
            "validation_notes": self.validation_notes,
            "operations": [op.to_dict() for op in self.operations],
        }

    def to_jsonl(self) -> str:
        """Convert proposal to JSONL format."""
        lines = []
        for operation in self.operations:
            lines.append(operation.to_jsonl())
        return "\n".join(lines)

    def get_high_confidence_operations(self, threshold: float = 0.8) -> list[RepairOperation]:
        """Get operations with confidence above threshold."""
        return [op for op in self.operations if op.confidence >= threshold]

    def get_safety_critical_operations(self) -> list[RepairOperation]:
        """Get safety-critical operations that need extra review."""
        return [op for op in self.operations if op.safety_critical]


class PromptComposer:
    """Composes prompts for repair operations."""

    def __init__(self, prompt_bank_path: str | None = None, domain: str = "general") -> None:
        """Initialize prompt composer.

        Args:
            prompt_bank_path: Path to prompt bank markdown file
            domain: Domain specialization (general, medical, etc.)
        """
        self.domain = domain
        self.prompt_bank_path = prompt_bank_path or (Path(__file__).parent / "prompt_bank.md")
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict[str, str]:
        """Load prompts from markdown file."""
        prompts = {}

        try:
            if Path(self.prompt_bank_path).exists():
                with open(self.prompt_bank_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse markdown sections
                sections = re.split(r"^#+\s+(.+)$", content, flags=re.MULTILINE)

                for i in range(1, len(sections), 2):
                    section_name = sections[i].strip().lower().replace(" ", "_")
                    section_content = sections[i + 1].strip() if i + 1 < len(sections) else ""
                    prompts[section_name] = section_content
        except Exception as e:
            logging.warning(f"Could not load prompts from {self.prompt_bank_path}: {e}")

        # Fallback prompts
        if not prompts:
            prompts = self._get_default_prompts()

        return prompts

    def _get_default_prompts(self) -> dict[str, str]:
        """Get default prompts if file loading fails."""
        return {
            "base_system_prompt": """You are a knowledge graph repair assistant (Innovator Agent) specializing in analyzing graph violations and proposing precise repair operations.

Your responsibilities:
- Analyze knowledge graph constraint violations
- Propose minimal, targeted repair operations
- Preserve data integrity and core relationships
- Provide clear rationales for each operation

CRITICAL RULES:
- NEVER delete core identity edges unless explicitly instructed
- Always provide a rationale for each operation
- Prefer minimal changes over extensive restructuring
- Maintain semantic consistency
- Include confidence scores when possible""",
            "general_repair_instructions": """Analyze the following knowledge graph violation and propose repair operations.

**Available Operations:**
- add_edge: Create new relationships
- delete_edge: Remove problematic relationships
- update_attr: Modify node/edge properties
- merge_nodes: Combine duplicate entities

**Output Format:**
Provide JSONL format with one operation per line:
{"op":"operation_type","target_id":"element_id","rationale":"explanation","confidence":0.9}

**Requirements:**
1. Each operation must include a clear rationale
2. Preserve core identity relationships
3. Minimize changes while resolving the violation
4. Include confidence score (0.0-1.0) if possible""",
        }

    def get_system_prompt(self) -> str:
        """Get system prompt for the domain."""
        if self.domain == "medical":
            return self.prompts.get(
                "medical_domain_system_prompt",
                self.prompts.get("base_system_prompt", ""),
            )
        return self.prompts.get("base_system_prompt", "")

    def get_repair_instructions(self) -> str:
        """Get repair instructions for the domain."""
        if self.domain == "medical":
            return self.prompts.get(
                "medical_domain_instructions",
                self.prompts.get("general_repair_instructions", ""),
            )
        return self.prompts.get("general_repair_instructions", "")

    def compose_repair_prompt(self, violation: ViolationTemplate, context: dict[str, Any]) -> str:
        """Compose complete repair prompt.

        Args:
            violation: The violation template
            context: Additional context for repair

        Returns:
            Complete prompt for LLM
        """
        # Get base instructions
        instructions = self.get_repair_instructions()

        # Add violation description
        violation_description = violation.to_description()

        # Add context information
        context_sections = []

        if "entity_analysis" in context:
            context_sections.append("**Entity Analysis:**")
            for entity_type, entities in context["entity_analysis"].items():
                context_sections.append(f"- {entity_type}: {len(entities)} instances")

        if "relationship_analysis" in context:
            context_sections.append("**Relationship Analysis:**")
            for rel_type, relationships in context["relationship_analysis"].items():
                context_sections.append(f"- {rel_type}: {len(relationships)} instances")

        # Domain-specific guidance
        domain_guidance = ""
        if self.domain == "medical":
            domain_guidance = """
**Medical Safety Guidelines:**
- Patient safety is paramount
- Drug-allergy conflicts must be resolved immediately
- Dosage information must be precise and validated
- Never remove safety-critical edges without explicit justification"""

        # Combine all parts
        prompt_parts = [
            instructions,
            "",
            "**Violation Details:**",
            violation_description,
            "",
        ]

        if context_sections:
            prompt_parts.extend(context_sections)
            prompt_parts.append("")

        if domain_guidance:
            prompt_parts.append(domain_guidance)
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "**Your Task:**",
                "Propose specific repair operations as a JSON array. Output only the JSON array, no additional text.",
                "",
                "**Required Format:**",
                "[",
                '  {"op":"delete_edge","target":"edge_123","rationale":"Removes unsafe drug prescription due to patient allergy","confidence":0.95},',
                '  {"op":"update_attr","target":"node_456","property":"dosage","value":"250mg","rationale":"Corrects dosage to therapeutic range","confidence":0.8}',
                "]",
                "",
                "JSON Array:",
            ]
        )

        return "\n".join(prompt_parts)


class InnovatorAgent:
    """Main Innovator Agent class for knowledge graph repair."""

    def __init__(
        self,
        llm_driver: LLMDriver,
        template_encoder: TemplateEncoder | None = None,
        prompt_composer: PromptComposer | None = None,
        domain: str = "general",
        guardian_gate: GuardianGate | None = None,
    ) -> None:
        """Initialize Innovator Agent.

        Args:
            llm_driver: LLM driver for generating repairs
            template_encoder: Template encoder for subgraph conversion
            prompt_composer: Prompt composer for repair instructions
            domain: Domain specialization
            guardian_gate: Guardian gate for proposal validation
        """
        self.llm_driver = llm_driver
        self.template_encoder = template_encoder or TemplateEncoder()
        self.prompt_composer = prompt_composer or PromptComposer(domain=domain)
        self.domain = domain
        self.guardian_gate = guardian_gate or GuardianGate()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.repair_history = []
        self.success_rate = 0.0

    async def analyze_violation(self, violation_data: dict[str, Any]) -> ViolationTemplate:
        """Analyze and encode a GDC violation.

        Args:
            violation_data: Raw violation data from GDC extractor

        Returns:
            Structured violation template
        """
        return self.template_encoder.encode_violation(violation_data)

    async def generate_repair_proposals(
        self,
        violation_data: dict[str, Any],
        max_operations: int = 10,
        confidence_threshold: float = 0.5,
    ) -> RepairProposalSet:
        """Generate repair proposals for a GDC violation.

        Args:
            violation_data: Raw violation data
            max_operations: Maximum number of operations to propose
            confidence_threshold: Minimum confidence for operations

        Returns:
            Complete repair proposal
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Encode violation
            violation = await self.analyze_violation(violation_data)

            # Create repair context
            context = self.template_encoder.create_repair_context(violation)

            # Compose prompt
            prompt = self.prompt_composer.compose_repair_prompt(violation, context)
            system_prompt = self.prompt_composer.get_system_prompt()

            # Generate repair operations
            response = await self.llm_driver.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.1,
            )

            # Parse operations
            operations = self._parse_repair_operations(response.text)

            # Filter by confidence
            filtered_ops = [op for op in operations if op.confidence >= confidence_threshold]

            # Limit number of operations
            if len(filtered_ops) > max_operations:
                # Sort by confidence and take top operations
                filtered_ops.sort(key=lambda x: x.confidence, reverse=True)
                filtered_ops = filtered_ops[:max_operations]

            # Create proposal
            generation_time = (asyncio.get_event_loop().time() - start_time) * 1000

            proposal = RepairProposal(
                violation_id=violation.violation_id,
                gdc_rule=violation.gdc_rule,
                operations=filtered_ops,
                model_used=self.llm_driver.config.model_name,
                generation_time_ms=generation_time,
                repair_summary=self._generate_repair_summary(filtered_ops),
                potential_risks=self._assess_potential_risks(filtered_ops),
                validation_notes=self._generate_validation_notes(filtered_ops, violation),
            )

            # Validate through Guardian Gate
            guardian_result = await self._validate_with_guardian(proposal, violation)
            proposal.validation_notes.extend(guardian_result.get("notes", []))

            # Record Guardian decision in metadata
            if "guardian_decision" not in proposal.validation_notes:
                proposal.validation_notes.append(f"Guardian decision: {guardian_result.get('decision', 'UNKNOWN')}")

            # Record in history
            self._record_repair_attempt(proposal, violation)

            return proposal

        except Exception as e:
            self.logger.exception(f"Failed to generate repair proposals: {e}")
            # Return empty proposal with error information
            return RepairProposal(
                violation_id=violation_data.get("violation_id", "unknown"),
                gdc_rule=violation_data.get("rule_name", "unknown"),
                operations=[],
                repair_summary=f"Failed to generate proposals: {e!s}",
            )

    def _parse_repair_operations(self, response_text: str) -> list[RepairOperation]:
        """Parse repair operations from LLM response.

        Args:
            response_text: Raw LLM response

        Returns:
            List of parsed repair operations
        """
        operations = []

        # Extract JSONL lines
        lines = response_text.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                # Parse JSON
                op_data = json.loads(line)

                # Validate required fields
                if "op" not in op_data or "target_id" not in op_data:
                    continue

                # Extract confidence from response or JSON
                confidence = op_data.get("confidence", 0.5)
                if confidence is None:
                    # Try to parse confidence from rationale
                    confidence = self.llm_driver.parse_confidence_from_response(op_data.get("rationale", "")) or 0.5

                # Create operation
                operation = RepairOperation(
                    operation_type=RepairOperationType(op_data["op"]),
                    target_id=op_data["target_id"],
                    rationale=op_data.get("rationale", "No rationale provided"),
                    confidence=float(confidence),
                    source_id=op_data.get("source_id"),
                    relationship_type=op_data.get("relationship_type"),
                    property_name=op_data.get("property_name"),
                    property_value=op_data.get("property_value"),
                    merge_target_id=op_data.get("merge_target_id"),
                    node_type=op_data.get("node_type"),
                    properties=op_data.get("properties"),
                )

                # Assess safety criticality
                operation.safety_critical = self._is_safety_critical(operation)
                operation.estimated_impact = self._estimate_impact(operation)

                operations.append(operation)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"Failed to parse operation on line {line_num}: {e}")
                continue

        return operations

    def _parse_repair_operations_enhanced(self, response_text: str) -> list[RepairOperation]:
        """Enhanced parsing with better JSON validation and confidence extraction.

        Args:
            response_text: Raw LLM response

        Returns:
            List of parsed repair operations
        """
        operations = []

        # Try to parse as JSON array first (preferred format)
        try:
            # Look for JSON array in the response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                operation_list = json.loads(json_str)

                for op_data in operation_list:
                    try:
                        # Validate required fields
                        if not isinstance(op_data, dict):
                            continue
                        if "op" not in op_data or "target" not in op_data:
                            continue

                        # Extract confidence with validation
                        confidence = op_data.get("confidence", 0.5)
                        if not isinstance(confidence, int | float):
                            confidence = 0.5
                        confidence = max(0.0, min(1.0, float(confidence)))

                        # Create operation with enhanced field mapping
                        operation = RepairOperation(
                            operation_type=RepairOperationType(op_data["op"]),
                            target_id=op_data["target"],
                            rationale=op_data.get("rationale", "No rationale provided"),
                            confidence=confidence,
                            source_id=op_data.get("source"),
                            relationship_type=op_data.get("type"),
                            property_name=op_data.get("property"),
                            property_value=op_data.get("value"),
                            merge_target_id=op_data.get("merge_target"),
                            node_type=op_data.get("node_type"),
                            properties=op_data.get("properties"),
                        )

                        # Assess safety criticality
                        operation.safety_critical = self._is_safety_critical(operation)
                        operation.estimated_impact = self._estimate_impact(operation)

                        operations.append(operation)

                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Failed to parse operation in array: {e}")
                        continue

                return operations
        except json.JSONDecodeError:
            pass

        # Fallback to JSONL parsing
        return self._parse_repair_operations(response_text)

    def _record_repair_attempt_enhanced(self, proposal_set: RepairProposalSet, violation: ViolationTemplate) -> None:
        """Record repair attempt for performance tracking with enhanced metrics."""
        self.repair_history.append(
            {
                "timestamp": datetime.now(),
                "violation_id": proposal_set.violation_id,
                "gdc_rule": proposal_set.gdc_rule,
                "operations_count": len(proposal_set.proposals),
                "overall_confidence": proposal_set.overall_confidence,
                "safety_score": proposal_set.safety_score,
                "generation_time_ms": proposal_set.generation_time_ms,
                "is_valid": proposal_set.is_valid,
                "validation_errors": len(proposal_set.validation_errors),
                "validation_warnings": len(proposal_set.validation_warnings),
            }
        )

        # Keep only recent history
        if len(self.repair_history) > 100:
            self.repair_history = self.repair_history[-100:]

    def _is_safety_critical(self, operation: RepairOperation) -> bool:
        """Determine if operation is safety-critical."""
        # Domain-specific safety rules
        if self.domain == "medical":
            # Medical safety patterns
            if operation.operation_type == RepairOperationType.DELETE_EDGE:
                if operation.relationship_type in ["ALLERGIC_TO", "PRESCRIBES"]:
                    return True

            # Safety-critical properties
            safety_keywords = ["allergy", "dosage", "medication", "prescription"]
            if any(keyword in operation.rationale.lower() for keyword in safety_keywords):
                return True

        # General safety patterns
        if operation.operation_type == RepairOperationType.DELETE_NODE:
            return True  # Node deletion is always potentially risky

        return False

    def _estimate_impact(self, operation: RepairOperation) -> str:
        """Estimate impact level of operation."""
        if operation.operation_type in [
            RepairOperationType.DELETE_NODE,
            RepairOperationType.MERGE_NODES,
        ]:
            return "high"
        if operation.operation_type == RepairOperationType.DELETE_EDGE or operation.operation_type in [
            RepairOperationType.ADD_EDGE,
            RepairOperationType.ADD_NODE,
        ]:
            return "medium"
        # UPDATE_ATTR
        return "low"

    def _generate_repair_summary(self, operations: list[RepairOperation]) -> str:
        """Generate summary of repair operations."""
        if not operations:
            return "No repair operations proposed"

        op_counts = {}
        for op in operations:
            op_type = op.operation_type.value
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        summary_parts = []
        for op_type, count in op_counts.items():
            summary_parts.append(f"{count} {op_type} operation{'s' if count > 1 else ''}")

        base_summary = f"Proposed {len(operations)} total operations: {', '.join(summary_parts)}"

        # Add confidence assessment
        avg_confidence = sum(op.confidence for op in operations) / len(operations)
        confidence_desc = "high" if avg_confidence >= 0.8 else "medium" if avg_confidence >= 0.6 else "low"

        return f"{base_summary}. Average confidence: {confidence_desc} ({avg_confidence:.2f})"

    def _assess_potential_risks(self, operations: list[RepairOperation]) -> list[str]:
        """Assess potential risks of proposed operations."""
        risks = []

        # Check for high-impact operations
        high_impact_ops = [op for op in operations if op.estimated_impact == "high"]
        if high_impact_ops:
            risks.append(f"{len(high_impact_ops)} high-impact operations that may affect graph structure significantly")

        # Check for safety-critical operations
        safety_ops = [op for op in operations if op.safety_critical]
        if safety_ops:
            risks.append(f"{len(safety_ops)} safety-critical operations requiring extra validation")

        # Check for low-confidence operations
        low_conf_ops = [op for op in operations if op.confidence < 0.6]
        if low_conf_ops:
            risks.append(f"{len(low_conf_ops)} operations with low confidence scores")

        # Domain-specific risks
        if self.domain == "medical":
            delete_ops = [op for op in operations if op.operation_type == RepairOperationType.DELETE_EDGE]
            for op in delete_ops:
                if any(keyword in op.rationale.lower() for keyword in ["prescription", "treatment", "allergy"]):
                    risks.append("Deletion of medical relationships may affect patient safety")
                    break

        return risks

    def _generate_validation_notes(self, operations: list[RepairOperation], violation: ViolationTemplate) -> list[str]:
        """Generate validation notes for manual review."""
        notes = []

        # Check completeness
        if len(operations) == 0:
            notes.append("No operations proposed - manual intervention may be required")
        elif len(operations) == 1:
            notes.append("Single operation proposed - verify this fully resolves the violation")

        # Check for identity preservation
        core_entities = {node.node_id for node in violation.nodes if "id" in node.label.lower()}
        deleted_entities = {op.target_id for op in operations if op.operation_type == RepairOperationType.DELETE_NODE}

        if core_entities & deleted_entities:
            notes.append("Operations may delete core entity nodes - verify this is intended")

        # Domain-specific validation
        if self.domain == "medical":
            # Check for medical validation needs
            dosage_ops = [op for op in operations if "dosage" in str(op.property_name).lower()]
            if dosage_ops:
                notes.append("Dosage modifications require clinical validation")

        # Check for operation conflicts
        edge_deletions = {op.target_id for op in operations if op.operation_type == RepairOperationType.DELETE_EDGE}
        edge_updates = {
            op.target_id
            for op in operations
            if op.operation_type == RepairOperationType.UPDATE_ATTR and op.target_id.startswith("edge")
        }

        conflicts = edge_deletions & edge_updates
        if conflicts:
            notes.append(f"Potential conflicts: {len(conflicts)} edges are both deleted and updated")

        return notes

    def _record_repair_attempt(self, proposal: RepairProposal, violation: ViolationTemplate) -> None:
        """Record repair attempt for performance tracking."""
        self.repair_history.append(
            {
                "timestamp": datetime.now(),
                "violation_id": proposal.violation_id,
                "gdc_rule": proposal.gdc_rule,
                "operations_count": len(proposal.operations),
                "overall_confidence": proposal.overall_confidence,
                "safety_score": proposal.safety_score,
                "generation_time_ms": proposal.generation_time_ms,
            }
        )

        # Keep only recent history
        if len(self.repair_history) > 100:
            self.repair_history = self.repair_history[-100:]

    async def validate_proposal(self, proposal: RepairProposal) -> dict[str, Any]:
        """Validate a repair proposal for correctness and safety.

        Args:
            proposal: The repair proposal to validate

        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        # Check for empty proposal
        if not proposal.operations:
            validation["warnings"].append("No operations proposed")
            validation["recommendations"].append("Consider manual review of the violation")

        # Validate individual operations
        for op in proposal.operations:
            # Check required fields
            if not op.target_id:
                validation["errors"].append(f"Operation {op.operation_id} missing target_id")
                validation["is_valid"] = False

            if not op.rationale:
                validation["warnings"].append(f"Operation {op.operation_id} missing rationale")

            # Check confidence scores
            if op.confidence < 0.3:
                validation["warnings"].append(f"Operation {op.operation_id} has very low confidence: {op.confidence}")

        # Domain-specific validation
        if self.domain == "medical":
            safety_ops = proposal.get_safety_critical_operations()
            if safety_ops:
                validation["recommendations"].append(
                    f"Review {len(safety_ops)} safety-critical operations before applying"
                )

        return validation

    def get_performance_stats(self) -> dict[str, Any]:
        """Get agent performance statistics."""
        if not self.repair_history:
            return {"message": "No repair history available"}

        recent_repairs = self.repair_history[-10:] if len(self.repair_history) >= 10 else self.repair_history

        return {
            "total_repairs_attempted": len(self.repair_history),
            "average_confidence": sum(r["overall_confidence"] for r in recent_repairs) / len(recent_repairs),
            "average_generation_time_ms": sum(r["generation_time_ms"] for r in recent_repairs) / len(recent_repairs),
            "average_operations_per_repair": sum(r["operations_count"] for r in recent_repairs) / len(recent_repairs),
            "domain": self.domain,
            "model_used": self.llm_driver.config.model_name,
        }

    @classmethod
    async def create_default(cls, model_name: str = "llama3.2:3b", domain: str = "general") -> "InnovatorAgent":
        """Create InnovatorAgent with default configuration.

        Args:
            model_name: LLM model to use
            domain: Domain specialization

        Returns:
            Configured InnovatorAgent
        """
        from .llm_driver import ModelBackend

        # Create model config
        config = ModelConfig(
            model_name=model_name,
            backend=ModelBackend.OLLAMA,
            temperature=0.1,
            max_tokens=2048,
        )

        # Create driver
        driver = LLMDriver(config)

        # Create agent
        return cls(llm_driver=driver, domain=domain)

    async def _validate_with_guardian(self, proposal: RepairProposal, violation: ViolationTemplate) -> dict[str, Any]:
        """Validate repair proposal through Guardian Gate.

        Args:
            proposal: The repair proposal to validate
            violation: The original violation

        Returns:
            Dictionary with validation results
        """
        try:
            # Create mock violation object for Guardian Gate

            # Convert ViolationTemplate to Violation format expected by Guardian
            mock_violation = type(
                "MockViolation",
                (),
                {
                    "id": violation.violation_id,
                    "severity": getattr(violation, "severity", "medium"),
                    "domain": self.domain,
                    "subgraph": getattr(violation, "subgraph", {"nodes": [], "edges": []}),
                },
            )()

            # Validate repair through Guardian
            decision = await self.guardian_gate.evaluate_repair(proposal.operations, mock_violation)

            validation_result = {"decision": decision, "notes": []}

            if decision == "APPLY":
                validation_result["notes"].append("Guardian Gate approved all repair operations")
            elif decision == "QUARANTINE":
                validation_result["notes"].append("Guardian Gate flagged proposals for manual review")
                proposal.potential_risks.append("Guardian validation required manual review")
            else:  # REJECT
                validation_result["notes"].append("Guardian Gate rejected repair proposals")
                proposal.potential_risks.append("Guardian blocked this repair due to safety concerns")

            self.logger.info(f"Guardian Gate decision for violation {violation.violation_id}: {decision}")
            return validation_result

        except Exception as e:
            self.logger.exception(f"Guardian validation failed: {e}")
            return {
                "decision": "ERROR",
                "notes": [f"Guardian validation failed: {e!s}"],
            }
