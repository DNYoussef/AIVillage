"""Template Encoder for GDC Violation Subgraphs

Converts violating subgraphs to human-readable template sentences
including critical properties (ids, labels, domain fields).
"""

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any


class DomainField(Enum):
    """Domain-specific critical fields for different KG schemas"""
    # Medical domain
    ALLERGY = "allergy"
    DOSAGE = "dosage"
    MEDICATION = "medication"
    CONDITION = "condition"
    DATE = "date"
    SEVERITY = "severity"

    # General domain
    CONFIDENCE = "confidence"
    TIMESTAMP = "timestamp"
    SOURCE = "source"
    CATEGORY = "category"
    STATUS = "status"


@dataclass
class NodeTemplate:
    """Template representation of a graph node"""
    node_id: str
    label: str
    properties: dict[str, Any]
    critical_fields: set[DomainField]

    def to_sentence(self) -> str:
        """Convert node to human-readable sentence (research shows template > raw format)"""
        # Extract critical properties with natural language formatting
        critical_props = []
        for field in self.critical_fields:
            if field.value in self.properties:
                value = self.properties[field.value]
                # Natural language property descriptions
                if field == DomainField.ALLERGY and isinstance(value, list):
                    critical_props.append(f"allergic to {', '.join(str(v) for v in value)}")
                elif field == DomainField.DOSAGE:
                    critical_props.append(f"prescribed at {value}")
                elif field == DomainField.CONDITION:
                    critical_props.append(f"diagnosed with {value}")
                elif field == DomainField.CONFIDENCE:
                    critical_props.append(f"confidence level {value}")
                else:
                    critical_props.append(f"{field.value} is {value}")

        # Build natural sentence structure
        if self.label == "Patient":
            base = f"Patient {self.node_id}"
            if critical_props:
                return f"{base} who is {', and '.join(critical_props)}"
            return f"{base} (no critical medical information)"
        if self.label == "Medication":
            base = f"Medication {self.node_id}"
            med_name = self.properties.get("name", "unknown medication")
            if critical_props:
                return f"{base} ({med_name}) with {', '.join(critical_props)}"
            return f"{base} ({med_name})"
        # Generic entity format
        if critical_props:
            return f"{self.label} {self.node_id} that has {', '.join(critical_props)}"
        return f"{self.label} {self.node_id}"


@dataclass
class EdgeTemplate:
    """Template representation of a graph edge"""
    edge_id: str
    source_id: str
    target_id: str
    relationship: str
    properties: dict[str, Any]
    critical_fields: set[DomainField]

    def to_sentence(self) -> str:
        """Convert edge to human-readable sentence with natural language"""
        # Extract critical properties with natural language formatting
        critical_props = []
        for field in self.critical_fields:
            if field.value in self.properties:
                value = self.properties[field.value]
                if field == DomainField.DOSAGE:
                    critical_props.append(f"at dosage {value}")
                elif field == DomainField.DATE:
                    critical_props.append(f"on {value}")
                elif field == DomainField.CONFIDENCE:
                    critical_props.append(f"with {value} confidence")
                else:
                    critical_props.append(f"{field.value} {value}")

        # Build natural sentence based on relationship type
        if self.relationship == "PRESCRIBES":
            props_text = f" {', '.join(critical_props)}" if critical_props else ""
            return f"Prescription relationship where {self.source_id} prescribes {self.target_id}{props_text}"
        if self.relationship == "ALLERGIC_TO":
            severity = ""
            if critical_props:
                severity = f" ({', '.join(critical_props)})"
            return f"Allergy relationship where {self.source_id} is allergic to {self.target_id}{severity}"
        if self.relationship == "TREATS":
            props_text = f" {', '.join(critical_props)}" if critical_props else ""
            return f"Treatment relationship where {self.source_id} treats condition {self.target_id}{props_text}"
        # Generic relationship format
        props_text = f" with {', '.join(critical_props)}" if critical_props else ""
        return f"Relationship {self.edge_id} where {self.source_id} {self.relationship.lower().replace('_', ' ')} {self.target_id}{props_text}"


@dataclass
class ViolationTemplate:
    """Complete template for a GDC violation subgraph"""
    violation_id: str
    gdc_rule: str
    violated_pattern: str
    nodes: list[NodeTemplate]
    edges: list[EdgeTemplate]
    context: dict[str, Any]

    def to_description(self) -> str:
        """Convert entire violation to human-readable description"""
        lines = [
            f"=== GDC Violation {self.violation_id} ===",
            f"Rule: {self.gdc_rule}",
            f"Pattern: {self.violated_pattern}",
            "",
            "Involved Nodes:",
        ]

        for node in self.nodes:
            lines.append(f"  - {node.to_sentence()}")

        lines.extend([
            "",
            "Involved Edges:",
        ])

        for edge in self.edges:
            lines.append(f"  - {edge.to_sentence()}")

        if self.context:
            lines.extend([
                "",
                "Additional Context:",
                f"  {json.dumps(self.context, indent=2)}"
            ])

        return "\n".join(lines)


class TemplateEncoder:
    """Encodes GDC violation subgraphs into human-readable templates"""

    def __init__(self, domain_config: dict[str, Any] | None = None):
        """Initialize template encoder with domain-specific configuration

        Args:
            domain_config: Configuration for domain-specific field mappings
        """
        self.domain_config = domain_config or {}
        self.field_mappings = self._load_field_mappings()

    def _load_field_mappings(self) -> dict[str, set[DomainField]]:
        """Load mappings from node/edge types to critical fields"""
        # Default mappings for HypeRAG schema
        mappings = {
            # Node type mappings
            "SemanticNode": {DomainField.CONFIDENCE, DomainField.TIMESTAMP, DomainField.SOURCE},
            "EntityNode": {DomainField.CATEGORY, DomainField.CONFIDENCE},
            "ConceptNode": {DomainField.CATEGORY, DomainField.SOURCE},
            "TemporalNode": {DomainField.DATE, DomainField.TIMESTAMP},

            # Medical domain extensions
            "Patient": {DomainField.ALLERGY, DomainField.CONDITION},
            "Medication": {DomainField.DOSAGE, DomainField.MEDICATION, DomainField.ALLERGY},
            "Treatment": {DomainField.DOSAGE, DomainField.DATE, DomainField.MEDICATION},
            "Condition": {DomainField.SEVERITY, DomainField.DATE, DomainField.STATUS},

            # Edge type mappings
            "HYPERCONNECTION": {DomainField.CONFIDENCE, DomainField.TIMESTAMP},
            "RELATES_TO": {DomainField.CONFIDENCE, DomainField.SOURCE},
            "PRESCRIBES": {DomainField.DOSAGE, DomainField.DATE, DomainField.MEDICATION},
            "ALLERGIC_TO": {DomainField.ALLERGY, DomainField.SEVERITY},
            "TREATS": {DomainField.CONDITION, DomainField.DATE},
        }

        # Override with domain config
        if "field_mappings" in self.domain_config:
            mappings.update(self.domain_config["field_mappings"])

        return mappings

    def encode_node(self, node_data: dict[str, Any]) -> NodeTemplate:
        """Encode a single node into template format

        Args:
            node_data: Node data from Neo4j result

        Returns:
            NodeTemplate representation
        """
        node_id = node_data.get("id", "unknown")
        labels = node_data.get("labels", [])
        label = labels[0] if labels else "Unknown"
        properties = node_data.get("properties", {})

        # Determine critical fields for this node type
        critical_fields = set()
        for node_label in labels:
            if node_label in self.field_mappings:
                critical_fields.update(self.field_mappings[node_label])

        # Add any fields specified in domain config
        if "default_node_fields" in self.domain_config:
            for field_name in self.domain_config["default_node_fields"]:
                try:
                    critical_fields.add(DomainField(field_name))
                except ValueError:
                    pass  # Skip invalid field names

        return NodeTemplate(
            node_id=str(node_id),
            label=label,
            properties=properties,
            critical_fields=critical_fields
        )

    def encode_edge(self, edge_data: dict[str, Any]) -> EdgeTemplate:
        """Encode a single edge into template format

        Args:
            edge_data: Edge data from Neo4j result

        Returns:
            EdgeTemplate representation
        """
        edge_id = edge_data.get("id", "unknown")
        source_id = edge_data.get("startNode", "unknown")
        target_id = edge_data.get("endNode", "unknown")
        relationship = edge_data.get("type", "UNKNOWN")
        properties = edge_data.get("properties", {})

        # Determine critical fields for this edge type
        critical_fields = set()
        if relationship in self.field_mappings:
            critical_fields.update(self.field_mappings[relationship])

        # Add any fields specified in domain config
        if "default_edge_fields" in self.domain_config:
            for field_name in self.domain_config["default_edge_fields"]:
                try:
                    critical_fields.add(DomainField(field_name))
                except ValueError:
                    pass  # Skip invalid field names

        return EdgeTemplate(
            edge_id=str(edge_id),
            source_id=str(source_id),
            target_id=str(target_id),
            relationship=relationship,
            properties=properties,
            critical_fields=critical_fields
        )

    def encode_violation(self, violation_data: dict[str, Any]) -> ViolationTemplate:
        """Encode a complete GDC violation into template format

        Args:
            violation_data: Complete violation data including subgraph

        Returns:
            ViolationTemplate representation
        """
        violation_id = violation_data.get("violation_id", "unknown")
        gdc_rule = violation_data.get("rule_name", "unknown_rule")
        violated_pattern = violation_data.get("violated_pattern", "unknown_pattern")

        # Extract subgraph data
        subgraph = violation_data.get("subgraph", {})
        nodes_data = subgraph.get("nodes", [])
        edges_data = subgraph.get("edges", [])

        # Encode nodes and edges
        nodes = [self.encode_node(node) for node in nodes_data]
        edges = [self.encode_edge(edge) for edge in edges_data]

        # Extract additional context
        context = {
            "confidence_score": violation_data.get("confidence_score"),
            "severity": violation_data.get("severity"),
            "timestamp": violation_data.get("detected_at"),
            "rule_description": violation_data.get("rule_description")
        }

        # Remove None values
        context = {k: v for k, v in context.items() if v is not None}

        return ViolationTemplate(
            violation_id=violation_id,
            gdc_rule=gdc_rule,
            violated_pattern=violated_pattern,
            nodes=nodes,
            edges=edges,
            context=context
        )

    def create_repair_context(self, violation: ViolationTemplate) -> dict[str, Any]:
        """Create structured context for repair operations

        Args:
            violation: The violation template

        Returns:
            Context dictionary for repair prompting
        """
        # Identify key entities and relationships
        entity_summary = {}
        for node in violation.nodes:
            node_type = node.label
            if node_type not in entity_summary:
                entity_summary[node_type] = []
            entity_summary[node_type].append({
                "id": node.node_id,
                "critical_props": {
                    field.value: node.properties.get(field.value)
                    for field in node.critical_fields
                    if field.value in node.properties
                }
            })

        relationship_summary = {}
        for edge in violation.edges:
            rel_type = edge.relationship
            if rel_type not in relationship_summary:
                relationship_summary[rel_type] = []
            relationship_summary[rel_type].append({
                "id": edge.edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "critical_props": {
                    field.value: edge.properties.get(field.value)
                    for field in edge.critical_fields
                    if field.value in edge.properties
                }
            })

        return {
            "violation_summary": {
                "id": violation.violation_id,
                "rule": violation.gdc_rule,
                "pattern": violation.violated_pattern
            },
            "entity_analysis": entity_summary,
            "relationship_analysis": relationship_summary,
            "repair_constraints": {
                "preserve_identity_edges": True,
                "maintain_core_entities": True,
                "require_rationale": True
            },
            "domain_context": self.domain_config.get("repair_guidelines", {})
        }

    def extract_critical_conflicts(self, violation: ViolationTemplate) -> list[dict[str, Any]]:
        """Extract specific conflicts that need resolution

        Args:
            violation: The violation template

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Check for property conflicts
        for node in violation.nodes:
            for field in node.critical_fields:
                if field.value in node.properties:
                    value = node.properties[field.value]

                    # Domain-specific conflict detection
                    if field == DomainField.ALLERGY and isinstance(value, list):
                        conflicts.append({
                            "type": "allergy_conflict",
                            "entity": node.node_id,
                            "field": field.value,
                            "value": value,
                            "description": f"Node {node.node_id} has allergy information that may conflict with treatments"
                        })

                    elif field == DomainField.CONFIDENCE and isinstance(value, (int, float)) and value < 0.5:
                        conflicts.append({
                            "type": "low_confidence",
                            "entity": node.node_id,
                            "field": field.value,
                            "value": value,
                            "description": f"Node {node.node_id} has low confidence score: {value}"
                        })

        # Check for relationship conflicts
        for edge in violation.edges:
            for field in edge.critical_fields:
                if field.value in edge.properties:
                    value = edge.properties[field.value]

                    if field == DomainField.DOSAGE and edge.relationship == "PRESCRIBES":
                        conflicts.append({
                            "type": "dosage_validation",
                            "entity": edge.edge_id,
                            "field": field.value,
                            "value": value,
                            "description": f"Prescription edge {edge.edge_id} specifies dosage: {value}"
                        })

        return conflicts

    def generate_repair_summary(self, violation: ViolationTemplate) -> str:
        """Generate a concise summary for repair prompting

        Args:
            violation: The violation template

        Returns:
            Human-readable repair summary
        """
        # Extract key information
        node_count = len(violation.nodes)
        edge_count = len(violation.edges)
        conflicts = self.extract_critical_conflicts(violation)

        summary_lines = [
            f"Violation: {violation.gdc_rule}",
            f"Scope: {node_count} nodes, {edge_count} edges",
            f"Critical conflicts: {len(conflicts)}"
        ]

        if conflicts:
            summary_lines.append("Key issues:")
            for conflict in conflicts[:3]:  # Show top 3 conflicts
                summary_lines.append(f"  - {conflict['description']}")

        return "\n".join(summary_lines)
