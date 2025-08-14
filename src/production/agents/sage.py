"""Sage Agent - Deep research and knowledge synthesis expert.

The Sage Agent specializes in comprehensive research, knowledge synthesis,
and providing deep insights within the AIVillage ecosystem.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResearchDomain(Enum):
    """Research domains supported by Sage."""

    ARTIFICIAL_INTELLIGENCE = "ai"
    MACHINE_LEARNING = "ml"
    COMPUTER_SCIENCE = "cs"
    MATHEMATICS = "math"
    PHYSICS = "physics"
    BIOLOGY = "biology"
    PHILOSOPHY = "philosophy"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    INTERDISCIPLINARY = "interdisciplinary"


class KnowledgeConfidence(Enum):
    """Confidence levels for knowledge assertions."""

    VERY_HIGH = 0.95
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.55
    VERY_LOW = 0.40


@dataclass
class ResearchQuery:
    """Research query with context and requirements."""

    query_id: str
    question: str
    domain: ResearchDomain
    depth_required: str  # "shallow", "medium", "deep", "comprehensive"
    sources_required: list[str]
    confidence_threshold: float
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    findings: dict[str, Any] | None = None


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph."""

    node_id: str
    concept: str
    domain: ResearchDomain
    confidence: float
    sources: list[str]
    connections: set[str] = field(default_factory=set)
    evidence: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class SageAgent:
    """Deep research and knowledge synthesis expert."""

    def __init__(self, spec=None) -> None:
        """Initialize Sage Agent."""
        self.spec = spec
        self.name = "Sage"
        self.role_description = "Deep research and knowledge synthesis expert"

        # Knowledge management
        self.knowledge_graph: dict[str, KnowledgeNode] = {}
        self.research_queries: dict[str, ResearchQuery] = {}
        self.domain_expertise: dict[ResearchDomain, float] = {}

        # Research capabilities
        self.supported_domains = list(ResearchDomain)
        self.research_methodologies = [
            "systematic_review",
            "meta_analysis",
            "comparative_analysis",
            "longitudinal_study",
            "cross_sectional",
            "case_study",
        ]

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.kpi_scores: dict[str, float] = {}

        # Initialize domain expertise
        self._initialize_domain_expertise()

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process research and knowledge requests."""
        task_type = request.get("task", "unknown")

        if task_type == "ping":
            return {
                "status": "completed",
                "agent": "sage",
                "result": "Deep research system online",
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif task_type == "research":
            return self._conduct_research(request)
        elif task_type == "synthesize_knowledge":
            return self._synthesize_knowledge(request)
        elif task_type == "validate_claim":
            return self._validate_claim(request)
        elif task_type == "expert_analysis":
            return self._provide_expert_analysis(request)
        elif task_type == "knowledge_graph_query":
            return self._query_knowledge_graph(request)
        elif task_type == "research_methodology":
            return self._recommend_methodology(request)
        else:
            return {
                "status": "completed",
                "agent": "sage",
                "result": f"Researched topic: {task_type}",
                "insights": self._generate_insights(request),
                "confidence": self._assess_confidence(request),
            }

    def _conduct_research(self, request: dict[str, Any]) -> dict[str, Any]:
        """Conduct comprehensive research on a topic."""
        question = request.get("question", "")
        domain = request.get("domain", "ai")
        depth = request.get("depth", "medium")

        query_id = f"research_{int(time.time() * 1000)}"

        try:
            research_domain = ResearchDomain(domain.lower())
        except ValueError:
            research_domain = ResearchDomain.INTERDISCIPLINARY

        # Create research query
        query = ResearchQuery(
            query_id=query_id,
            question=question,
            domain=research_domain,
            depth_required=depth,
            sources_required=request.get("sources", ["academic", "primary"]),
            confidence_threshold=request.get("confidence_threshold", 0.7),
        )

        # Conduct research
        findings = self._perform_research(query)

        query.findings = findings
        query.completed_at = time.time()
        self.research_queries[query_id] = query

        # Update knowledge graph
        self._update_knowledge_graph(findings, research_domain)

        return {
            "status": "completed",
            "agent": "sage",
            "result": "Research completed successfully",
            "query_id": query_id,
            "findings": findings,
            "domain": domain,
            "depth": depth,
            "confidence": findings.get("overall_confidence", 0.7),
            "sources_consulted": findings.get("sources_count", 0),
        }

    def _synthesize_knowledge(self, request: dict[str, Any]) -> dict[str, Any]:
        """Synthesize knowledge from multiple sources or domains."""
        concepts = request.get("concepts", [])
        domains = request.get("domains", [])
        synthesis_type = request.get("type", "comparative")  # comparative, integrative, meta

        # Gather relevant knowledge nodes
        relevant_nodes = self._gather_relevant_nodes(concepts, domains)

        # Perform synthesis
        synthesis = self._perform_synthesis(relevant_nodes, synthesis_type)

        return {
            "status": "completed",
            "agent": "sage",
            "result": "Knowledge synthesis completed",
            "synthesis_type": synthesis_type,
            "synthesis": synthesis,
            "nodes_analyzed": len(relevant_nodes),
            "confidence": synthesis.get("confidence", 0.75),
            "novel_insights": synthesis.get("novel_insights", []),
        }

    def _validate_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate a knowledge claim against available evidence."""
        claim = request.get("claim", "")
        domain = request.get("domain", "ai")

        # Analyze claim
        validation_result = self._analyze_claim(claim, domain)

        return {
            "status": "completed",
            "agent": "sage",
            "result": "Claim validation completed",
            "claim": claim,
            "validity": validation_result["validity"],
            "confidence": validation_result["confidence"],
            "supporting_evidence": validation_result["supporting_evidence"],
            "contradicting_evidence": validation_result["contradicting_evidence"],
            "uncertainty_factors": validation_result["uncertainty_factors"],
        }

    def _provide_expert_analysis(self, request: dict[str, Any]) -> dict[str, Any]:
        """Provide expert analysis on a complex topic."""
        topic = request.get("topic", "")
        analysis_type = request.get("analysis_type", "comprehensive")
        domain = request.get("domain", "ai")

        try:
            research_domain = ResearchDomain(domain.lower())
        except ValueError:
            research_domain = ResearchDomain.INTERDISCIPLINARY

        # Check domain expertise
        expertise_level = self.domain_expertise.get(research_domain, 0.7)

        # Perform expert analysis
        analysis = self._perform_expert_analysis(topic, research_domain, analysis_type)

        return {
            "status": "completed",
            "agent": "sage",
            "result": "Expert analysis completed",
            "topic": topic,
            "domain": domain,
            "expertise_level": expertise_level,
            "analysis": analysis,
            "key_insights": analysis.get("key_insights", []),
            "implications": analysis.get("implications", []),
            "recommendations": analysis.get("recommendations", []),
        }

    def _query_knowledge_graph(self, request: dict[str, Any]) -> dict[str, Any]:
        """Query the internal knowledge graph."""
        query = request.get("query", "")
        max_results = request.get("max_results", 10)
        min_confidence = request.get("min_confidence", 0.5)

        # Search knowledge graph
        results = self._search_knowledge_graph(query, max_results, min_confidence)

        return {
            "status": "completed",
            "agent": "sage",
            "result": "Knowledge graph query completed",
            "query": query,
            "results": results,
            "total_nodes": len(self.knowledge_graph),
            "results_count": len(results),
        }

    def _recommend_methodology(self, request: dict[str, Any]) -> dict[str, Any]:
        """Recommend research methodology for a given question."""
        research_question = request.get("question", "")
        domain = request.get("domain", "ai")
        constraints = request.get("constraints", {})

        # Analyze question to determine best methodology
        methodology = self._determine_methodology(research_question, domain, constraints)

        return {
            "status": "completed",
            "agent": "sage",
            "result": "Research methodology recommendation completed",
            "question": research_question,
            "recommended_methodology": methodology["name"],
            "rationale": methodology["rationale"],
            "steps": methodology["steps"],
            "expected_duration": methodology["duration"],
            "resource_requirements": methodology["resources"],
        }

    def _perform_research(self, query: ResearchQuery) -> dict[str, Any]:
        """Perform research based on query parameters."""
        # Simulate comprehensive research process
        findings = {
            "summary": f"Research findings for: {query.question}",
            "key_points": [],
            "evidence_strength": "medium",
            "sources_count": 0,
            "overall_confidence": 0.75,
        }

        # Depth-dependent research
        if query.depth_required == "shallow":
            findings["key_points"] = [
                "Basic overview of the topic",
                "Primary definitions and concepts",
                "Surface-level analysis",
            ]
            findings["sources_count"] = 5
            findings["overall_confidence"] = 0.6

        elif query.depth_required == "medium":
            findings["key_points"] = [
                "Comprehensive overview with context",
                "Multiple perspectives considered",
                "Intermediate-level analysis",
                "Some empirical evidence",
            ]
            findings["sources_count"] = 15
            findings["overall_confidence"] = 0.75

        elif query.depth_required == "deep":
            findings["key_points"] = [
                "Thorough investigation with historical context",
                "Multiple methodological approaches",
                "Critical analysis of competing theories",
                "Substantial empirical evidence",
                "Identification of knowledge gaps",
            ]
            findings["sources_count"] = 30
            findings["overall_confidence"] = 0.85

        else:  # comprehensive
            findings["key_points"] = [
                "Exhaustive investigation across multiple domains",
                "Systematic review of all relevant literature",
                "Meta-analysis of empirical studies",
                "Novel theoretical synthesis",
                "Comprehensive gap analysis",
                "Future research directions identified",
            ]
            findings["sources_count"] = 50
            findings["overall_confidence"] = 0.90

        # Domain-specific adjustments
        domain_expertise = self.domain_expertise.get(query.domain, 0.7)
        findings["overall_confidence"] *= domain_expertise

        # Add domain-specific insights
        findings["domain_insights"] = self._generate_domain_insights(query.domain, query.question)

        return findings

    def _generate_domain_insights(self, domain: ResearchDomain, question: str) -> list[str]:
        """Generate domain-specific insights."""
        insights = []

        if domain == ResearchDomain.ARTIFICIAL_INTELLIGENCE:
            insights = [
                "Consider ethical implications of AI systems",
                "Examine scalability and computational requirements",
                "Evaluate human-AI interaction patterns",
                "Assess safety and alignment considerations",
            ]
        elif domain == ResearchDomain.MACHINE_LEARNING:
            insights = [
                "Analyze model interpretability and explainability",
                "Consider data quality and bias implications",
                "Evaluate generalization capabilities",
                "Assess computational efficiency",
            ]
        elif domain == ResearchDomain.MATHEMATICS:
            insights = [
                "Verify mathematical rigor and proof validity",
                "Consider alternative formulations",
                "Examine computational complexity",
                "Explore practical applications",
            ]
        elif domain == ResearchDomain.PHYSICS:
            insights = [
                "Consider experimental validation requirements",
                "Examine theoretical consistency",
                "Evaluate measurement precision needs",
                "Assess implications for fundamental understanding",
            ]
        else:
            insights = [
                "Consider interdisciplinary connections",
                "Examine methodological approaches",
                "Evaluate empirical support",
                "Assess practical implications",
            ]

        return insights

    def _update_knowledge_graph(self, findings: dict[str, Any], domain: ResearchDomain) -> None:
        """Update knowledge graph with new findings."""
        # Extract key concepts from findings
        concepts = findings.get("key_points", [])
        confidence = findings.get("overall_confidence", 0.7)

        for i, concept in enumerate(concepts):
            node_id = f"node_{domain.value}_{int(time.time())}_{i}"

            node = KnowledgeNode(
                node_id=node_id,
                concept=concept,
                domain=domain,
                confidence=confidence,
                sources=["research_query"],
                evidence={"findings": findings},
            )

            self.knowledge_graph[node_id] = node

            # Create connections between related nodes
            self._create_node_connections(node)

    def _create_node_connections(self, new_node: KnowledgeNode) -> None:
        """Create connections between knowledge nodes."""
        # Simple similarity-based connection
        for node_id, existing_node in self.knowledge_graph.items():
            if node_id != new_node.node_id:
                # Check for domain overlap or concept similarity
                if (
                    existing_node.domain == new_node.domain
                    or existing_node.domain == ResearchDomain.INTERDISCIPLINARY
                    or new_node.domain == ResearchDomain.INTERDISCIPLINARY
                ):
                    # Simple keyword-based similarity
                    if self._calculate_concept_similarity(new_node.concept, existing_node.concept) > 0.3:
                        new_node.connections.add(node_id)
                        existing_node.connections.add(new_node.node_id)

    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concepts."""
        # Simple word overlap similarity
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _gather_relevant_nodes(self, concepts: list[str], domains: list[str]) -> list[KnowledgeNode]:
        """Gather relevant knowledge nodes for synthesis."""
        relevant_nodes = []

        for node in self.knowledge_graph.values():
            # Check domain relevance
            if not domains or node.domain.value in domains:
                # Check concept relevance
                for concept in concepts:
                    if self._calculate_concept_similarity(node.concept, concept) > 0.3:
                        relevant_nodes.append(node)
                        break

        return relevant_nodes

    def _perform_synthesis(self, nodes: list[KnowledgeNode], synthesis_type: str) -> dict[str, Any]:
        """Perform knowledge synthesis."""
        synthesis = {
            "type": synthesis_type,
            "nodes_count": len(nodes),
            "confidence": 0.7,
            "synthesis_result": "",
            "novel_insights": [],
            "connections_found": 0,
        }

        if synthesis_type == "comparative":
            synthesis[
                "synthesis_result"
            ] = "Comparative analysis reveals similarities and differences across knowledge domains"
            synthesis["novel_insights"] = [
                "Cross-domain patterns identified",
                "Methodological convergences observed",
                "Theoretical bridges established",
            ]

        elif synthesis_type == "integrative":
            synthesis["synthesis_result"] = "Integrative synthesis creates unified understanding"
            synthesis["novel_insights"] = [
                "Unified theoretical framework developed",
                "Contradictions resolved through higher-level synthesis",
                "Emergent properties identified",
            ]

        elif synthesis_type == "meta":
            synthesis["synthesis_result"] = "Meta-analysis reveals overarching patterns and principles"
            synthesis["novel_insights"] = [
                "Universal principles identified",
                "Boundary conditions clarified",
                "Predictive patterns discovered",
            ]

        # Count connections
        total_connections = sum(len(node.connections) for node in nodes)
        synthesis["connections_found"] = total_connections

        # Adjust confidence based on node quality and connections
        avg_confidence = sum(node.confidence for node in nodes) / len(nodes) if nodes else 0.5
        synthesis["confidence"] = min(0.95, avg_confidence + (total_connections * 0.01))

        return synthesis

    def _analyze_claim(self, claim: str, domain: str) -> dict[str, Any]:
        """Analyze and validate a knowledge claim."""
        # Search for relevant evidence in knowledge graph
        relevant_nodes = []
        for node in self.knowledge_graph.values():
            if self._calculate_concept_similarity(node.concept, claim) > 0.2:
                relevant_nodes.append(node)

        # Analyze evidence
        supporting_evidence = []
        contradicting_evidence = []

        for node in relevant_nodes:
            if node.confidence > 0.7:
                supporting_evidence.append(
                    {
                        "concept": node.concept,
                        "confidence": node.confidence,
                        "sources": len(node.sources),
                    }
                )
            elif node.confidence < 0.5:
                contradicting_evidence.append(
                    {
                        "concept": node.concept,
                        "confidence": node.confidence,
                        "sources": len(node.sources),
                    }
                )

        # Determine overall validity
        support_strength = sum(e["confidence"] for e in supporting_evidence)
        contradict_strength = sum(e["confidence"] for e in contradicting_evidence)

        if support_strength > contradict_strength * 2:
            validity = "likely_true"
            confidence = min(0.9, support_strength / (support_strength + contradict_strength))
        elif contradict_strength > support_strength * 2:
            validity = "likely_false"
            confidence = min(0.9, contradict_strength / (support_strength + contradict_strength))
        else:
            validity = "uncertain"
            confidence = 0.5

        return {
            "validity": validity,
            "confidence": confidence,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "uncertainty_factors": [
                "Limited available evidence",
                "Conflicting sources",
                "Domain complexity",
            ],
        }

    def _perform_expert_analysis(self, topic: str, domain: ResearchDomain, analysis_type: str) -> dict[str, Any]:
        """Perform expert-level analysis."""
        expertise = self.domain_expertise.get(domain, 0.7)

        analysis = {
            "analysis_type": analysis_type,
            "expertise_applied": expertise,
            "key_insights": [],
            "implications": [],
            "recommendations": [],
            "confidence": expertise,
        }

        # Generate insights based on domain expertise
        if domain == ResearchDomain.ARTIFICIAL_INTELLIGENCE:
            analysis["key_insights"] = [
                "AI system design requires careful consideration of alignment",
                "Scalability challenges emerge at deployment scale",
                "Human-AI collaboration patterns are crucial",
                "Safety measures must be built into the foundation",
            ]
        elif domain == ResearchDomain.MACHINE_LEARNING:
            analysis["key_insights"] = [
                "Model interpretability vs performance trade-offs exist",
                "Data quality fundamentally limits model performance",
                "Overfitting remains a persistent challenge",
                "Transfer learning enables efficient adaptation",
            ]
        else:
            analysis["key_insights"] = [
                "Interdisciplinary approaches yield novel solutions",
                "Methodological rigor is essential for validity",
                "Practical applications drive theoretical development",
                "Collaboration enhances research outcomes",
            ]

        # Generate implications
        analysis["implications"] = [
            "Research findings have broader applicability",
            "Policy considerations may be necessary",
            "Ethical review processes should be established",
            "Long-term monitoring is recommended",
        ]

        # Generate recommendations
        analysis["recommendations"] = [
            "Conduct follow-up studies to validate findings",
            "Develop practical implementation guidelines",
            "Establish interdisciplinary collaborations",
            "Create public engagement strategies",
        ]

        return analysis

    def _search_knowledge_graph(self, query: str, max_results: int, min_confidence: float) -> list[dict[str, Any]]:
        """Search the knowledge graph."""
        results = []

        for node in self.knowledge_graph.values():
            if node.confidence >= min_confidence:
                similarity = self._calculate_concept_similarity(node.concept, query)
                if similarity > 0.1:
                    results.append(
                        {
                            "node_id": node.node_id,
                            "concept": node.concept,
                            "domain": node.domain.value,
                            "confidence": node.confidence,
                            "similarity": similarity,
                            "connections": len(node.connections),
                            "last_updated": node.last_updated,
                        }
                    )

        # Sort by similarity and confidence
        results.sort(key=lambda x: (x["similarity"] * x["confidence"]), reverse=True)

        return results[:max_results]

    def _determine_methodology(self, question: str, domain: str, constraints: dict[str, Any]) -> dict[str, Any]:
        """Determine the best research methodology."""
        # Analyze question type
        if "compare" in question.lower() or "vs" in question.lower():
            methodology = "comparative_analysis"
        elif "effect" in question.lower() or "impact" in question.lower():
            methodology = "experimental_design"
        elif "trend" in question.lower() or "over time" in question.lower():
            methodology = "longitudinal_study"
        elif "relationship" in question.lower() or "correlation" in question.lower():
            methodology = "correlational_study"
        else:
            methodology = "systematic_review"

        methodologies = {
            "comparative_analysis": {
                "name": "Comparative Analysis",
                "rationale": "Best for comparing different approaches or systems",
                "steps": [
                    "Define comparison criteria",
                    "Gather data for each alternative",
                    "Analyze differences and similarities",
                    "Draw conclusions",
                ],
                "duration": "4-6 weeks",
                "resources": ["Access to comparison targets", "Analysis tools"],
            },
            "experimental_design": {
                "name": "Experimental Design",
                "rationale": "Best for establishing causal relationships",
                "steps": [
                    "Define hypothesis",
                    "Design controlled experiment",
                    "Collect data",
                    "Analyze results",
                    "Validate findings",
                ],
                "duration": "8-12 weeks",
                "resources": [
                    "Experimental setup",
                    "Control groups",
                    "Statistical tools",
                ],
            },
            "systematic_review": {
                "name": "Systematic Review",
                "rationale": "Best for comprehensive literature analysis",
                "steps": [
                    "Define search strategy",
                    "Screen and select studies",
                    "Extract and synthesize data",
                    "Assess quality",
                    "Draw conclusions",
                ],
                "duration": "6-8 weeks",
                "resources": [
                    "Database access",
                    "Review tools",
                    "Quality assessment criteria",
                ],
            },
        }

        return methodologies.get(methodology, methodologies["systematic_review"])

    def _generate_insights(self, request: dict[str, Any]) -> list[str]:
        """Generate insights for general requests."""
        topic = request.get("topic", request.get("task", "unknown"))

        insights = [
            f"Comprehensive analysis of {topic} reveals multiple perspectives",
            "Cross-domain connections provide valuable context",
            "Evidence-based approach ensures reliable conclusions",
            "Future research directions are clearly identified",
        ]

        return insights

    def _assess_confidence(self, request: dict[str, Any]) -> float:
        """Assess confidence level for a request."""
        domain = request.get("domain", "ai")
        complexity = request.get("complexity", "medium")

        base_confidence = (
            self.domain_expertise.get(ResearchDomain(domain), 0.7)
            if domain in [d.value for d in ResearchDomain]
            else 0.6
        )

        if complexity == "low":
            return min(0.95, base_confidence + 0.1)
        elif complexity == "high":
            return max(0.4, base_confidence - 0.2)
        else:
            return base_confidence

    def _initialize_domain_expertise(self) -> None:
        """Initialize domain expertise levels."""
        self.domain_expertise = {
            ResearchDomain.ARTIFICIAL_INTELLIGENCE: 0.9,
            ResearchDomain.MACHINE_LEARNING: 0.85,
            ResearchDomain.COMPUTER_SCIENCE: 0.8,
            ResearchDomain.MATHEMATICS: 0.75,
            ResearchDomain.PHYSICS: 0.7,
            ResearchDomain.PHILOSOPHY: 0.8,
            ResearchDomain.INTERDISCIPLINARY: 0.85,
            ResearchDomain.BIOLOGY: 0.6,
            ResearchDomain.ECONOMICS: 0.65,
            ResearchDomain.PSYCHOLOGY: 0.7,
        }

    def update_performance(self, performance_data: dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_history.append({**performance_data, "timestamp": time.time()})

        # Calculate KPIs
        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            success_rate = sum(1 for p in recent_performance if p.get("success", False)) / len(recent_performance)

            self.kpi_scores = {
                "research_accuracy": success_rate,
                "knowledge_synthesis_quality": self._calculate_synthesis_quality(),
                "domain_expertise_breadth": len(self.domain_expertise),
                "knowledge_graph_growth": len(self.knowledge_graph) / 100,  # Normalized
            }

    def _calculate_synthesis_quality(self) -> float:
        """Calculate quality of knowledge synthesis."""
        # Based on knowledge graph connectivity and confidence
        if not self.knowledge_graph:
            return 0.7

        total_confidence = sum(node.confidence for node in self.knowledge_graph.values())
        avg_confidence = total_confidence / len(self.knowledge_graph)

        total_connections = sum(len(node.connections) for node in self.knowledge_graph.values())
        connectivity = total_connections / len(self.knowledge_graph) if self.knowledge_graph else 0

        return min(1.0, (avg_confidence * 0.7) + (min(connectivity / 5, 1.0) * 0.3))

    def evaluate_kpi(self) -> dict[str, float]:
        """Evaluate current KPI metrics."""
        if not self.kpi_scores:
            return {
                "research_accuracy": 0.8,
                "knowledge_synthesis_quality": 0.75,
                "domain_expertise_breadth": 0.8,
                "knowledge_graph_growth": 0.7,
                "overall_performance": 0.76,
            }

        overall = sum(self.kpi_scores.values()) / len(self.kpi_scores)
        return {**self.kpi_scores, "overall_performance": overall}
