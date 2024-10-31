"""Research integration module for MAGI agent."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.utils.logging import setup_logger

logger = setup_logger(__name__)

class ResearchIntegration:
    """Integrates research findings and manages knowledge synthesis."""
    
    def __init__(self):
        self.research_cache = {}
        self.integration_metrics = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0
        }
    
    async def integrate_findings(
        self,
        findings: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate research findings into a cohesive knowledge structure.
        
        Args:
            findings: List of research findings to integrate
            context: Optional context for integration
            
        Returns:
            Integrated knowledge structure
        """
        try:
            logger.info(f"Integrating {len(findings)} research findings")
            
            # Process findings
            processed_findings = []
            for finding in findings:
                processed = await self._process_finding(finding)
                processed_findings.append(processed)
            
            # Synthesize knowledge
            synthesis = await self._synthesize_knowledge(processed_findings, context)
            
            # Update metrics
            self.integration_metrics['total_integrations'] += 1
            self.integration_metrics['successful_integrations'] += 1
            
            return {
                'status': 'success',
                'integrated_knowledge': synthesis,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'findings_processed': len(processed_findings),
                    'synthesis_quality': synthesis.get('quality', 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error integrating findings: {str(e)}")
            self.integration_metrics['failed_integrations'] += 1
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _process_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single research finding."""
        try:
            # Extract key information
            key_points = finding.get('key_points', [])
            evidence = finding.get('evidence', [])
            confidence = finding.get('confidence', 0.0)
            
            # Validate finding
            if not key_points or not evidence:
                raise ValueError("Finding must have key points and evidence")
            
            # Process and structure the finding
            processed = {
                'key_points': key_points,
                'evidence': evidence,
                'confidence': confidence,
                'processed_timestamp': datetime.now().isoformat(),
                'validation': {
                    'has_key_points': bool(key_points),
                    'has_evidence': bool(evidence),
                    'confidence_valid': 0 <= confidence <= 1
                }
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing finding: {str(e)}")
            raise
    
    async def _synthesize_knowledge(
        self,
        findings: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize processed findings into integrated knowledge."""
        try:
            # Group findings by topic/theme
            grouped_findings = self._group_findings(findings)
            
            # Synthesize each group
            synthesis = {
                'groups': {},
                'cross_connections': [],
                'quality': 0.0
            }
            
            for topic, group in grouped_findings.items():
                group_synthesis = await self._synthesize_group(group, context)
                synthesis['groups'][topic] = group_synthesis
            
            # Find cross-connections between groups
            synthesis['cross_connections'] = self._find_cross_connections(synthesis['groups'])
            
            # Calculate overall quality
            synthesis['quality'] = self._calculate_synthesis_quality(synthesis)
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Error synthesizing knowledge: {str(e)}")
            raise
    
    def _group_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group findings by topic/theme."""
        groups = {}
        for finding in findings:
            topic = self._determine_topic(finding)
            if topic not in groups:
                groups[topic] = []
            groups[topic].append(finding)
        return groups
    
    def _determine_topic(self, finding: Dict[str, Any]) -> str:
        """Determine the topic of a finding."""
        # For now, use a simple approach - could be enhanced with NLP
        if 'topic' in finding:
            return finding['topic']
        return 'general'
    
    def _find_cross_connections(self, groups: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find connections between different groups of findings."""
        connections = []
        processed_pairs = set()
        
        for topic1 in groups:
            for topic2 in groups:
                if topic1 != topic2 and (topic2, topic1) not in processed_pairs:
                    connection = self._analyze_connection(
                        topic1, groups[topic1],
                        topic2, groups[topic2]
                    )
                    if connection:
                        connections.append(connection)
                    processed_pairs.add((topic1, topic2))
        
        return connections
    
    def _analyze_connection(
        self,
        topic1: str,
        group1: Dict[str, Any],
        topic2: str,
        group2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze potential connection between two groups."""
        # Simple implementation - could be enhanced with more sophisticated analysis
        common_points = set(str(p) for p in group1.get('key_points', [])) & \
                       set(str(p) for p in group2.get('key_points', []))
        
        if common_points:
            return {
                'topics': [topic1, topic2],
                'common_points': list(common_points),
                'strength': len(common_points) / max(
                    len(group1.get('key_points', [])),
                    len(group2.get('key_points', []))
                )
            }
        return None
    
    def _calculate_synthesis_quality(self, synthesis: Dict[str, Any]) -> float:
        """Calculate the quality score of the synthesis."""
        # Consider multiple factors for quality
        factors = {
            'group_quality': self._calculate_group_quality(synthesis['groups']),
            'connection_quality': self._calculate_connection_quality(synthesis['cross_connections']),
            'coverage': self._calculate_coverage(synthesis)
        }
        
        # Weighted average of factors
        weights = {'group_quality': 0.4, 'connection_quality': 0.3, 'coverage': 0.3}
        quality = sum(score * weights[factor] for factor, score in factors.items())
        
        return min(1.0, max(0.0, quality))
    
    def _calculate_group_quality(self, groups: Dict[str, Any]) -> float:
        """Calculate quality score for groups."""
        if not groups:
            return 0.0
        
        scores = []
        for group in groups.values():
            # Consider factors like evidence strength, confidence
            evidence_strength = len(group.get('evidence', [])) / 10  # Normalize
            confidence = group.get('confidence', 0.0)
            scores.append((evidence_strength + confidence) / 2)
        
        return sum(scores) / len(scores)
    
    def _calculate_connection_quality(self, connections: List[Dict[str, Any]]) -> float:
        """Calculate quality score for cross-connections."""
        if not connections:
            return 0.0
        
        return sum(conn.get('strength', 0.0) for conn in connections) / len(connections)
    
    def _calculate_coverage(self, synthesis: Dict[str, Any]) -> float:
        """Calculate coverage score for the synthesis."""
        total_points = sum(
            len(group.get('key_points', [])) 
            for group in synthesis['groups'].values()
        )
        total_connections = len(synthesis['cross_connections'])
        
        # Consider both breadth (points) and depth (connections)
        points_score = min(1.0, total_points / 20)  # Normalize
        connections_score = min(1.0, total_connections / 10)  # Normalize
        
        return (points_score + connections_score) / 2
    
    async def synthesize_group(
        self,
        group: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize a group of related findings."""
        try:
            # Combine key points
            all_points = []
            all_evidence = []
            confidence_scores = []
            
            for finding in group:
                all_points.extend(finding.get('key_points', []))
                all_evidence.extend(finding.get('evidence', []))
                confidence_scores.append(finding.get('confidence', 0.0))
            
            # Remove duplicates while preserving order
            unique_points = list(dict.fromkeys(all_points))
            unique_evidence = list(dict.fromkeys(all_evidence))
            
            # Calculate aggregate confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                'key_points': unique_points,
                'evidence': unique_evidence,
                'confidence': avg_confidence,
                'context': context,
                'synthesis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing group: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        return {
            **self.integration_metrics,
            'success_rate': (
                self.integration_metrics['successful_integrations'] /
                self.integration_metrics['total_integrations']
                if self.integration_metrics['total_integrations'] > 0
                else 0.0
            )
        }
