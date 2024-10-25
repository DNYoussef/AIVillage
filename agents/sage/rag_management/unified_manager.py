"""Unified RAG management system."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
from rag_system.core.config import RAGConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.utils.standardized_formats import create_standardized_prompt, create_standardized_output, OutputFormat
import logging
import json

logger = logging.getLogger(__name__)

class UnifiedRAGManager:
    """
    Unified RAG management system that combines:
    - Health monitoring
    - Performance optimization
    - Data consistency
    - Resource management
    """
    
    def __init__(self, rag_config: RAGConfig, llm_config: OpenAIGPTConfig):
        self.rag_system = EnhancedRAGPipeline(rag_config)
        self.llm = llm_config.create()
        self.health_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}

    @error_handler.handle_error
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            health_check_prompt = create_standardized_prompt(
                task="health_check",
                context="RAG system diagnostics",
                instructions=[
                    "Analyze index health, performance metrics, and data consistency",
                    "For each aspect, provide a boolean indicating if it's healthy",
                    "Include severity levels for any issues",
                    "Provide detailed descriptions of problems",
                    "Return structured results"
                ]
            )
            
            health_check_result = await self.llm.complete(health_check_prompt)
            parsed_result = self._parse_json_response(health_check_result.text)
            
            # Record health check
            self._record_health_check(parsed_result)
            
            return {
                "issue_detected": not all([
                    parsed_result['index_health']['healthy'],
                    parsed_result['performance_metrics']['acceptable'],
                    parsed_result['data_consistency']['consistent']
                ]),
                **parsed_result
            }
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            raise AIVillageException(f"Error performing health check: {str(e)}") from e

    @error_handler.handle_error
    async def handle_health_issues(self, health_check_result: Dict[str, Any]):
        """Handle detected health issues."""
        try:
            # Generate handling plan
            handling_plan = await self._generate_handling_plan(health_check_result)
            
            # Execute plan
            await self._execute_handling_plan(handling_plan)
            
            # Notify administrators
            await self._notify_administrators(health_check_result, handling_plan)
            
            # Verify fixes
            verification_result = await self.perform_health_check()
            
            return {
                "original_issues": health_check_result,
                "handling_plan": handling_plan,
                "verification_result": verification_result
            }
            
        except Exception as e:
            logger.error(f"Error handling health issues: {str(e)}")
            raise AIVillageException(f"Error handling health issues: {str(e)}") from e

    async def _generate_handling_plan(self, health_check_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan to handle health issues."""
        handling_prompt = self.prompt_template.generate_prompt(
            task="generate_handling_plan",
            context=f"Health check result: {json.dumps(health_check_result, indent=2)}",
            instructions=[
                "For each issue, provide detailed resolution steps",
                "Include risk assessment and mitigation strategies",
                "Estimate required resources and timeline",
                "Prioritize critical issues",
                "Return structured plan"
            ]
        )
        
        response = await self.llm.complete(handling_prompt)
        return self._parse_json_response(response.text)

    async def _execute_handling_plan(self, plan: Dict[str, Any]):
        """Execute the handling plan."""
        try:
            # Handle index issues
            if plan.get('index_issues'):
                await self._handle_index_issues(plan['index_issues'])
            
            # Handle performance issues
            if plan.get('performance_issues'):
                await self._handle_performance_issues(plan['performance_issues'])
            
            # Handle consistency issues
            if plan.get('consistency_issues'):
                await self._handle_consistency_issues(plan['consistency_issues'])
                
        except Exception as e:
            logger.error(f"Error executing handling plan: {str(e)}")
            raise

    async def _handle_index_issues(self, issues: List[Dict[str, Any]]):
        """Handle index-related issues."""
        for issue in issues:
            severity = issue.get('severity', 'low')
            
            if severity == 'high':
                await self._rebuild_index()
            else:
                await self._repair_index()
                
            await self._optimize_index()

    async def _handle_performance_issues(self, issues: List[Dict[str, Any]]):
        """Handle performance-related issues."""
        for issue in issues:
            # Tune performance parameters
            await self._tune_performance()
            
            # Scale resources if needed
            if issue.get('requires_scaling', False):
                await self._scale_resources()

    async def _handle_consistency_issues(self, issues: List[Dict[str, Any]]):
        """Handle data consistency issues."""
        for issue in issues:
            # Reconcile inconsistent data
            await self._reconcile_data()
            
            # Validate data integrity
            await self._validate_data()

    async def _rebuild_index(self):
        """Rebuild the RAG index."""
        logger.info("Rebuilding RAG index")
        await self.rag_system.rebuild_index()

    async def _repair_index(self):
        """Repair the RAG index."""
        logger.info("Repairing RAG index")
        await self.rag_system.repair_index()

    async def _optimize_index(self):
        """Optimize the RAG index."""
        logger.info("Optimizing RAG index")
        await self.rag_system.optimize_index()

    async def _tune_performance(self):
        """Tune system performance parameters."""
        logger.info("Tuning RAG performance")
        current_params = await self.rag_system.get_performance_params()
        
        # Generate optimal parameters
        tuning_prompt = self.prompt_template.generate_prompt(
            task="optimize_performance",
            context=f"Current parameters: {json.dumps(current_params, indent=2)}",
            instructions=[
                "Analyze current parameter values",
                "Suggest optimal values for each parameter",
                "Explain reasoning for each suggestion",
                "Consider system constraints and trade-offs"
            ]
        )
        
        response = await self.llm.complete(tuning_prompt)
        optimal_params = self._parse_json_response(response.text)
        
        # Apply new parameters
        await self.rag_system.set_performance_params(optimal_params)

    async def _scale_resources(self):
        """Scale system resources."""
        logger.info("Scaling RAG system resources")
        current_resources = await self.rag_system.get_resource_allocation()
        
        # Generate scaling plan
        scaling_prompt = self.prompt_template.generate_prompt(
            task="scale_resources",
            context=f"Current allocation: {json.dumps(current_resources, indent=2)}",
            instructions=[
                "Analyze current resource usage",
                "Suggest optimal resource allocation",
                "Consider cost-effectiveness",
                "Include scaling strategy"
            ]
        )
        
        response = await self.llm.complete(scaling_prompt)
        optimal_resources = self._parse_json_response(response.text)
        
        # Apply new resource allocation
        await self.rag_system.set_resource_allocation(optimal_resources)

    async def _reconcile_data(self):
        """Reconcile inconsistent data."""
        logger.info("Reconciling RAG data")
        inconsistencies = await self.rag_system.find_data_inconsistencies()
        
        if inconsistencies:
            # Generate reconciliation steps
            reconciliation_prompt = self.prompt_template.generate_prompt(
                task="reconcile_data",
                context=f"Inconsistencies: {json.dumps(inconsistencies, indent=2)}",
                instructions=[
                    "Analyze each inconsistency",
                    "Provide step-by-step reconciliation process",
                    "Prioritize critical inconsistencies",
                    "Include validation steps"
                ]
            )
            
            response = await self.llm.complete(reconciliation_prompt)
            reconciliation_steps = self._parse_json_response(response.text)
            
            # Execute reconciliation
            for step in reconciliation_steps:
                await self.rag_system.execute_reconciliation_step(step)

    async def _validate_data(self):
        """Validate data integrity."""
        logger.info("Validating RAG data")
        validation_results = await self.rag_system.validate_data()
        
        if not validation_results['all_valid']:
            # Generate validation fixes
            validation_prompt = self.prompt_template.generate_prompt(
                task="validate_data",
                context=f"Validation results: {json.dumps(validation_results, indent=2)}",
                instructions=[
                    "Analyze each validation failure",
                    "Suggest specific fixes",
                    "Prioritize by importance",
                    "Include verification steps"
                ]
            )
            
            response = await self.llm.complete(validation_prompt)
            fix_instructions = self._parse_json_response(response.text)
            
            # Execute fixes
            for instruction in fix_instructions:
                await self.rag_system.execute_data_fix(instruction)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            raise AIVillageException("Failed to parse JSON response")

    def _record_health_check(self, result: Dict[str, Any]):
        """Record health check results."""
        self.health_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        # Keep only recent history
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]

    @property
    def system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "health_checks": len(self.health_history),
            "recent_issues": [h["result"] for h in self.health_history[-5:]],
            "performance_metrics": self.performance_metrics
        }

    async def introspect(self) -> Dict[str, Any]:
        """Get system introspection data."""
        return {
            "rag_system_config": self.rag_system.config.dict(),
            "system_stats": self.system_stats,
            "health_history": self.health_history[-5:],
            "performance_metrics": self.performance_metrics
        }
