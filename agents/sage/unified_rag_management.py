from typing import Dict, Any
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
from rag_system.core.config import RAGConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
import logging

logger = logging.getLogger(__name__)

class UnifiedRAGManagement:
    def __init__(self, rag_config: RAGConfig, llm_config: OpenAIGPTConfig, knowledge_tracker: UnifiedKnowledgeTracker | None = None):
        self.rag_system = EnhancedRAGPipeline(rag_config, knowledge_tracker)
        self.llm = llm_config.create()

    @error_handler.handle_error
    async def perform_health_check(self) -> Dict[str, Any]:
        try:
            health_check_prompt = """
            Perform a comprehensive health check on the RAG system. Analyze:
            1. Index health
            2. Performance metrics
            3. Data consistency
            
            For each aspect, provide:
            - A boolean indicating if it's healthy/acceptable/consistent
            - A severity level (low, medium, high) if there are issues
            - A brief description of any detected problems
            
            Return the results in a structured JSON format.
            """
            health_check_result = await self.llm.complete(health_check_prompt)
            parsed_result = self._parse_json_response(health_check_result.text)

            return {
                "issue_detected": not all([parsed_result['index_health']['healthy'],
                                           parsed_result['performance_metrics']['acceptable'],
                                           parsed_result['data_consistency']['consistent']]),
                **parsed_result
            }
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            raise AIVillageException(f"Error performing health check: {str(e)}") from e

    @error_handler.handle_error
    async def handle_rag_health_issue(self, health_check_result: Dict[str, Any]):
        try:
            handling_prompt = f"""
            Given the following health check result: {health_check_result}
            
            Suggest appropriate actions to address RAG system issues. For each issue:
            1. Provide a detailed step-by-step plan to resolve it
            2. Include potential risks and mitigation strategies
            3. Estimate time and resources required for each plan
            
            Return the results in a structured JSON format.
            """
            handling_plan = await self.llm.complete(handling_prompt)
            parsed_plan = self._parse_json_response(handling_plan.text)

            for issue, plan in parsed_plan.items():
                if issue == 'index_health' and not health_check_result['index_health']['healthy']:
                    await self._handle_index_issue(plan)
                elif issue == 'performance_metrics' and not health_check_result['performance_metrics']['acceptable']:
                    await self._handle_performance_issue(plan)
                elif issue == 'data_consistency' and not health_check_result['data_consistency']['consistent']:
                    await self._handle_consistency_issue(plan)

            await self._notify_administrators(health_check_result, parsed_plan)
        except Exception as e:
            logger.error(f"Error handling RAG health issue: {str(e)}")
            raise AIVillageException(f"Error handling RAG health issue: {str(e)}") from e

    async def _handle_index_issue(self, plan: Dict[str, Any]):
        logger.info(f"Handling index issue with plan: {plan}")
        if plan['severity'] == 'high':
            await self._rebuild_index()
        else:
            await self._repair_index()
        await self._optimize_index()

    async def _handle_performance_issue(self, plan: Dict[str, Any]):
        logger.info(f"Handling performance issue with plan: {plan}")
        await self._tune_performance()
        if plan.get('scale_resources', False):
            await self._scale_resources()

    async def _handle_consistency_issue(self, plan: Dict[str, Any]):
        logger.info(f"Handling consistency issue with plan: {plan}")
        await self._reconcile_data()
        await self._validate_data()

    async def _notify_administrators(self, health_check_result: Dict[str, Any], handling_plan: Dict[str, Any]):
        notification_prompt = f"""
        Create a notification for administrators about RAG system issues.
        Health Check Result: {health_check_result}
        Handling Plan: {handling_plan}

        Include:
        1. Summary of detected issues
        2. Key actions being taken
        3. Any immediate actions required from administrators
        4. Estimated timeline for resolution

        Format the notification in a clear, easy-to-read structure.
        """
        notification = await self.llm.complete(notification_prompt)
        logger.info(f"Notifying administrators about RAG health issues:\n{notification.text}")

    async def _rebuild_index(self):
        logger.info("Rebuilding RAG index")
        await self.rag_system.rebuild_index()

    async def _repair_index(self):
        logger.info("Repairing RAG index")
        await self.rag_system.repair_index()

    async def _optimize_index(self):
        logger.info("Optimizing RAG index")
        await self.rag_system.optimize_index()

    async def _tune_performance(self):
        logger.info("Tuning RAG performance")
        current_params = await self.rag_system.get_performance_params()
        tuning_prompt = f"""
        Suggest optimal performance parameters for the RAG system.
        Current parameters: {current_params}

        For each parameter:
        1. Analyze its current value
        2. Suggest an improvement
        3. Explain the reasoning behind the suggestion

        Return the results as a JSON dictionary of parameter-value pairs.
        """
        optimal_params = await self.llm.complete(tuning_prompt)
        parsed_params = self._parse_json_response(optimal_params.text)
        await self.rag_system.set_performance_params(parsed_params)

    async def _scale_resources(self):
        logger.info("Scaling RAG system resources")
        current_resources = await self.rag_system.get_resource_allocation()
        scaling_prompt = f"""
        Suggest optimal resource allocation for the RAG system.
        Current resource allocation: {current_resources}

        For each resource type:
        1. Analyze its current allocation
        2. Suggest an improvement
        3. Explain the reasoning behind the suggestion

        Return the results as a JSON dictionary of resource-value pairs.
        """
        optimal_resources = await self.llm.complete(scaling_prompt)
        parsed_resources = self._parse_json_response(optimal_resources.text)
        await self.rag_system.set_resource_allocation(parsed_resources)

    async def _reconcile_data(self):
        logger.info("Reconciling RAG data")
        inconsistencies = await self.rag_system.find_data_inconsistencies()
        if inconsistencies:
            reconciliation_prompt = f"""
            Suggest data reconciliation steps for the RAG system.
            Data inconsistencies: {inconsistencies}

            For each inconsistency:
            1. Analyze the issue
            2. Suggest a step-by-step reconciliation process
            3. Prioritize the reconciliation steps

            Return the results as a JSON list of ordered reconciliation steps.
            """
            reconciliation_steps = await self.llm.complete(reconciliation_prompt)
            parsed_steps = self._parse_json_response(reconciliation_steps.text)
            for step in parsed_steps:
                await self.rag_system.execute_reconciliation_step(step)

    async def _validate_data(self):
        logger.info("Validating RAG data")
        validation_results = await self.rag_system.validate_data()
        if not validation_results['all_valid']:
            validation_prompt = f"""
            Suggest data validation fixes for the RAG system.
            Validation results: {validation_results}

            For each validation failure:
            1. Analyze the issue
            2. Suggest a fix
            3. Prioritize the fixes based on importance and impact

            Return the results as a JSON list of ordered fix instructions.
            """
            fix_instructions = await self.llm.complete(validation_prompt)
            parsed_instructions = self._parse_json_response(fix_instructions.text)
            for instruction in parsed_instructions:
                await self.rag_system.execute_data_fix(instruction)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            raise AIVillageException("Failed to parse JSON response")

    @safe_execute
    async def process_query(self, query: str) -> Dict[str, Any]:
        return await self.rag_system.process_query(query)

    async def update_knowledge_base(self, new_data: Dict[str, Any]):
        await self.rag_system.update_knowledge_base(new_data)

    async def get_system_stats(self) -> Dict[str, Any]:
        return await self.rag_system.get_system_stats()

    async def introspect(self) -> Dict[str, Any]:
        return {
            "rag_system_config": self.rag_system.config.dict(),
            "system_stats": await self.get_system_stats(),
        }
