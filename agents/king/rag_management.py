from typing import Dict, Any
from ...utils.exceptions import AIVillageException
from ...utils.prompt_template import PromptTemplate
from ...utils.ai_provider import AIProvider
from ...knowledge_base.rag_system import RAGSystem
import logging

logger = logging.getLogger(__name__)

class KingRAGManagement:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.ai_provider = AIProvider()
        self.prompt_template = PromptTemplate()

    async def perform_health_check(self) -> Dict[str, Any]:
        try:
            health_check_prompt = self.prompt_template.generate_prompt(
                task="Perform a comprehensive health check on the RAG system",
                context="You are an AI assistant specialized in RAG system diagnostics",
                instructions=[
                    "Analyze index health, performance metrics, and data consistency",
                    "For each aspect, provide a boolean indicating if it's healthy/acceptable/consistent",
                    "Include a severity level (low, medium, high) if there are issues",
                    "Provide a brief description of any detected problems",
                    "Return the results in a structured format"
                ]
            )
            health_check_result = await self.ai_provider.generate_structured_response(health_check_prompt)

            return {
                "issue_detected": not all([health_check_result['index_health']['healthy'],
                                           health_check_result['performance_metrics']['acceptable'],
                                           health_check_result['data_consistency']['consistent']]),
                **health_check_result
            }
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            raise AIVillageException(f"Error performing health check: {str(e)}") from e

    async def handle_rag_health_issue(self, health_check_result: Dict[str, Any]):
        try:
            handling_prompt = self.prompt_template.generate_prompt(
                task="Suggest appropriate actions to address RAG system issues",
                context=f"Health check result: {health_check_result}",
                instructions=[
                    "For each issue, provide a detailed step-by-step plan to resolve it",
                    "Include potential risks and mitigation strategies",
                    "Estimate time and resources required for each plan",
                    "Return the results in a structured format"
                ]
            )
            handling_plan = await self.ai_provider.generate_structured_response(handling_prompt)

            for issue, plan in handling_plan.items():
                if issue == 'index_health' and not health_check_result['index_health']['healthy']:
                    await self._handle_index_issue(plan)
                elif issue == 'performance_metrics' and not health_check_result['performance_metrics']['acceptable']:
                    await self._handle_performance_issue(plan)
                elif issue == 'data_consistency' and not health_check_result['data_consistency']['consistent']:
                    await self._handle_consistency_issue(plan)

            await self._notify_administrators(health_check_result, handling_plan)
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
        notification_prompt = self.prompt_template.generate_prompt(
            task="Create a notification for administrators about RAG system issues",
            context=f"Health Check Result: {health_check_result}\nHandling Plan: {handling_plan}",
            instructions=[
                "Summarize detected issues",
                "List key actions being taken",
                "Specify any immediate actions required from administrators",
                "Provide an estimated timeline for resolution",
                "Format the notification in a clear, easy-to-read structure"
            ]
        )
        notification = await self.ai_provider.generate_text(notification_prompt)
        
        logger.info(f"Notifying administrators about RAG health issues:\n{notification}")
        # In a real-world scenario, you would send this notification via email, Slack, etc.

    async def _rebuild_index(self):
        logger.info("Rebuilding RAG index")
        try:
            await self.rag_system.rebuild_index()
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise AIVillageException(f"Error rebuilding index: {str(e)}") from e

    async def _repair_index(self):
        logger.info("Repairing RAG index")
        try:
            await self.rag_system.repair_index()
        except Exception as e:
            logger.error(f"Error repairing index: {str(e)}")
            raise AIVillageException(f"Error repairing index: {str(e)}") from e

    async def _optimize_index(self):
        logger.info("Optimizing RAG index")
        try:
            await self.rag_system.optimize_index()
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
            raise AIVillageException(f"Error optimizing index: {str(e)}") from e

    async def _tune_performance(self):
        logger.info("Tuning RAG performance")
        try:
            current_params = await self.rag_system.get_performance_params()
            tuning_prompt = self.prompt_template.generate_prompt(
                task="Suggest optimal performance parameters for the RAG system",
                context=f"Current parameters: {current_params}",
                instructions=[
                    "Analyze the current parameters",
                    "Suggest improvements for each parameter",
                    "Explain the reasoning behind each suggestion",
                    "Return the results as a dictionary of parameter-value pairs"
                ]
            )
            optimal_params = await self.ai_provider.generate_structured_response(tuning_prompt)
            await self.rag_system.set_performance_params(optimal_params)
        except Exception as e:
            logger.error(f"Error tuning performance: {str(e)}")
            raise AIVillageException(f"Error tuning performance: {str(e)}") from e

    async def _scale_resources(self):
        logger.info("Scaling RAG system resources")
        try:
            current_resources = await self.rag_system.get_resource_allocation()
            scaling_prompt = self.prompt_template.generate_prompt(
                task="Suggest optimal resource allocation for the RAG system",
                context=f"Current resource allocation: {current_resources}",
                instructions=[
                    "Analyze the current resource allocation",
                    "Suggest improvements for each resource type",
                    "Explain the reasoning behind each suggestion",
                    "Return the results as a dictionary of resource-value pairs"
                ]
            )
            optimal_resources = await self.ai_provider.generate_structured_response(scaling_prompt)
            await self.rag_system.set_resource_allocation(optimal_resources)
        except Exception as e:
            logger.error(f"Error scaling resources: {str(e)}")
            raise AIVillageException(f"Error scaling resources: {str(e)}") from e

    async def _reconcile_data(self):
        logger.info("Reconciling RAG data")
        try:
            inconsistencies = await self.rag_system.find_data_inconsistencies()
            if inconsistencies:
                reconciliation_prompt = self.prompt_template.generate_prompt(
                    task="Suggest data reconciliation steps for the RAG system",
                    context=f"Data inconsistencies: {inconsistencies}",
                    instructions=[
                        "Analyze each inconsistency",
                        "Suggest a step-by-step reconciliation process for each",
                        "Prioritize the reconciliation steps",
                        "Return the results as a list of ordered reconciliation steps"
                    ]
                )
                reconciliation_steps = await self.ai_provider.generate_structured_response(reconciliation_prompt)
                for step in reconciliation_steps:
                    await self.rag_system.execute_reconciliation_step(step)
        except Exception as e:
            logger.error(f"Error reconciling data: {str(e)}")
            raise AIVillageException(f"Error reconciling data: {str(e)}") from e

    async def _validate_data(self):
        logger.info("Validating RAG data")
        try:
            validation_results = await self.rag_system.validate_data()
            if not validation_results['all_valid']:
                validation_prompt = self.prompt_template.generate_prompt(
                    task="Suggest data validation fixes for the RAG system",
                    context=f"Validation results: {validation_results}",
                    instructions=[
                        "Analyze each validation failure",
                        "Suggest a fix for each invalid data point",
                        "Prioritize the fixes based on importance and impact",
                        "Return the results as a list of ordered fix instructions"
                    ]
                )
                fix_instructions = await self.ai_provider.generate_structured_response(validation_prompt)
                for instruction in fix_instructions:
                    await self.rag_system.execute_data_fix(instruction)
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise AIVillageException(f"Error validating data: {str(e)}") from e