import logging
from typing import Dict, Any
from .route_llm import RouteLLM
from ..utils.exceptions import AIVillageException

logger = logging.getLogger(__name__)

class Router:
    def __init__(self):
        self.route_llm = RouteLLM()

    async def route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Routing task: {task}")
            routed_task = await self.route_llm.route_task(task)
            return routed_task
        except Exception as e:
            logger.error(f"Error routing task: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error routing task: {str(e)}")

    async def update_model(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating router model with task result: {result}")
            await self.route_llm.update_model(task, result)
        except Exception as e:
            logger.error(f"Error updating router model: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating router model: {str(e)}")

    async def save_models(self, path: str):
        try:
            logger.info(f"Saving router models to {path}")
            await self.route_llm.save_model(path)
        except Exception as e:
            logger.error(f"Error saving router models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error saving router models: {str(e)}")

    async def load_models(self, path: str):
        try:
            logger.info(f"Loading router models from {path}")
            await self.route_llm.load_model(path)
        except Exception as e:
            logger.error(f"Error loading router models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error loading router models: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "type": "Router",
            "description": "Routes tasks to appropriate agents or components",
            "route_llm_info": str(self.route_llm)
        }
