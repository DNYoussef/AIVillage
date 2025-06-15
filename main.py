import asyncio
import logging
from typing import Dict, Any
from agents.utils.task import Task as LangroidTask
from agents.king.king_agent import KingAgent
from agents.sage.sage_agent import SageAgent
from agents.magi.magi_agent import MagiAgent
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore
from communications.protocol import StandardCommunicationProtocol
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIVillageSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = UnifiedConfig(**config)
        self.communication_protocol = StandardCommunicationProtocol()
        self.vector_store = VectorStore(self.config.vector_store_config)
        self.rag_pipeline = EnhancedRAGPipeline(self.config)
        
        self.king_agent = KingAgent(self.config, self.communication_protocol, self.vector_store)
        self.sage_agent = SageAgent(self.config, self.communication_protocol, self.vector_store)
        self.magi_agent = MagiAgent(self.config, self.communication_protocol, self.vector_store)
        
        self.agents = {
            "king": self.king_agent,
            "sage": self.sage_agent,
            "magi": self.magi_agent
        }

    @safe_execute
    async def initialize(self):
        logger.info("Initializing AI Village System")
        for agent_name, agent in self.agents.items():
            await agent.initialize()
        await self.king_agent.coordinator.update_agent_list()
        logger.info("AI Village System initialized successfully")

    @safe_execute
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Processing task: {task}")
        langroid_task = LangroidTask(
            self.king_agent,
            task.get("content"),
            task.get("id", ""),
            task.get("priority", 1),
        )
        langroid_task.type = task.get("type", "general")
        result = await self.king_agent.execute_task(langroid_task)
        logger.info(f"Task result: {result}")
        return result

    @safe_execute
    async def run_analytics(self):
        while True:
            analytics_report = self.king_agent.unified_analytics.generate_summary_report()
            logger.info(f"Analytics Report: {analytics_report}")
            await asyncio.sleep(3600)  # Run analytics every hour

    @safe_execute
    async def evolve_system(self):
        while True:
            logger.info("Evolving AI Village System")
            for agent in self.agents.values():
                await agent.evolve()
            logger.info("System evolution complete")
            await asyncio.sleep(86400)  # Evolve daily

    @safe_execute
    async def run(self):
        await self.initialize()
        analytics_task = asyncio.create_task(self.run_analytics())
        evolution_task = asyncio.create_task(self.evolve_system())
        
        while True:
            user_input = await self.get_user_input()
            if user_input.lower() == 'exit':
                break
            task = self.create_task_from_input(user_input)
            await self.process_task(task)

        analytics_task.cancel()
        evolution_task.cancel()
        logger.info("AI Village System shutting down")

    async def get_user_input(self) -> str:
        return await asyncio.to_thread(input, "Enter your task (or 'exit' to quit): ")

    def create_task_from_input(self, user_input: str) -> Dict[str, Any]:
        return {
            "type": "user_query",
            "content": user_input,
            "priority": 1
        }

@error_handler.handle_error
async def main():
    config = {
        "vector_store_config": {
            "engine": "faiss",
            "dimension": 768,
            "index_path": "path/to/faiss/index"
        },
        "rag_config": {
            "retriever_type": "hybrid",
            "reranker_model": "path/to/reranker/model",
            "top_k": 5
        },
        "llm_config": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 150
        }
    }
    
    ai_village = AIVillageSystem(config)
    await ai_village.run()

if __name__ == "__main__":
    asyncio.run(main())
