import asyncio
import logging
from agents.king.king_agent import KingAgent, KingAgentConfig
from agents.communication.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.pipeline import EnhancedRAGPipeline as RAGSystem
from agents.sage.sage_agent import SageAgent
from agents.magi.magi_agent import MagiAgent
from agents.utils.exceptions import AIVillageException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_demo():
    try:
        # Initialize components
        logger.info("Initializing components...")
        communication_protocol = StandardCommunicationProtocol()
        rag_system = RAGSystem()
        
        # Create KingAgent
        logger.info("Creating KingAgent...")
        config = KingAgentConfig(name="KingAgent", description="Main coordinator for AI Village", model="gpt-4")
        king_agent = KingAgent(config, communication_protocol, rag_system)

        # Create and register other agents
        logger.info("Creating and registering other agents...")
        sage_agent = SageAgent(communication_protocol)
        magi_agent = MagiAgent(communication_protocol)
        
        await king_agent.coordinator.add_agent("sage", sage_agent)
        await king_agent.coordinator.add_agent("magi", magi_agent)

        # Example tasks
        tasks = [
            Message(type=MessageType.TASK, sender="User", receiver="KingAgent", content={"description": "Analyze the impact of AI on job markets"}),
            Message(type=MessageType.TASK, sender="User", receiver="KingAgent", content={"description": "Develop a simple machine learning model for sentiment analysis"}),
            Message(type=MessageType.TASK, sender="User", receiver="KingAgent", content={"description": "Summarize recent advancements in quantum computing"}),
        ]

        # Process tasks
        for task in tasks:
            try:
                logger.info(f"Processing task: {task.content['description']}")
                result = await king_agent.execute_task(task)
                logger.info(f"Task result: {result}")
            except AIVillageException as e:
                logger.error(f"Error processing task: {str(e)}")
            except Exception as e:
                logger.exception(f"Unexpected error processing task: {str(e)}")

        # Demonstrate introspection
        try:
            logger.info("Performing introspection...")
            introspection = await king_agent.introspect()
            logger.info("King Agent Introspection:")
            logger.info(introspection)
        except Exception as e:
            logger.exception(f"Error during introspection: {str(e)}")

    except Exception as e:
        logger.exception(f"Fatal error in run_demo: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {str(e)}")
