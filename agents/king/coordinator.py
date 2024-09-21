import asyncio
from typing import Dict, Any, List
from ...communication.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from ...knowledge_base.rag_system import RAGSystem
from .unified_task_manager import UnifiedTaskManager
from .rag_management import KingRAGManagement
from .project_manager import ProjectManager
from ...utils.exceptions import AIVillageException
from ...utils.logger import logger

class KingCoordinator:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: RAGSystem):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.task_manager = UnifiedTaskManager()
        self.rag_management = KingRAGManagement(rag_system)
        self.project_manager = ProjectManager()

    async def run(self):
        try:
            await asyncio.gather(
                self.task_manager.monitor_tasks(),
                self.handle_messages(),
                self.manage_rag_system(),
                self.manage_community()
            )
        except Exception as e:
            logger.error(f"Error in King's run method: {str(e)}")
            raise AIVillageException(f"Error in King's run method: {str(e)}") from e

    async def handle_messages(self):
        while True:
            try:
                message = await self.communication_protocol.get_next_message()
                if message:
                    await self.process_message(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error handling messages: {str(e)}")

    async def process_message(self, message: Message):
        try:
            if message.type == MessageType.TASK:
                await self.handle_task_message(message)
            elif message.type == MessageType.QUERY:
                await self.handle_query_message(message)
            elif message.type == MessageType.COMMAND:
                await self.handle_command_message(message)
            else:
                logger.warning(f"Unhandled message type: {message.type}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    async def handle_task_message(self, message: Message):
        try:
            task = await self.task_manager.create_task(message.content['description'], message.content.get('assigned_agents', []))
            await self.task_manager.assign_task(task)
            response = Message(
                type=MessageType.RESPONSE,
                sender="King",
                receiver=message.sender,
                content={"task_id": task.id, "status": "assigned"},
                priority=Priority.MEDIUM
            )
            await self.communication_protocol.send_message(response)
        except Exception as e:
            logger.error(f"Error handling task message: {str(e)}")

    async def handle_query_message(self, message: Message):
        try:
            result = await self.rag_system.query(message.content['query'])
            response = Message(
                type=MessageType.RESPONSE,
                sender="King",
                receiver=message.sender,
                content={"response": result},
                priority=message.priority
            )
            await self.communication_protocol.send_message(response)
        except Exception as e:
            logger.error(f"Error handling query message: {str(e)}")

    async def handle_command_message(self, message: Message):
        try:
            command = message.content['command']
            if command == 'create_project':
                project_id = await self.project_manager.create_project(message.content['project_data'])
                response = {"project_id": project_id}
            elif command == 'get_project_status':
                status = await self.project_manager.get_project_status(message.content['project_id'])
                response = {"status": status}
            else:
                response = {"error": "Unknown command"}
            
            response_message = Message(
                type=MessageType.RESPONSE,
                sender="King",
                receiver=message.sender,
                content=response,
                priority=Priority.MEDIUM
            )
            await self.communication_protocol.send_message(response_message)
        except Exception as e:
            logger.error(f"Error handling command message: {str(e)}")

    async def manage_rag_system(self):
        while True:
            try:
                health_check_result = await self.rag_management.perform_health_check()
                if health_check_result['issue_detected']:
                    await self.rag_management.handle_rag_health_issue(health_check_result)
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error managing RAG system: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def manage_community(self):
        while True:
            try:
                projects = await self.project_manager.get_all_projects()
                for project_id, project_data in projects.items():
                    await self.manage_project(project_id, project_data)
                await asyncio.sleep(600)  # Check every 10 minutes
            except Exception as e:
                logger.error(f"Error managing community: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def manage_project(self, project_id: str, project_data: dict):
        try:
            for task in project_data.get('tasks', []):
                if task['status'] == 'unassigned':
                    await self.task_manager.assign_task(task)
            await self.project_manager.update_project_status(project_id)
        except Exception as e:
            logger.error(f"Error managing project {project_id}: {str(e)}")

    async def handle_task_result(self, agent_id: str, task_id: str, result: dict):
        try:
            await self.task_manager.complete_task(task_id, result)
            await self.rag_system.update(task_id, result)
            
            if result.get('needs_analysis', False):
                analysis = await self.perform_analysis(result['content'])
                await self.rag_system.update(f"{task_id}_analysis", analysis)
        except Exception as e:
            logger.error(f"Error handling task result: {str(e)}")

    async def perform_analysis(self, content: str) -> Dict[str, Any]:
        # This is a placeholder for the analysis logic
        # In a real implementation, this might involve calling a specialized agent or service
        return {"summary": "Analysis of content", "key_points": ["Point 1", "Point 2"]}