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
        self.decision_maker = DecisionMaker(communication_protocol, rag_system)

    async def create_final_analysis(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Prepare the prompt for the AI provider
            prompt = self._prepare_final_analysis_prompt(revised_analyses, rag_info)
            
            # Generate the final analysis using the AI provider
            final_analysis = await self.ai_provider.generate_structured_response(prompt)
            
            # Validate and refine the final analysis
            refined_analysis = await self._refine_final_analysis(final_analysis, revised_analyses, rag_info)
            
            return refined_analysis

        except Exception as e:
            raise AIVillageException(f"Error in creating final analysis: {str(e)}")

    def _prepare_final_analysis_prompt(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> str:
        analyses_summary = "\n".join([f"Agent {analysis['agent']}: {analysis['revised_analysis']}" for analysis in revised_analyses])
        
        prompt = f"""
        As the King agent, your task is to create a final, comprehensive analysis of the problem based on the following inputs:

        1. Revised analyses from all agents:
        {analyses_summary}

        2. Relevant information from the RAG system:
        {rag_info}

        Please synthesize this information into a cohesive final analysis. Your analysis should:
        1. Identify the core problem and its root causes
        2. Summarize the key insights from the agents' analyses
        3. Highlight any conflicting viewpoints and resolve them if possible
        4. Incorporate relevant historical context from the RAG system
        5. Identify any potential blind spots or areas requiring further investigation

        Provide your final analysis in a structured format with the following sections:
        - Problem Statement
        - Root Causes
        - Key Insights
        - Conflicting Viewpoints and Resolutions
        - Historical Context
        - Potential Blind Spots
        - Recommendations for Further Investigation

        Output your analysis as a JSON object with these sections as keys.
        """
        
        return prompt

    async def _refine_final_analysis(self, final_analysis: Dict[str, Any], revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        # Validate that all required sections are present
        required_sections = [
            "Problem Statement", "Root Causes", "Key Insights", "Conflicting Viewpoints and Resolutions",
            "Historical Context", "Potential Blind Spots", "Recommendations for Further Investigation"
        ]
        
        missing_sections = [section for section in required_sections if section not in final_analysis]
        
        if missing_sections:
            # If any sections are missing, ask the AI to fill them in
            fill_in_prompt = f"""
            The final analysis is missing the following sections: {', '.join(missing_sections)}
            Please provide content for these missing sections based on the revised analyses and RAG info.
            Output the missing sections as a JSON object with the missing section names as keys.
            """
            missing_content = await self.ai_provider.generate_structured_response(fill_in_prompt)
            final_analysis.update(missing_content)
        
        # Check for consistency and completeness
        consistency_prompt = f"""
        Review the following final analysis for consistency and completeness:
        {final_analysis}

        Considering the original revised analyses and RAG info, are there any inconsistencies or important points missing?
        If so, provide suggestions for improvements. If not, respond with "No improvements needed."

        Output your response as a JSON object with keys "inconsistencies" and "missing_points", each containing a list of strings.
        If there are no issues, these lists should be empty.
        """
        
        consistency_check = await self.ai_provider.generate_structured_response(consistency_prompt)
        
        if consistency_check["inconsistencies"] or consistency_check["missing_points"]:
            # If there are issues, ask the AI to resolve them
            resolution_prompt = f"""
            The following inconsistencies and missing points have been identified in the final analysis:
            Inconsistencies: {consistency_check["inconsistencies"]}
            Missing Points: {consistency_check["missing_points"]}

            Please update the final analysis to address these issues. Output the entire updated analysis as a JSON object.
            """
            final_analysis = await self.ai_provider.generate_structured_response(resolution_prompt)
        
        return final_analysis

    async def handle_task_message(self, message: Message):
        decision_result = await self.decision_maker.make_decision(message.content['description'])
        await self._implement_decision(decision_result)

    async def _implement_decision(self, decision_result: Dict[str, Any]):
        # Implementation of decision implementation
        pass

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

