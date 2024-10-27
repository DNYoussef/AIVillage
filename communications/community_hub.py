from datetime import datetime
from .protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from agents.utils.exceptions import AIVillageException
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CommunityHub:
    """
    Central hub for agent community communication and resource sharing.
    
    Manages research results, agent registrations, and project coordination
    through a standardized communication protocol.
    """
    def __init__(self, communication_protocol: StandardCommunicationProtocol):
        self.research_results = {}
        self.agents = {}
        self.projects = {}
        self.communication_protocol = communication_protocol

    async def register_agent(self, agent_id: str, capabilities: Dict[str, Any]) -> None:
        """Register a new agent with the community hub."""
        if agent_id in self.agents:
            raise AIVillageException(f"Agent {agent_id} is already registered")
        self.agents[agent_id] = capabilities
        logger.info(f"Agent {agent_id} registered with capabilities: {capabilities}")

    async def submit_research(self, agent_id: str, research_id: str, results: Dict[str, Any]) -> None:
        """Submit research results to the community hub."""
        if agent_id not in self.agents:
            raise AIVillageException(f"Agent {agent_id} is not registered")
        
        self.research_results[research_id] = {
            'agent': agent_id,
            'results': results,
            'timestamp': datetime.now()
        }
        logger.info(f"Research {research_id} submitted by agent {agent_id}")

    async def get_research(self, research_id: str) -> Dict[str, Any]:
        """Retrieve research results by ID."""
        if research_id not in self.research_results:
            raise AIVillageException(f"Research {research_id} not found")
        return self.research_results[research_id]['results']

    async def create_project(self, project_id: str, details: Dict[str, Any]) -> None:
        """Create a new collaborative project."""
        if project_id in self.projects:
            raise AIVillageException(f"Project {project_id} already exists")
        
        self.projects[project_id] = {
            'details': details,
            'participants': set(),
            'status': 'created'
        }
        logger.info(f"Project {project_id} created with details: {details}")

    async def join_project(self, project_id: str, agent_id: str) -> None:
        """Add an agent to a project."""
        if project_id not in self.projects:
            raise AIVillageException(f"Project {project_id} not found")
        if agent_id not in self.agents:
            raise AIVillageException(f"Agent {agent_id} is not registered")
        
        self.projects[project_id]['participants'].add(agent_id)
        logger.info(f"Agent {agent_id} joined project {project_id}")

    async def broadcast_to_project(self, project_id: str, message: Dict[str, Any], 
                                 priority: Priority = Priority.MEDIUM) -> None:
        """Broadcast a message to all project participants."""
        if project_id not in self.projects:
            raise AIVillageException(f"Project {project_id} not found")
        
        participants = self.projects[project_id]['participants']
        for participant in participants:
            msg = Message(
                type=MessageType.NOTIFICATION,
                sender="community_hub",
                receiver=participant,
                content=message,
                priority=priority
            )
            await self.communication_protocol.send_message(msg)
        
        logger.info(f"Message broadcast to project {project_id} participants")

    def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get the current status of a project."""
        if project_id not in self.projects:
            raise AIVillageException(f"Project {project_id} not found")
        return self.projects[project_id]

    def list_projects(self) -> List[Dict[str, Any]]:
        """Get a list of all active projects."""
        return [
            {'id': pid, **pdata}
            for pid, pdata in self.projects.items()
        ]

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a registered agent."""
        if agent_id not in self.agents:
            raise AIVillageException(f"Agent {agent_id} not found")
        return self.agents[agent_id]

    def list_agents(self) -> List[Dict[str, Any]]:
        """Get a list of all registered agents."""
        return [
            {'id': aid, 'capabilities': capabilities}
            for aid, capabilities in self.agents.items()
        ]
