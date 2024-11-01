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
        logger.info("Initialized CommunityHub")
    
    async def initialize(self):
        """Initialize the community hub."""
        try:
            logger.info("Initializing CommunityHub...")
            
            # Initialize communication protocol if it has an initialize method
            if hasattr(self.communication_protocol, 'initialize'):
                await self.communication_protocol.initialize()
            
            # Initialize storage
            self.research_results.clear()
            self.agents.clear()
            self.projects.clear()
            
            logger.info("Successfully initialized CommunityHub")
            
        except Exception as e:
            logger.error(f"Error initializing CommunityHub: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the community hub."""
        try:
            logger.info("Shutting down CommunityHub...")
            
            # Notify all agents of shutdown
            for agent_id in self.agents:
                try:
                    await self.communication_protocol.send_message(
                        Message(
                            type=MessageType.SYSTEM,
                            sender="community_hub",
                            receiver=agent_id,
                            content={"action": "shutdown"},
                            priority=Priority.HIGH
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error notifying agent {agent_id} of shutdown: {str(e)}")
            
            # Shutdown communication protocol if it has a shutdown method
            if hasattr(self.communication_protocol, 'shutdown'):
                await self.communication_protocol.shutdown()
            
            # Clear storage
            self.research_results.clear()
            self.agents.clear()
            self.projects.clear()
            
            logger.info("Successfully shut down CommunityHub")
            
        except Exception as e:
            logger.error(f"Error shutting down CommunityHub: {str(e)}")
            raise

    async def register_agent(self, agent_id: str, capabilities: Dict[str, Any]) -> None:
        """Register a new agent with the community hub."""
        if agent_id in self.agents:
            raise AIVillageException(f"Agent {agent_id} is already registered")
        self.agents[agent_id] = capabilities
        
        # Notify other agents of new registration
        notification = Message(
            type=MessageType.NOTIFICATION,
            sender="community_hub",
            receiver="all",
            content={
                "event": "agent_registered",
                "agent_id": agent_id,
                "capabilities": capabilities
            },
            priority=Priority.LOW
        )
        await self.communication_protocol.send_message(notification)
        
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
        
        # Notify relevant agents of new research
        notification = Message(
            type=MessageType.NOTIFICATION,
            sender="community_hub",
            receiver="all",
            content={
                "event": "research_submitted",
                "research_id": research_id,
                "agent_id": agent_id,
                "summary": results.get("summary", "No summary provided")
            },
            priority=Priority.MEDIUM
        )
        await self.communication_protocol.send_message(notification)
        
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
            'status': 'created',
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        
        # Notify agents of new project
        notification = Message(
            type=MessageType.NOTIFICATION,
            sender="community_hub",
            receiver="all",
            content={
                "event": "project_created",
                "project_id": project_id,
                "details": details
            },
            priority=Priority.MEDIUM
        )
        await self.communication_protocol.send_message(notification)
        
        logger.info(f"Project {project_id} created with details: {details}")

    async def join_project(self, project_id: str, agent_id: str) -> None:
        """Add an agent to a project."""
        if project_id not in self.projects:
            raise AIVillageException(f"Project {project_id} not found")
        if agent_id not in self.agents:
            raise AIVillageException(f"Agent {agent_id} is not registered")
        
        self.projects[project_id]['participants'].add(agent_id)
        self.projects[project_id]['last_updated'] = datetime.now()
        
        # Notify project participants
        notification = Message(
            type=MessageType.NOTIFICATION,
            sender="community_hub",
            receiver="project_participants",
            content={
                "event": "agent_joined",
                "project_id": project_id,
                "agent_id": agent_id
            },
            priority=Priority.LOW
        )
        await self.broadcast_to_project(project_id, notification.content)
        
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
                content={
                    **message,
                    "project_id": project_id,
                    "timestamp": datetime.now().isoformat()
                },
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
            {
                'id': pid,
                'details': pdata['details'],
                'participant_count': len(pdata['participants']),
                'status': pdata['status'],
                'created_at': pdata['created_at'],
                'last_updated': pdata['last_updated']
            }
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
            {
                'id': aid,
                'capabilities': capabilities,
                'projects': [
                    pid for pid, pdata in self.projects.items()
                    if aid in pdata['participants']
                ]
            }
            for aid, capabilities in self.agents.items()
        ]
