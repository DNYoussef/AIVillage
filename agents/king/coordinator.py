...
from ..sage.sage_agent import SageAgent
from ..magi.magi_agent import MagiAgent

class KingCoordinator:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: RAGSystem):
        ...
        self.sage_agent = SageAgent(communication_protocol)  
        self.magi_agent = MagiAgent(communication_protocol)
    ...
    
    async def handle_task_message(self, message: Message):
        if "information" in message.content['description'].lower():
            await self.assign_task_to_sage(message)
        elif "code" in message.content['description'].lower(): 
            await self.assign_task_to_magi(message)
        else:
            decision_result = await self.decision_maker.make_decision(message.content['description'])
            await self._implement_decision(decision_result)

    async def assign_task_to_sage(self, message: Message):
        task = await self.task_manager.create_task(
            message.content['description'], 
            assigned_agents=[self.sage_agent.name]
        )
        await self.task_manager.assign_task(task)

    async def assign_task_to_magi(self, message: Message):  
        task = await self.task_manager.create_task(
            message.content['description'],
            assigned_agents=[self.magi_agent.name]  
        )
        await self.task_manager.assign_task(task)
