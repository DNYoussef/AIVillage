import random
import json
from typing import List, Dict, Any
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig

from .technique_archive import PROMPT_TECHNIQUE_ARCHIVE

class ADASTask(Task):
    def __init__(self, task_description: str):
        config = ChatAgentConfig(
            name="ADAS",
            system_message="You are an expert machine learning researcher designing agentic systems.",
            llm=OpenAIGPTConfig(model="gpt-4")
        )
        agent = ChatAgent(config)
        super().__init__(agent)
        self.task_description = task_description
        self.archive = PROMPT_TECHNIQUE_ARCHIVE
        self.best_agent = None
        self.best_performance = float('-inf')

    def generate_prompt(self, archive: List[Dict]) -> str:
        archive_str = ",\n".join([str(agent) for agent in archive])
        
        return f"""
        Your objective is to create an optimal agent for the following task:

        {self.task_description}

        Here is the archive of discovered architectures:

        [{archive_str}]

        Design a new agent that improves upon these existing architectures. Be creative and think outside the box.
        Your response should be a JSON object with the following fields:
        - thought: Your reasoning behind the agent design
        - name: A name for your proposed agent architecture
        - code: The complete Python code for the forward() function of your agent

        Ensure your code uses the Langroid ChatAgent class and follows the correct structure.
        """

    def create_new_agent(self) -> Dict[str, Any]:
        prompt = self.generate_prompt(self.archive)
        response = self.agent.llm_response(prompt)
        return response.content

    def evaluate_agent(self, agent: Dict[str, Any]) -> float:
        """Simple evaluation of a generated agent.

        The ``agent`` argument can either be a dictionary or a JSON string
        representing an object with a ``code`` field.  We attempt to compile
        the code to make sure it is syntactically valid.  If compilation
        succeeds we return a deterministic score based on the number of lines in
        the code.  If it fails, ``0.0`` is returned.
        """

        if isinstance(agent, str):
            try:
                agent = json.loads(agent)
            except json.JSONDecodeError:
                return 0.0

        code = agent.get("code", "")

        try:
            compile(code, "<adas-agent>", "exec")
        except Exception:
            return 0.0

        # Use the length of the code as a very rough heuristic for performance
        # so that results are deterministic during testing.
        return float(len(code.splitlines()))

    async def run(self):
        num_iterations = 10  # You can adjust this or make it a parameter
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}")
            
            new_agent = self.create_new_agent()
            performance = self.evaluate_agent(new_agent)
            
            if performance > self.best_performance:
                self.best_agent = new_agent
                self.best_performance = performance
                print(f"New best agent found! Performance: {performance}")
            
            self.archive.append(new_agent)
        
        print(f"Evolution complete. Best agent performance: {self.best_performance}")
        return self.best_agent

class AgentTechnique(ToolMessage):
    request: str = "apply_technique"
    purpose: str = "Apply a specific AI technique"
    technique_name: str
    code: str

    def handle(self):
        # This method would be implemented to actually apply the technique
        pass

# Example usage
if __name__ == "__main__":
    task_description = "Design an agent that can solve abstract reasoning tasks in the ARC challenge."
    adas_task = ADASTask(task_description)
    best_agent = adas_task.run()
    
    print("Best Agent:")
    print(best_agent)
