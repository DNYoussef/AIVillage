# King Agent

The King Agent is a sophisticated AI system designed to coordinate and manage multiple AI agents in the AI Village project. It uses advanced decision-making processes, task routing, and management techniques to efficiently handle complex tasks and workflows.

## System Architecture

The King Agent system consists of several interconnected components that work together to manage tasks, make decisions, and coordinate multiple AI agents. Here's an overview of the main components:

### 1. KingAgent

The KingAgent is the central class that integrates all other components. It serves as the main interface for interacting with the King Agent system.

Key features:
- Initialization of all sub-components
- High-level task execution and management
- Integration with the RAG (Retrieval-Augmented Generation) system
- Model saving and loading

### 2. KingCoordinator

The KingCoordinator manages the interaction between different parts of the system. It handles task messages, routes tasks to appropriate agents, and manages the overall workflow.

Key features:
- Task routing using the AgentRouter
- Decision making for complex tasks
- Task assignment and completion management
- Agent addition and removal
- Integration with other components

### 3. UnifiedTaskManager

The UnifiedTaskManager is responsible for creating, assigning, and managing tasks across different agents. It also handles workflows and uses an incentive model to motivate agents.

Key features:
- Task creation and assignment
- Workflow management
- Incentive-based task allocation
- Task status tracking
- Performance monitoring

### 4. DecisionMaker

The DecisionMaker component is responsible for making complex decisions when the AgentRouter is uncertain about task allocation. It uses various AI techniques, including Monte Carlo Tree Search (MCTS), to optimize decision-making.

Key features:
- Problem analysis
- Alternative generation and evaluation
- MCTS-based workflow optimization
- Best agent suggestion for tasks

### 5. ProblemAnalyzer

The ProblemAnalyzer component is responsible for analyzing tasks and generating comprehensive problem analyses. It collaborates with other agents to create a well-rounded understanding of the problem at hand.

Key features:
- Collection of agent analyses
- Critique and revision of analyses
- Integration with the SEALEnhancedPlanGenerator
- Creation of final, synthesized analyses

### 6. AgentRouter

The AgentRouter is responsible for efficiently routing tasks to the most appropriate agents based on their capabilities and past performance.

Key features:
- Task routing based on learned preferences
- Continuous learning from task results
- Confidence-based decision making

### 7. IncentiveModel

The IncentiveModel is used by the UnifiedTaskManager to calculate incentives for agents based on their performance and task difficulty.

Key features:
- Dynamic incentive calculation
- Performance-based agent motivation
- Adaptive task allocation

## Usage

To use the King Agent in your project:

1. Initialize the KingAgent with the necessary dependencies (communication protocol, RAG system).
2. Set up your agents and register them with the KingCoordinator.
3. Send task messages to the KingAgent for processing.

Example:

```python
from agents.king.king_agent import KingAgent, KingAgentConfig
from your_communication_protocol import CommunicationProtocol
from your_rag_system import RAGSystem

# Initialize dependencies
comm_protocol = CommunicationProtocol()
rag_system = RAGSystem()

# Create KingAgent
config = KingAgentConfig(name="KingAgent", description="Main coordinator for AI Village", model="gpt-4")
king_agent = KingAgent(config, comm_protocol, rag_system)

# Register agents
await king_agent.coordinator.add_agent("sage", SageAgent(comm_protocol))
await king_agent.coordinator.add_agent("magi", MagiAgent(comm_protocol))

# Send a task message
task_message = Message(content={"description": "Analyze this dataset"})
result = await king_agent.execute_task(task_message)
```

## Testing

The King Agent comes with a comprehensive test suite, including unit tests for individual components and integration tests. To run the tests:

```
pytest agents/king/tests/
```

## Extending the King Agent

To add new functionality or agents to the King Agent:

1. Implement new agent classes and register them with the KingCoordinator.
2. Extend the DecisionMaker or AgentRouter if new decision-making or routing capabilities are needed.
3. Update the UnifiedTaskManager if new task types or workflow patterns are required.
4. Always update the test suite to cover new functionality.

## Contributing

Contributions to the King Agent are welcome. Please ensure that your code adheres to the project's coding standards and is accompanied by appropriate tests and documentation.

## Future Improvements

- Implement more advanced MCTS algorithms for better decision-making
- Enhance the RAG system integration for improved information retrieval
- Develop more sophisticated incentive models for agent motivation
- Implement advanced error handling and recovery mechanisms
- Enhance the system's ability to explain its decision-making process
