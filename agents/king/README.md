# King Agent

The King Agent is a sophisticated AI system designed to coordinate and manage multiple AI agents in the AI Village project. It uses advanced decision-making processes, task routing, and management techniques to efficiently handle complex tasks and workflows.

## Components

### 1. KingCoordinator

The KingCoordinator is the central component that manages the interaction between different parts of the system. It handles task messages, routes tasks to appropriate agents, and manages the overall workflow.

Key features:
- Task routing using the AgentRouter
- Decision making for complex tasks
- Task assignment and completion management
- Agent addition and removal
- Model saving and loading

### 2. UnifiedTaskManager

The UnifiedTaskManager is responsible for creating, assigning, and managing tasks across different agents. It also handles workflows and uses an incentive model to motivate agents.

Key features:
- Task creation and assignment
- Workflow management
- Incentive-based task allocation
- Task status tracking
- Performance monitoring

### 3. DecisionMaker

The DecisionMaker component is responsible for making complex decisions when the AgentRouter is uncertain about task allocation. It uses various AI techniques, including Monte Carlo Tree Search (MCTS), to optimize decision-making.

Key features:
- Problem analysis
- Alternative generation and evaluation
- MCTS-based workflow optimization
- Best agent suggestion for tasks

### 4. AgentRouter

The AgentRouter is responsible for efficiently routing tasks to the most appropriate agents based on their capabilities and past performance.

Key features:
- Task routing based on learned preferences
- Continuous learning from task results
- Confidence-based decision making

## Usage

To use the King Agent in your project:

1. Initialize the KingCoordinator with the necessary dependencies (communication protocol, RAG system, AI provider).
2. Set up your agents and register them with the KingCoordinator.
3. Send task messages to the KingCoordinator for processing.

Example:

```python
from agents.king.coordinator import KingCoordinator
from your_communication_protocol import CommunicationProtocol
from your_rag_system import RAGSystem
from your_ai_provider import AIProvider

# Initialize dependencies
comm_protocol = CommunicationProtocol()
rag_system = RAGSystem()
ai_provider = AIProvider()

# Create KingCoordinator
king = KingCoordinator(comm_protocol, rag_system, ai_provider)

# Register agents
king.add_agent("sage", SageAgent(comm_protocol))
king.add_agent("magi", MagiAgent(comm_protocol))

# Send a task message
task_message = Message(content={"description": "Analyze this dataset"})
await king.handle_task_message(task_message)
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
