# MagiPlanning

MagiPlanning is a sophisticated planning and problem-solving class that integrates various advanced reasoning techniques to generate, optimize, and execute plans. It is designed to handle complex problems by leveraging multiple AI-powered approaches.

## Features

- Tree-of-Thoughts reasoning
- Chain-of-Thought problem-solving
- Program-of-Thoughts for code generation
- Least-to-Most Prompting for breaking down complex problems
- Contrastive Chain-of-Thought for comparative reasoning
- Dynamic technique selection based on task characteristics and past performance
- Antifragility assessment and improvement
- Premortem analysis for risk identification and mitigation
- Xanatos Gambits for creating plans that benefit from both success and failure
- SWOT analysis
- Plan visualization
- Performance tracking and caching for efficiency

## Usage

Here's a basic example of how to use the MagiPlanning class:

```python
from agents.magi.magi_planning import MagiPlanning
from agents.quality_assurance_layer import QualityAssuranceLayer
from communications.protocol import StandardCommunicationProtocol
from agents.magi.magi_planning import GraphManager

# Initialize required components
communication_protocol = StandardCommunicationProtocol()
quality_assurance_layer = QualityAssuranceLayer()
graph_manager = GraphManager()

# Create MagiPlanning instance
magi_planner = MagiPlanning(communication_protocol, quality_assurance_layer, graph_manager)

# Define a goal and problem analysis
goal = "Develop a new feature for our AI-powered chatbot"
problem_analysis = {
    "problem_statement": "Our chatbot needs to handle multi-turn conversations more effectively.",
    "constraints": ["Must be implemented within 2 weeks", "Should not increase latency by more than 100ms"],
    "resources": ["2 senior developers", "1 ML engineer", "Access to GPT-4 API"]
}

# Generate a plan
plan = await magi_planner.generate_plan(goal, problem_analysis)

# Print plan summary
print(f"Plan for goal: {plan['goal']}")
print(f"Success likelihood: {plan['success_likelihood']:.2f}")
print(f"Number of tasks: {plan['metrics']['total_tasks']}")
print(f"Estimated time: {plan['metrics']['total_estimated_time']} hours")
print(f"Capability gaps: {', '.join(plan['capability_gaps'])}")
print(f"Plan visualization saved as: {plan['visualization']}")
```

## Advanced Usage

MagiPlanning offers more advanced features for complex problem-solving scenarios:

1. **Executing individual tasks:**
   ```python
   task_result = await magi_planner.execute_task("Implement a context management system for the chatbot")
   print(f"Task executed using {task_result['technique']} technique")
   print(f"Execution time: {task_result['execution_time']:.2f} seconds")
   ```

2. **Analyzing execution results:**
   ```python
   analysis = await magi_planner._analyze_execution_result(task_result, plan['plan_tree'])
   print("Success factors:", analysis['success_factors'])
   print("Areas for improvement:", analysis['areas_for_improvement'])
   ```

3. **Updating models based on execution results:**
   ```python
   await magi_planner._update_models(task_result, task_result, analysis)
   ```

## Extending MagiPlanning

To add new reasoning techniques or planning strategies, you can extend the MagiPlanning class and implement new methods. Make sure to update the `choose_technique` method to include your new technique in the selection process.

## Note

This class requires access to a language model (defaulting to GPT-4) for generating responses and performing reasoning tasks. Ensure you have the necessary API access and credentials set up before using MagiPlanning.
