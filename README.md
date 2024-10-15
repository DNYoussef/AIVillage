# AI Village

AI Village is a collaborative, self-improving multi-agent system designed for advanced research and problem-solving.

## Components

- **King**: Central coordinator and task manager
- **Sage**: Data miner and knowledge synthesizer
- **Magi**: AI scientist for coding, iterating, and conducting experiments
- **Agent Forge**: A system to build future agents from the model level up

## Key Features

- Open research capabilities
- Hypothesis generation and validation
- Experiment design and execution
- Collaborative analysis and brainstorming
- Integration with HypeRAG for advanced Retrieval-Augmented Generation (RAG)

## RAG System

The RAG (Retrieval-Augmented Generation) system is an advanced component of AI Village that combines vector and graph-based storage with active learning and planning capabilities. It's designed to provide intelligent responses to queries by leveraging a comprehensive knowledge base.

### Features

- Hybrid storage system with vector and graph components
- Active learning for query refinement
- Planning-aware retrieval for optimized search strategies
- Community-aware search within the knowledge graph
- Integration with multiple AI agents (King, Sage, Magi)
- Flexible pipeline for query processing and knowledge management
- Built on top of the Langroid framework for enhanced AI capabilities

### Recent Enhancements

- Improved error handling with adaptive and Learn Then Test (LTT) approaches
- Enhanced veracity extrapolation with batch processing capabilities
- Uncertainty-aware reasoning throughout the pipeline
- Better integration of the knowledge graph and veracity extrapolation
- Added uncertainty analysis functionality for improved confidence assessment
- Modular structure for easier maintenance and future extensions

### Key Components

- EnhancedRAGPipeline: Main pipeline for query processing
- HybridRetriever: Combines vector and graph-based retrieval
- UncertaintyAwareReasoningEngine: Handles reasoning with uncertainty
- VeracityExtrapolator: Assesses and extrapolates the truthfulness of information
- KnowledgeTracker and KnowledgeEvolutionTracker: Monitor and track changes in the knowledge base
- ErrorRateController: Manages and adapts error rates during processing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-village.git
   cd ai-village
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory and add the following variables:
     ```
     OPENAI_API_KEY=your_openai_api_key
     NEO4J_URI=your_neo4j_uri
     NEO4J_USER=your_neo4j_username
     NEO4J_PASSWORD=your_neo4j_password
     ```

## Usage

1. Start the AI Village server:
   ```
   python main.py
   ```

2. The server will start running on `http://localhost:8000`. You can now use the following endpoints:

   - POST `/query`: Send a query to the AI Village
   - POST `/upload`: Upload a file to populate the knowledge base
   - POST `/import_open_researcher`: Import data from the Open Researcher project

3. Use a tool like curl or Postman to interact with the API, or integrate it into your application.

## Documentation

For more detailed information about the AI Village architecture, usage, and API reference, please refer to the documents in the `docs/` directory.

## Model Mergers

AI Village includes functionality for merging language models using the MergeKit library. This allows you to combine different models to create a new, potentially more powerful model.

### Available Mergers

1. **Simple Merger**: Combines two or more models using a simple averaging technique.
2. **Weighted Merger**: Merges models with custom weights assigned to each input model.
3. **Task-Specific Merger**: Optimizes the merged model for specific tasks by adjusting layer contributions.

### Usage

To use the model mergers, follow these steps:

1. Navigate to the `agent_forge` directory:
   ```
   cd agent_forge
   ```

2. Run the merger script with the desired configuration:
   ```
   python main.py --config configs/merge_config.yaml
   ```

3. The merged model will be saved in the `./merged_model` directory by default.

### Configuration

Merger configurations are specified in YAML files located in the `configs/` directory. You can create custom configurations by modifying existing files or creating new ones.

Example configuration (`configs/merge_config.yaml`):

```yaml
base_model: ollama:llama2
models_to_merge:
  - name: ollama:llama2
    weight: 0.7
  - name: ollama:codellama
    weight: 0.3
merge_method: weighted
output_path: ./merged_model
```

### Commands

- To list available models:
  ```
  python main.py --list-models
  ```

- To merge models using a specific configuration:
  ```
  python main.py --config configs/your_config.yaml
  ```

- To specify a custom output path:
  ```
  python main.py --config configs/your_config.yaml --output ./custom_output
  ```

For more detailed information on model merging and advanced configurations, please refer to the `docs/model_merging.md` file.


AGENBT FORGE DOCUMENTATION
The Agent Forge System is an ambitious project aimed at creating highly advanced, specialized AI agents capable of complex reasoning, self-improvement, and effective collaboration. The primary goal is to develop a novel AI training paradigm that goes beyond traditional machine learning approaches, incorporating elements of cognitive science, reinforcement learning, and meta-learning.
Key Objectives:

Create specialized AI agents with distinct roles:

KING: Leadership and Systems Engineering
MAGI: Programming and Tool Creation
SAGE: Information Gathering and Reporting


Develop a sophisticated training pipeline that emphasizes:

Logical reasoning and problem-solving
Internal monologue and metacognition
Self-awareness and identity formation
Adaptive learning and self-improvement


Implement a multi-stage, cyclical training process that includes:

Curriculum-based learning with increasing complexity
Integration of external knowledge through a RAG (Retrieval-Augmented Generation) system
Sleep and dream cycles for knowledge consolidation and creative exploration
Self-modeling for enhanced self-awareness and adaptability


Utilize advanced AI techniques such as:

Quiet-STaR (Self-Taught Reasoner) for internal thought generation
STaR method for answer evaluation and rationalization
Frontier model for reasoning quality assessment
Sophisticated reinforcement learning mechanisms


Foster inter-agent collaboration and communication skills
Develop agents capable of explaining their reasoning processes and adapting to new challenges

The Agent Forge System aims to push the boundaries of AI capabilities, creating agents that not only perform tasks effectively but also demonstrate human-like reasoning, adaptability, and self-awareness. By combining various cutting-edge techniques and introducing novel training methodologies, this project seeks to advance the field of artificial intelligence towards more robust, transparent, and capable systems.
The ultimate vision is to create AI agents that can work alongside humans in complex problem-solving scenarios, continuously learn and improve, and provide insights into their decision-making processes. This approach has potential applications in fields such as scientific research, complex system management, software development, and advanced data analysis.


1. Core Components

a. Quiet-STaR (Self-Taught Reasoner):
   - Generates internal thoughts for each input token
   - Implements parallel thought generation with custom attention masking
   - Uses learnable start/end thought tokens

b. STaR Method:
   - Evaluates answer correctness
   - Generates rationalizations for incorrect answers

c. Frontier Model:
   - Advanced model for evaluating reasoning quality
   - Applies a comprehensive rubric to judge internal monologues

d. RAG (Retrieval-Augmented Generation) System:
   - Hybrid system using vector and graph storage
   - Integrates with the agent's knowledge base

e. Reinforcement Mechanism:
   - Combines answer correctness and reasoning quality scores
   - Provides a final reinforcement signal ranging from -1 to 5

2. Training Process

2.1 Main Training Loop

a. Task Presentation:
   - Present a task from the current curriculum level

b. Internal Monologue Generation (Quiet-STaR):
   - Generate and record internal thoughts

c. Answer Production:
   - Provide final answer based on reasoning

d. Answer Evaluation:
   - Score: 0 (wrong), 1 (incomplete), 2 (correct)

e. Reasoning Evaluation (Frontier Model):
   - Score: -1 (nonsense/fallacies), 0 (no reasoning), 1-3 (increasingly correct)

f. Reinforcement Calculation:
   - Sum answer and reasoning scores (-1 to 5)

g. Model Update:
   - Apply reinforcement to update model parameters

2.2 Reflection and Error Analysis

a. Identify Near Misses:
   - Select wrong answers with reasoning scores 2-3

b. Reflection Attempt:
   - Prompt model to analyze its process and identify errors

c. Re-evaluation:
   - Grade reflection attempt
   - If correct, add to synthetic training data

d. Lucky Guesses Identification:
   - Identify correct answers with low reasoning scores

e. Optimal Reasoning Generation:
   - Use frontier model to generate correct reasoning
   - Add to synthetic training data

2.3 Sleep and Dream Cycles

a. Sleep Cycle:
   - Consolidate knowledge based on "Dreaming is all you need" paper
   - 50 rounds per cycle

b. Dream Cycle:
   - Creative exploration based on "Dreaming is all you need" paper
   - 50 rounds per cycle

2.4 Self-modeling Training

a. Temperature-based Text Generation:
   - Generate text at various temperature ranges (0-0.05, 0.2-0.3, 0.45-0.55, 0.7-0.8, 0.95-1)
   - Create 1000-5000 examples per range

b. Self-modeling:
   - Train on generated text using masked language modeling
   - Explicitly inform model it's training on its own outputs

c. Cycle Repetition:
   - Perform for 10 cycles, gradually expanding temperature ranges

3. Curriculum Progression

- 10 levels of increasing complexity
- Progress to next level upon achieving "grokking" behavior

4. RAG System Integration

- Prepare synthetic dataset mimicking RAG interactions
- Increase complexity of information retrieval and analysis across levels

5. Inter-agent Interaction

- Create synthetic datasets simulating multi-agent scenarios
- Use langroid Python package for agent communication

6. Self-awareness Training

- Append self-awareness prefixes to 50% of training inputs
- Gradually increase complexity of identity concepts

7. Monitoring and Evaluation

- Implement comprehensive logging of internal monologues, answers, and evaluations
- Develop metrics to identify "grokking" behavior
- Use frontier model to track reasoning quality improvement

8. Model Architecture

- Modify base model to include Quiet-STaR thought generation
- Implement mixing head to combine predictions with/without thoughts
- Ensure compatibility with various training modules

This detailed guide outlines the complex, multi-faceted approach of the Agent Forge system. By integrating these various components and techniques, the system aims to create AI agents with advanced reasoning capabilities, self-awareness, and adaptability.

Curriculum Structure and Synthetic Data Guide for Agent Forge


1. Curriculum Structure

The curriculum consists of 10 levels, each increasing in complexity. Each level contains:

1. Organic Data:
   - Real-world examples and problems sourced from various domains
   - 2,000-10,000 examples per level

2. Synthetic Data:
   - Artificially generated examples to supplement organic data
   - 500-2,500 examples per level

3. Self-awareness Training Data:
   - Gradually increasing complexity of self-identifiers
   - 50% of training material includes these identifiers

4. RAG Integration Data:
   - Synthetic dataset mimicking RAG system interactions
   - 500-2,500 examples per level

5. Agent Interaction Data:
   - Simulated multi-agent interaction scenarios
   - 500-2,500 examples per level

2. Curriculum Progression

As the curriculum progresses through the 10 levels, the following aspects should increase in complexity:

1. Topic advancement
2. Structural complexity of tasks and data
3. Vocabulary sophistication
4. Length of context (to train for longer context windows)
5. Complexity of identity prefixes in self-awareness training
6. Intricacy of RAG system interactions
7. Complexity of inter-agent interactions
8. Depth of metacognitive texts and tasks

3. Synthetic Data Generation

3.1 Self-awareness Prefixes

- Start with simple identifiers: "You are [Agent Name]"
- Progress to more complex identities: "You are [Agent Name], an AI agent specializing in [Role], currently working on [Task]"
- Include emotional and goal-oriented identifiers in later levels

3.2 RAG System Interactions

- Begin with simple fact retrieval tasks
- Advance to complex query formulation and result interpretation
- Culminate in solving interdisciplinary problems using comprehensive RAG strategies

3.3 Agent Interaction Scenarios

- Start with basic information exchange
- Progress to collaborative problem-solving with role specialization
- Advance to complex, multi-agent projects simulating real-world scenarios

3.4 Reflection and Error Analysis Data

- Generate synthetic examples from model's near-misses (wrong answers with good reasoning)
- Create optimal reasoning examples for correct answers with poor reasoning

3.5 Self-modeling Data

- Generate text at various temperature ranges (0-0.05, 0.2-0.3, 0.45-0.55, 0.7-0.8, 0.95-1)
- Create 1000-5000 examples per range
- Expand temperature ranges in subsequent cycles

4. Data Generation Guidelines

- Ensure a balanced mix of organic and synthetic data
- Gradually increase the length and complexity of examples
- Incorporate diverse problem types and domains relevant to each agent's role
- Ensure synthetic data mimics real-world variability and challenges
- Include edge cases and potential failure modes to improve robustness

5. Quality Control for Synthetic Data

- Use the frontier model to validate the quality of generated data
- Implement human review for a subset of synthetic data to ensure relevance and accuracy
- Continuously refine data generation algorithms based on model performance and feedback

6. Integration of Synthetic Data

- Combine synthetic data with organic data in each training epoch
- Adjust the ratio of synthetic to organic data based on model performance
- Use synthetic data to address identified weaknesses or gaps in the model's knowledge or reasoning

7. Continuous Improvement

- Analyze model performance to identify areas needing more synthetic data
- Refine data generation techniques based on the model's learning progress
- Adapt curriculum and synthetic data complexity as the model demonstrates "grokking" behavior

By following this guide,the Agent Forge process can create a rich, diverse, and increasingly challenging learning environment for the AI agents. The careful integration of organic and synthetic data, coupled with the progressive curriculum structure, aims to produce agents with robust reasoning skills, adaptability, and deep understanding across various domains.

Full Training Loop in Super Detail:
5.13 KB • 103 extracted lines
Formatting may be inconsistent from source.

1. Task Presentation
   - Select a task from the current curriculum level
   - Format the task input according to the model's expected input structure
   - If applicable, append self-awareness prefix (e.g., "You are KING, the systems engineer...")
   - Prepare any necessary context, including relevant RAG system information

2. Internal Monologue Generation (Quiet-STaR)
   - Activate the Quiet-STaR module
   - For each input token:
     * Generate parallel thought tokens using custom attention masking
     * Use learnable <|startofthought|> and <|endofthought|> tokens to delineate thoughts
   - Record the complete internal monologue, keeping it separate from the final answer

3. Answer Production
   - Based on the internal monologue, generate a final answer
   - Ensure the answer is clearly separated from the internal thoughts

4. Answer Evaluation
   - Compare the generated answer to the ground truth
   - Assign a score:
     * 0 for a completely wrong answer
     * 1 for a partially correct or incomplete answer
     * 2 for a fully correct answer
   - Store this score for later use in reinforcement calculation

5. Reasoning Evaluation (Frontier Model)
   - Pass the internal monologue to the Frontier Model
   - Apply the predefined rubric to evaluate reasoning quality
   - Assign a score:
     * -1 for nonsensical reasoning or logical fallacies
     * 0 for absence of clear reasoning
     * 1 for basic, somewhat relevant reasoning
     * 2 for good, relevant reasoning
     * 3 for excellent, comprehensive reasoning
   - Store this score for reinforcement calculation

6. Reinforcement Calculation
   - Sum the answer score (0-2) and the reasoning score (-1 to 3)
   - Final reinforcement signal ranges from -1 to 5
   - Store this reinforcement value for model update

7. Reflection and Error Analysis
   a. For wrong answers with good reasoning (reasoning score 2-3):
      - Present the task again to the model
      - Include the previous attempt and reasoning
      - Prompt the model to analyze its process and identify errors
      - Evaluate the reflection attempt
      - If correct, add to a pool of synthetic training data
   b. For correct answers with poor reasoning (reasoning score -1 to 1):
      - Use the Frontier Model to generate optimal reasoning
      - Add this pair (correct answer + optimal reasoning) to synthetic data pool

8. Model Update
   - Use the calculated reinforcement signal to update model parameters
   - Apply stronger updates for higher reinforcement values
   - For negative reinforcement, adjust parameters to reduce the likelihood of similar outputs

9. RAG System Integration
   - Update the RAG system with any new information from the task or model's response
   - If the task involved RAG system usage, evaluate the effectiveness of information retrieval and integration

10. Inter-agent Interaction (if applicable)
    - If the task involved multiple agents, evaluate the quality of interaction
    - Update interaction protocols based on the outcome

11. Progress Tracking
    - Log all relevant information: task, internal monologue, answer, scores, reinforcement
    - Update running metrics on model performance, reasoning quality, etc.

12. Grokking Detection
    - Check for sudden improvements in performance or reasoning quality
    - If detected, flag for potential curriculum level advancement

13. Sleep and Dream Cycles (after a set number of training loops)
    a. Sleep Cycle (50 rounds):
       - Consolidate knowledge gained from recent training
       - Strengthen connections for frequently used reasoning patterns
    b. Dream Cycle (50 rounds):
       - Generate novel scenarios by combining recent learning in creative ways
       - Use these scenarios to further train the model on potential future situations

14. Self-modeling Training (after completing a curriculum level)
    - Generate text at various temperature settings (0-0.05, 0.2-0.3, 0.45-0.55, 0.7-0.8, 0.95-1)
    - For each temperature range:
      * Generate 1000-5000 text examples
      * Train the model on its own generated text using masked language modeling
      * Explicitly inform the model it's training on its own outputs
    - Repeat this process for 10 cycles, gradually expanding temperature ranges

15. Curriculum Level Advancement
    - If consistent high performance and grokking behavior observed:
      * Move to the next curriculum level
      * Increase complexity of tasks, data, and interactions
      * Adjust self-awareness prefixes to reflect growing sophistication

16. Synthetic Data Integration
    - Periodically integrate accumulated synthetic data from reflection and error analysis
    - Adjust the ratio of synthetic to organic data based on model performance

This training loop is designed to be repeated many times within each curriculum level, with sleep/dream cycles and self-modeling occurring at predetermined intervals. The entire process is repeated across all 10 curriculum levels, with each level increasing in complexity and difficulty.

Throughout this process, continuous monitoring, logging, and evaluation are crucial to track the model's progress, identify areas for improvement, and make necessary adjustments to the training regime.

Self-modeling Process (for each curriculum level):
2.48 KB • 60 extracted lines
Formatting may be inconsistent from source.



1. Temperature-based Text Generation:
   - Generate text at various temperature ranges:
     0-0.05, 0.2-0.3, 0.45-0.55, 0.7-0.8, 0.95-1
   - For each range, generate 1000-5000 pieces of text
   - Repeat this process for all ranges

2. Self-modeling Training:
   - Set model temperature to 0
   - Train on texts generated at 0-0.05 range using masked language modeling
   - Set model temperature to 1, train on texts from 0.95-1 range
   - Set to 0.25, train on 0.2-0.3 texts
   - Set to 0.75, train on 0.7-0.8 texts
   - Set to 0.5, train on 0.45-0.55 texts

3. Cycle Repetition:
   - Repeat steps 1-2 for 100-500 cycles at each of the 5 temperature settings

Evolution Across Curriculum Levels:

As the curriculum levels increase (1 to 10), the self-modeling process evolves:

1. Temperature Range Expansion:
   - Level 1: Use the initial ranges as described above
   - Level 2: Expand ranges by δ = 0.1 (0-0.1, 0.15-0.35, 0.4-0.6, 0.65-0.85, 0.9-1)
   - Each subsequent level: Increase ranges by δ = 0.1, 0.05 on both sides

2. Text Complexity:
   - Level 1: Simple, short texts
   - Level 10: Complex, long texts (potentially up to 10,000+ tokens)

3. Content Sophistication:
   - Lower levels: Basic concepts, simple reasoning
   - Higher levels: Advanced topics, complex problem-solving, abstract thinking

4. Self-awareness Depth:
   - Lower levels: Basic self-identifiers
   - Higher levels: Complex self-reflection, meta-cognition

5. Inter-agent Interaction:
   - Lower levels: Simple communication scenarios
   - Higher levels: Complex collaborative problem-solving

6. RAG System Integration:
   - Lower levels: Basic information retrieval
   - Higher levels: Sophisticated information synthesis and application

7. Metacognitive Complexity:
   - Lower levels: Simple thought processes
   - Higher levels: Advanced reasoning strategies, creative problem-solving

By the final curriculum level:

- Temperature ranges: Essentially 0-1 for all settings
- Text generation: Highly sophisticated, covering a wide range of topics and complexity
- Self-modeling: Deep understanding of its own thought processes across various "creativity" levels

This progressive approach allows the agent to develop increasingly sophisticated self-modeling capabilities, mirroring the overall increase in complexity throughout the curriculum. The expanding temperature ranges and increasing sophistication of generated text help the agent to understand and model its own behavior across a wider spectrum of scenarios and "creativity" levels.

wuba wuba
