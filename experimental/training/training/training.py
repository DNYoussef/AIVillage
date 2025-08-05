from dataclasses import dataclass
import os
import random
import sqlite3
from typing import Any

from langroid import ChatAgent, ChatAgentConfig, Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig
import numpy as np
import torch
from torch import nn, optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_forge.foundation.bitnet import q_bitnet as quantize_weights

from .grokfast import GrokFastTask
from .quiet_star import QuietSTaRModel


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    enable_quiet_star: bool = False
    model_name: str = "distilgpt2"


class CodingTask(ToolMessage):
    request: str = "coding_task"
    purpose: str = "Present a coding task to be solved"
    task_description: str
    expected_result: str
    difficulty: int
    success_count: int = 0


class CodeState:
    def __init__(self, task: CodingTask, code: str = "", thoughts: str = "", response: str = "") -> None:
        self.task = task
        self.code = code
        self.thoughts = thoughts
        self.response = response


class MCTSNode:
    def __init__(self, state: CodeState, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value = 0.0


class EnhancedMagiAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig) -> None:
        super().__init__(config)
        self.enable_message(CodingTask)
        self.cognitive_strategies = [
            "systems_thinking",
            "first_principles",
            "cross_domain",
            "probabilistic_thinking",
            "rapid_iteration",
            "paradox_resolution",
        ]

    async def coding_task(self, task: CodingTask) -> tuple[str, str, str]:
        thoughts = await self.generate_thoughts(task)
        response = await self.llm_response(f"Explain your approach to solving this task: {task.task_description}")
        code = await self.llm_response(f"Write code to solve this task: {task.task_description}")
        return thoughts, response.content, code.content

    async def generate_thoughts(self, task: CodingTask) -> str:
        prompt = f"""
        Apply the following cognitive strategies to solve this coding task:
        {", ".join(self.cognitive_strategies)}

        Task: {task.task_description}

        Provide your thoughts in the following format:
        <start of thought>
        [Your thought process here]
        <end of thought>
        """
        response = await self.llm_response(prompt)
        return response.content

    async def mcts_action(self, state: CodeState) -> str:
        if not state.thoughts:
            return await self.generate_thoughts(state.task)
        if not state.response:
            response = await self.llm_response(
                f"Explain your approach to solving this task: {state.task.task_description}"
            )
            return response.content
        response = await self.llm_response(
            f"Write or improve the code to solve this task: {state.task.task_description}\nCurrent code: {state.code}"
        )
        return response.content


class EnhancedSupervisorAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig) -> None:
        super().__init__(config)

    async def create_initial_test(self) -> list[CodingTask]:
        tasks = []
        for i in range(1, 101):
            prompt = f"""
            Create a coding task with difficulty level {i}/100.
            The task should include:
            1. A clear problem description
            2. The expected code result
            3. The difficulty level ({i})
            Format the output as: 'Description: <description>\nExpected Result: <result>\nDifficulty: {i}'
            """
            response = await self.llm_response(prompt)
            parts = response.content.split("\n")
            tasks.append(
                CodingTask(
                    task_description=parts[0].split(": ", 1)[1],
                    expected_result=parts[1].split(": ", 1)[1],
                    difficulty=i,
                )
            )
        return tasks

    async def grade_attempt(
        self, task: CodingTask, thoughts: str, response: str, code: str
    ) -> tuple[int, int, int, bool]:
        grading_prompt = f"""
        Grade the following coding attempt for the task:
        Task: {task.task_description}
        Expected Result: {task.expected_result}

        Attempt:
        Thoughts: {thoughts}
        Response: {response}
        Code: {code}

        Provide four outputs:
        1. Thought grade (1-10)
        2. Response grade (1-10)
        3. Code grade (1-10)
        4. Overall success (True/False)

        Return only these four values separated by commas.
        """
        grades = await self.llm_response(grading_prompt)
        thought_grade, response_grade, code_grade, success = grades.content.split(",")
        return (
            int(thought_grade),
            int(response_grade),
            int(code_grade),
            success.strip().lower() == "true",
        )

    async def grade_mcts_state(self, state: CodeState) -> tuple[int, int, int, bool]:
        return await self.grade_attempt(state.task, state.thoughts, state.response, state.code)

    def determine_competence_level(self, results: list[tuple[CodingTask, bool]]) -> int:
        sorted_results = sorted(results, key=lambda x: x[0].difficulty)
        for _i, (task, success) in enumerate(sorted_results):
            if not success:
                return task.difficulty - 1
        return 100  # If all tasks were successful

    async def create_training_tasks(self, competence_level: int) -> list[CodingTask]:
        tasks = []
        for _ in range(500):
            difficulty = random.randint(max(1, competence_level - 10), competence_level)
            prompt = f"Create a coding task with difficulty {difficulty}/100. Include description and expected result."
            response = await self.llm_response(prompt)
            description, expected_result = response.content.split("\nExpected Result:")
            tasks.append(
                CodingTask(
                    task_description=description.strip(),
                    expected_result=expected_result.strip(),
                    difficulty=difficulty,
                )
            )

        for _ in range(500):
            difficulty = random.randint(competence_level + 1, min(100, competence_level + 20))
            prompt = f"Create a coding task with difficulty {difficulty}/100. Include description and expected result."
            response = await self.llm_response(prompt)
            description, expected_result = response.content.split("\nExpected Result:")
            tasks.append(
                CodingTask(
                    task_description=description.strip(),
                    expected_result=expected_result.strip(),
                    difficulty=difficulty,
                )
            )

        return tasks

    async def reword_task(self, task: CodingTask) -> CodingTask:
        prompt = f"""
        Reword the following coding task without changing its difficulty or core concept:
        {task.task_description}
        Expected Result: {task.expected_result}

        Provide the reworded task in the same format.
        """
        response = await self.llm_response(prompt)
        description, expected_result = response.content.split("\nExpected Result:")
        return CodingTask(
            task_description=description.strip(),
            expected_result=expected_result.strip(),
            difficulty=task.difficulty,
        )

    async def compute_reward(self, thoughts: str, code: str, task: CodingTask) -> float:
        reasoning_score = await self.analyze_reasoning(thoughts)
        output_score = await self.score_output(code, task.expected_result)
        insight_creativity_score = await self.grade_insight_creativity(thoughts)

        total_reward = reasoning_score + output_score + insight_creativity_score
        return total_reward / 3  # Normalize to [0, 1]

    async def analyze_reasoning(self, thoughts: str) -> float:
        strategies_used = sum(
            1 for strategy in EnhancedMagiAgent(self.config).cognitive_strategies if f"<{strategy}>" in thoughts
        )
        return strategies_used / len(EnhancedMagiAgent(self.config).cognitive_strategies)

    async def score_output(self, code: str, expected_result: str) -> float:
        response = await self.llm_response(
            f"On a scale of 0 to 1, how similar is this code's output likely to be to the expected result?\nCode: {code}\nExpected Result: {expected_result}"
        )
        return float(response.content)

    async def grade_insight_creativity(self, thoughts: str) -> float:
        prompt = f"""
        On a scale of 0 to 1, rate the insight and creativity of these thoughts:
        {thoughts}
        Consider factors like novel connections, unique perspectives, and creative problem-solving approaches.
        """
        grade = await self.llm_response(prompt)
        return float(grade.content)


class AIFeedbackTask(Task):
    async def evaluate_code(self, task: CodingTask, code: str) -> float:
        prompt = f"""
        Evaluate the following code for the given task:

        Task: {task.task_description}
        Expected Result: {task.expected_result}

        Code:
        {code}

        Provide a score between 0 and 1, where 0 is completely incorrect and 1 is perfect.
        Consider the following aspects:
        1. Correctness: Does the code solve the task?
        2. Efficiency: Is the solution efficient?
        3. Readability: Is the code well-structured and easy to understand?
        4. Best practices: Does the code follow good programming practices?

        Return only the numeric score.
        """

        response = await self.agent.llm_response(prompt)
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))  # Ensure the score is between 0 and 1
        except ValueError:
            print(f"Error parsing AI feedback score: {response.content}")
            return 0.0  # Return 0 if we can't parse the score


class CodePreferenceModel(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.encoder(x)


class DPOTrainer:
    def __init__(self, model: CodePreferenceModel, learning_rate: float = 1e-4, beta: float = 0.1) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta

    def train_step(self, preferred: torch.Tensor, non_preferred: torch.Tensor):
        self.optimizer.zero_grad()

        preferred_score = self.model(preferred)
        non_preferred_score = self.model(non_preferred)

        loss = -torch.log(torch.sigmoid(self.beta * (preferred_score - non_preferred_score))).mean()

        loss.backward()

        # Apply 1.58-bit quantization to gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = quantize_weights(param.grad.data)

        self.optimizer.step()

        # Quantize weights after the update
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = quantize_weights(param.data)

        return loss.item()


class HyperparameterOptimizationTask(Task):
    def __init__(self, agent: ChatAgent) -> None:
        super().__init__(agent)
        self.current_hyperparameters = {
            "mcts_iterations": 100,
            "exploration_weight": 1.0,
            "dpo_learning_rate": 1e-4,
            "dpo_beta": 0.1,
            "ai_feedback_weight": 0.3,
        }
        self.performance_history = []

    async def suggest_hyperparameters(self, performance_metric: float, grok_detected: bool) -> dict[str, Any]:
        self.performance_history.append((self.current_hyperparameters.copy(), performance_metric, grok_detected))

        prompt = f"""
        Current hyperparameters:
        {self.current_hyperparameters}

        Performance history:
        {self.performance_history}

        Groking detected: {grok_detected}

        Based on the current hyperparameters, performance history, and groking status, suggest new hyperparameter values to improve performance.
        If groking was detected, consider more aggressive exploration.
        Provide your suggestions in the following format:
        mcts_iterations: <value>
        exploration_weight: <value>
        dpo_learning_rate: <value>
        dpo_beta: <value>
        ai_feedback_weight: <value>

        Explain your reasoning for each suggested change.
        """

        response = await self.agent.llm_response(prompt)

        suggested_hyperparameters = {}
        for line in response.content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":")
                key = key.strip()
                if key in self.current_hyperparameters:
                    try:
                        suggested_hyperparameters[key] = type(self.current_hyperparameters[key])(value.strip())
                    except ValueError:
                        print(f"Failed to parse suggested value for {key}: {value}")

        new_hyperparameters = self.current_hyperparameters.copy()
        for key, value in suggested_hyperparameters.items():
            if random.random() < 0.8:  # 80% chance to accept the suggestion
                new_hyperparameters[key] = value
            else:
                # Random perturbation
                new_hyperparameters[key] *= random.uniform(0.8, 1.2)

        self.current_hyperparameters = new_hyperparameters
        return new_hyperparameters


@dataclass
class TrajectoryStep:
    task_id: int
    step_number: int
    code: str
    thoughts: str
    response: str
    action: str
    reward: float


class DataCollectionPipeline:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                description TEXT,
                expected_result TEXT,
                difficulty INTEGER
            )
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY,
                task_id INTEGER,
                step_number INTEGER,
                code TEXT,
                thoughts TEXT,
                response TEXT,
                action TEXT,
                reward REAL,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """
        )
        self.conn.commit()

    def store_task(self, task: CodingTask) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO tasks (description, expected_result, difficulty)
            VALUES (?, ?, ?)
        """,
            (task.task_description, task.expected_result, task.difficulty),
        )
        self.conn.commit()
        return cursor.lastrowid

    def store_trajectory(self, task_id: int, trajectory: list[TrajectoryStep]) -> None:
        for step in trajectory:
            self.conn.execute(
                """
                INSERT INTO trajectories (task_id, step_number, code, thoughts, response, action, reward)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task_id,
                    step.step_number,
                    step.code,
                    step.thoughts,
                    step.response,
                    step.action,
                    step.reward,
                ),
            )
        self.conn.commit()

    def get_trajectories(self, task_id: int) -> list[TrajectoryStep]:
        cursor = self.conn.execute(
            """
            SELECT * FROM trajectories WHERE task_id = ? ORDER BY step_number
        """,
            (task_id,),
        )
        return [TrajectoryStep(**row) for row in cursor.fetchall()]

    def get_all_tasks(self) -> list[CodingTask]:
        cursor = self.conn.execute("SELECT * FROM tasks")
        return [CodingTask(**row) for row in cursor.fetchall()]


def uct_select_child(node: MCTSNode, exploration_weight: float = 1.0) -> MCTSNode:
    def uct_value(n):
        if n.visits == 0:
            return float("inf")
        return n.value / n.visits + exploration_weight * np.sqrt(np.log(node.visits) / n.visits)

    return max(node.children, key=uct_value)


async def expand(node: MCTSNode, magi_agent: EnhancedMagiAgent) -> MCTSNode:
    new_action = await magi_agent.mcts_action(node.state)
    new_state = CodeState(
        node.state.task,
        code=node.state.code + new_action if node.state.response else node.state.code,
        thoughts=new_action if not node.state.thoughts else node.state.thoughts,
        response=new_action if node.state.thoughts and not node.state.response else node.state.response,
    )
    child = MCTSNode(new_state, parent=node)
    node.children.append(child)
    return child


async def simulate(
    node: MCTSNode,
    supervisor_agent: EnhancedSupervisorAgent,
    ai_feedback_task: AIFeedbackTask,
    ai_feedback_weight: float,
) -> float:
    (
        thought_grade,
        response_grade,
        code_grade,
        success,
    ) = await supervisor_agent.grade_mcts_state(node.state)
    ai_score = await ai_feedback_task.evaluate_code(node.state.task, node.state.code)
    return (thought_grade + response_grade + code_grade) / 30.0 * (
        1 - ai_feedback_weight
    ) + ai_score * ai_feedback_weight


def backpropagate(node: MCTSNode, reward: float) -> None:
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent


async def mcts_code_generation(
    root_state: CodeState,
    magi_agent: EnhancedMagiAgent,
    supervisor_agent: EnhancedSupervisorAgent,
    ai_feedback_task: AIFeedbackTask,
    iterations: int,
    exploration_weight: float,
    ai_feedback_weight: float,
    trajectory: list[TrajectoryStep],
) -> CodeState:
    root = MCTSNode(root_state)

    for i in range(iterations):
        node = root

        # Selection
        while node.children and node.state.code:
            node = uct_select_child(node, exploration_weight)

        # Expansion
        if node.visits > 0 and not node.state.code:
            node = await expand(node, magi_agent)

        # Simulation
        reward = await simulate(node, supervisor_agent, ai_feedback_task, ai_feedback_weight)

        # Backpropagation
        backpropagate(node, reward)

        # Collect trajectory data
        trajectory.append(
            TrajectoryStep(
                task_id=-1,  # This will be updated when storing in the database
                step_number=i,
                code=node.state.code,
                thoughts=node.state.thoughts,
                response=node.state.response,
                action=node.state.code[-1] if node.state.code else "",
                reward=reward,
            )
        )

    # Return the best child's state
    best_child = max(root.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
    return best_child.state


def create_preference_pairs(
    trajectories: list[tuple[CodeState, float]],
) -> list[tuple[CodeState, CodeState]]:
    pairs = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            if trajectories[i][1] > trajectories[j][1]:
                pairs.append((trajectories[i][0], trajectories[j][0]))
            elif trajectories[j][1] > trajectories[i][1]:
                pairs.append((trajectories[j][0], trajectories[i][0]))
    return pairs


def code_state_to_tensor(
    state: CodeState,
    quiet_model: QuietSTaRModel | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> torch.Tensor:
    """Convert a CodeState to a feature tensor.

    If a QuietSTaRModel is provided, an additional feature based on the
    mixed logits is appended.
    """
    features = torch.tensor(
        [
            len(state.code),
            len(state.thoughts),
            len(state.response),
            state.task.difficulty,
        ],
        dtype=torch.float32,
    )

    if quiet_model is not None and tokenizer is not None:
        tokens = tokenizer.encode(state.code or "", return_tensors="pt")
        with torch.no_grad():
            mixed, _ = quiet_model(tokens, generate_thoughts=True)
        features = torch.cat([features, mixed.mean().float().view(1)])

    return features


class TrainingTask(Task):
    async def run_training_loop(
        self,
        magi_agent: EnhancedMagiAgent,
        supervisor_agent: EnhancedSupervisorAgent,
        hyperparameters: dict[str, Any],
        config: TrainingConfig | None = None,
    ) -> tuple[float, bool]:
        config = config or TrainingConfig()

        ai_feedback_task = AIFeedbackTask(self.agent)
        data_pipeline = DataCollectionPipeline("magi_training.db")

        quiet_model = None
        tokenizer = None
        input_size = 4
        if config.enable_quiet_star:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
            quiet_model = QuietSTaRModel(base_model)
            quiet_model.eval()
            input_size = 5

        dpo_model = CodePreferenceModel(input_size=input_size, hidden_size=64)
        dpo_trainer = DPOTrainer(
            dpo_model,
            learning_rate=hyperparameters["dpo_learning_rate"],
            beta=hyperparameters["dpo_beta"],
        )

        grokfast_task = None
        best_val_performance = float("-inf")
        overfitting_detected = False
        grokfast_activated = False
        patience = hyperparameters.get("patience", 5)
        patience_counter = 0
        grok_threshold = hyperparameters.get("grok_threshold", 0.1)
        grok_detected = False

        # Create initial test
        initial_test = await supervisor_agent.create_initial_test()

        # Evaluate Magi's current competence
        test_results = []
        for task in initial_test:
            task_id = data_pipeline.store_task(task)
            initial_state = CodeState(task)
            trajectory = []
            final_state = await mcts_code_generation(
                initial_state,
                magi_agent,
                supervisor_agent,
                ai_feedback_task,
                iterations=hyperparameters["mcts_iterations"],
                exploration_weight=hyperparameters["exploration_weight"],
                ai_feedback_weight=hyperparameters["ai_feedback_weight"],
                trajectory=trajectory,
            )
            _, _, _, success = await supervisor_agent.grade_mcts_state(final_state)
            test_results.append((task, success))

            # Store the trajectory
            data_pipeline.store_trajectory(task_id, trajectory)

        competence_level = supervisor_agent.determine_competence_level(test_results)
        print(f"Magi's initial competence level: {competence_level}/100")

        # Create training tasks
        training_tasks = await supervisor_agent.create_training_tasks(competence_level)

        # Training loop
        max_rounds = 20
        total_success = 0
        total_tasks = 0
        for round in range(max_rounds):
            print(f"Starting training round {round + 1}/{max_rounds}")
            new_training_tasks = []

            all_trajectories = []
            for task in training_tasks:
                task_id = data_pipeline.store_task(task)
                initial_state = CodeState(task)
                trajectory = []
                final_state = await mcts_code_generation(
                    initial_state,
                    magi_agent,
                    supervisor_agent,
                    ai_feedback_task,
                    iterations=hyperparameters["mcts_iterations"],
                    exploration_weight=hyperparameters["exploration_weight"],
                    ai_feedback_weight=hyperparameters["ai_feedback_weight"],
                    trajectory=trajectory,
                )
                (
                    thought_grade,
                    response_grade,
                    code_grade,
                    success,
                ) = await supervisor_agent.grade_mcts_state(final_state)

                # Store the trajectory
                data_pipeline.store_trajectory(task_id, trajectory)
                all_trajectories.extend([(step, step.reward) for step in trajectory])

                if success:
                    total_success += 1
                    task.success_count += 1
                    if task.success_count < 5:
                        new_task = await supervisor_agent.reword_task(task)
                        new_task.success_count = task.success_count
                        new_training_tasks.append(new_task)
                else:
                    task.success_count = 0
                    feedback = f"Grades: Thinking ({thought_grade}/10), Response ({response_grade}/10), Code ({code_grade}/10). "
                    feedback += (
                        await supervisor_agent.llm_response(
                            f"Explain why this attempt failed and how to improve: {final_state.thoughts}\n{final_state.response}\n{final_state.code}"
                        )
                    ).content
                    task.feedback = feedback
                    new_training_tasks.append(task)

                total_tasks += 1

            # Train DPO model
            preference_pairs = create_preference_pairs(all_trajectories)
            for preferred, non_preferred in preference_pairs:
                preferred_tensor = code_state_to_tensor(
                    preferred,
                    quiet_model=quiet_model,
                    tokenizer=tokenizer,
                )
                non_preferred_tensor = code_state_to_tensor(
                    non_preferred,
                    quiet_model=quiet_model,
                    tokenizer=tokenizer,
                )
                dpo_trainer.train_step(preferred_tensor, non_preferred_tensor)

                if grokfast_task:
                    await grokfast_task.filter_gradients()

            training_tasks = new_training_tasks
            print(f"Tasks remaining: {len(training_tasks)}")

            # Validation step
            val_performance = await self.evaluate_model(
                magi_agent,
                supervisor_agent,
                hyperparameters,
                quiet_model=quiet_model,
                tokenizer=tokenizer,
            )
            print(f"Validation performance: {val_performance:.4f}")

            if val_performance > best_val_performance:
                best_val_performance = val_performance
                torch.save(dpo_model.state_dict(), f"best_model_round_{round + 1}.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if not overfitting_detected:
                    print("Overfitting detected. Activating GrokFast.")
                    grokfast_task = GrokFastTask(self.agent, dpo_model, method="EMA", lamb=2.0, alpha=0.98)
                    overfitting_detected = True
                    grokfast_activated = True
                    patience_counter = 0  # Reset patience counter
                elif grokfast_activated:
                    # Check for the "second spike" indicating groking
                    if val_performance > best_val_performance * (1 + grok_threshold):
                        print("Groking detected! Significant performance improvement observed.")
                        best_val_performance = val_performance
                        patience_counter = 0  # Reset patience counter
                        grok_detected = True
                    else:
                        print("No significant improvement. Continuing training with GrokFast.")
                else:
                    print("Training complete. No groking observed.")
                    break

            if not training_tasks:
                print("All tasks completed successfully!")
                break

        print(f"Training completed after {round + 1} rounds.")
        print(f"Final validation performance: {best_val_performance:.4f}")

        return best_val_performance, grok_detected

    async def evaluate_model(
        self,
        magi_agent: EnhancedMagiAgent,
        supervisor_agent: EnhancedSupervisorAgent,
        hyperparameters: dict[str, Any],
        quiet_model: QuietSTaRModel | None = None,
        tokenizer: AutoTokenizer | None = None,
    ) -> float:
        validation_tasks = await supervisor_agent.create_training_tasks(50)  # Create 50 validation tasks
        total_reward = 0

        for task in validation_tasks:
            initial_state = CodeState(task)
            trajectory = []
            final_state = await mcts_code_generation(
                initial_state,
                magi_agent,
                supervisor_agent,
                AIFeedbackTask(self.agent),
                iterations=hyperparameters["mcts_iterations"],
                exploration_weight=hyperparameters["exploration_weight"],
                ai_feedback_weight=hyperparameters["ai_feedback_weight"],
                trajectory=trajectory,
            )
            reward = await supervisor_agent.compute_reward(final_state.thoughts, final_state.code, task)
            total_reward += reward

        return total_reward / len(validation_tasks)


class OptimizationTask(Task):
    async def run_training_loop_with_optimization(
        self,
        magi_agent: EnhancedMagiAgent,
        supervisor_agent: EnhancedSupervisorAgent,
        *,
        config: TrainingConfig | None = None,
        optimization_rounds: int = 10,
    ):
        config = config or TrainingConfig()

        hyperparameter_optimization_task = HyperparameterOptimizationTask(self.agent)
        training_task = TrainingTask(self.agent)

        best_performance = float("-inf")
        best_hyperparameters = None
        grok_detected = False

        for optimization_round in range(optimization_rounds):
            print(f"Starting optimization round {optimization_round + 1}")

            # Get current hyperparameters
            hyperparameters = hyperparameter_optimization_task.current_hyperparameters

            # Run the training loop with current hyperparameters
            performance_metric, grok_occurred = await training_task.run_training_loop(
                magi_agent,
                supervisor_agent,
                hyperparameters,
                config,
            )

            print(f"Performance metric: {performance_metric}")
            if grok_occurred:
                print("Groking detected in this round!")
                grok_detected = True

            if performance_metric > best_performance:
                best_performance = performance_metric
                best_hyperparameters = hyperparameters.copy()

            # Suggest new hyperparameters based on performance and grok occurrence
            new_hyperparameters = await hyperparameter_optimization_task.suggest_hyperparameters(
                performance_metric, grok_detected
            )

            print(f"New suggested hyperparameters: {new_hyperparameters}")

            if grok_detected:
                # If groking occurred, we might want to explore these hyperparameters more thoroughly
                for _ in range(3):  # Run a few more times with these hyperparameters
                    performance_metric, _ = await training_task.run_training_loop(
                        magi_agent,
                        supervisor_agent,
                        hyperparameters,
                        config,
                    )
                    if performance_metric > best_performance:
                        best_performance = performance_metric
                        best_hyperparameters = hyperparameters.copy()

        print(f"Best performance: {best_performance}")
        print(f"Best hyperparameters: {best_hyperparameters}")
        print(f"Groking detected: {grok_detected}")

        return best_hyperparameters, grok_detected


async def main() -> None:
    magi_config = ChatAgentConfig(
        name="Magi",
        llm=OpenAIGPTConfig(
            chat_model="gpt-4",
        ),
    )
    magi_agent = EnhancedMagiAgent(magi_config)

    supervisor_config = ChatAgentConfig(
        name="Supervisor",
        llm=OpenAIGPTConfig(
            chat_model="gpt-4",
        ),
    )
    supervisor_agent = EnhancedSupervisorAgent(supervisor_config)

    optimization_task = OptimizationTask(magi_agent)
    training_config = TrainingConfig(enable_quiet_star=os.environ.get("ENABLE_QUIET_STAR") == "1")
    (
        best_hyperparameters,
        grok_detected,
    ) = await optimization_task.run_training_loop_with_optimization(
        magi_agent,
        supervisor_agent,
        config=training_config,
    )

    # Final run with best hyperparameters
    print("Running final training loop with best hyperparameters")
    training_task = TrainingTask(magi_agent)
    final_performance, _ = await training_task.run_training_loop(
        magi_agent,
        supervisor_agent,
        best_hyperparameters,
        training_config,
    )
    print(f"Final performance: {final_performance}")

    # Save the final model
    torch.save(magi_agent.state_dict(), "final_magi_model.pth")
    print("Final model saved as 'final_magi_model.pth'")

    # Additional analysis and reporting could be added here
    # For example, you might want to generate a report on the training process,
    # including learning curves, task success rates, and examples of solved tasks.


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
