import ast
import importlib.util
import os
import tempfile
import types
from typing import Any

# Import necessary dependencies (adjust import paths as needed)
from agent_forge.utils.tool_message import ToolMessage

# Safe builtins for code execution
SAFE_BUILTINS: dict[str, Any] = {
    "__import__": __import__,
    "open": open,
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "str": str,
    "float": float,
}


class AgentTechnique(ToolMessage):
    request: str = "apply_technique"
    purpose: str = "Apply a specific AI technique"
    thought: str
    name: str
    code: str

    def handle(self):
        """Return the callable defined by this technique's code."""
        module = self._load_module(self.code)

        if hasattr(module, "run"):
            return module.run
        if hasattr(module, "forward"):
            return module.forward
        return None

    def _load_module(self, code: str) -> types.ModuleType:
        """Safely load a technique's code into a module."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise RuntimeError(f"Invalid code for technique '{self.name}': {e}") from e

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"exec", "eval", "__import__", "compile"}:
                    raise RuntimeError("Disallowed operation in technique code")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            path = tmp.name
        try:
            spec = importlib.util.spec_from_file_location(f"tech_{self.name}", path)
            module = importlib.util.module_from_spec(spec)
            module.__dict__["__builtins__"] = SAFE_BUILTINS
            if spec.loader is not None:
                spec.loader.exec_module(module)
            return module
        finally:
            os.remove(path)


# Rest of the PROMPT_TECHNIQUE_ARCHIVE content remains the same
PROMPT_TECHNIQUE_ARCHIVE = [
    AgentTechnique(
        thought=(
            "Zero-shot prompting provides direct instructions without examples, "
            "allowing for quick task performance without prior demonstrations. "
            "This approach leverages the model's pre-trained knowledge to "
            "generalize to new tasks."
        ),
        name="Zero-Shot Prompting",
        code="""
async def run(self):
    instruction = (
        "Based on the given task, provide a direct answer without any "
        "additional examples or explanation."
    )
    response = await self.agent.llm_response(instruction)
    return response.content
""",
    ),
    AgentTechnique(
        thought=(
            "Few-shot prompting includes a small number of examples before the "
            "main task, enabling the model to learn from demonstrations. This "
            "technique can improve performance with minimal example overhead."
        ),
        name="Few-Shot Prompting",
        code="""
async def run(self):
    few_shot_examples = [
        {"input": "Example input 1", "output": "Example output 1"},
        {"input": "Example input 2", "output": "Example output 2"},
    ]
    instruction = "Here are a few examples of how to approach this type of task:\\n"
    for example in few_shot_examples:
        instruction += f"Input: {example['input']}\\nOutput: {example['output']}\\n\\n"
    instruction += "Now, solve the following task:"

    response = await self.agent.llm_response(instruction)
    return response.content
""",
    ),
    AgentTechnique(
        thought=(
            "Chain-of-Thought (CoT) prompting encourages the model to show its "
            "reasoning steps, enhancing problem-solving capabilities for complex "
            "tasks that require multi-step reasoning."
        ),
        name="Chain-of-Thought",
        code="""
def forward(self, taskInfo):
    instruction = (
        "Think through this step-by-step:\\n"
        "1. Understand the question\\n"
        "2. Identify key information\\n"
        "3. Reason through the problem\\n"
        "4. Formulate your answer\\n"
        "Now, solve the task."
    )
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
    thinking, answer = cot_agent([taskInfo], instruction)
    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Zero-Shot Chain-of-Thought adds a thought-inducing phrase to prompt "
            "step-by-step reasoning without examples, improving reasoning "
            "capabilities in a zero-shot setting."
        ),
        name="Zero-Shot Chain-of-Thought",
        code="""
def forward(self, taskInfo):
    instruction = (
        "Let's approach this step-by-step:\\n"
        "1. Understand the question\\n"
        "2. Identify key information\\n"
        "3. Reason through the problem\\n"
        "4. Formulate your answer\\n"
        "Now, solve the task."
    )
    zero_shot_cot_agent = LLMAgentBase(['thinking', 'answer'], 'Zero-Shot CoT Agent')
    thinking, answer = zero_shot_cot_agent([taskInfo], instruction)
    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Self-Consistency generates multiple Chain-of-Thought paths and "
            "selects the majority answer, increasing answer reliability through "
            "an ensemble method."
        ),
        name="Self-Consistency",
        code="""
def forward(self, taskInfo):
    instruction = (
        "Solve the given task using step-by-step reasoning. "
        "Provide your final answer at the end."
    )
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'CoT Agent', temperature=0.7)

    num_iterations = 5
    answers = []
    for _ in range(num_iterations):
        thinking, answer = cot_agent([taskInfo], instruction)
        answers.append(answer.content)

    from collections import Counter
    most_common_answer = Counter(answers).most_common(1)[0][0]

    return Info('answer', 'Self-Consistency Agent', most_common_answer, 0)
""",
    ),
    AgentTechnique(
        thought=(
            "Least-to-Most Prompting breaks complex problems into simpler "
            "sub-problems, solving them sequentially. This approach helps in "
            "handling complex tasks more effectively by decomposing them into "
            "manageable parts."
        ),
        name="Least-to-Most Prompting",
        code="""
def forward(self, taskInfo):
    decomposition_agent = LLMAgentBase(['sub_tasks'], 'Decomposition Agent')
    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Solving Agent')

    decomp_instruction = (
        "Break down the given task into a list of simpler sub-tasks, "
        "ordered from least to most complex."
    )
    sub_tasks = decomposition_agent([taskInfo], decomp_instruction)[0]

    context = [taskInfo]
    for i, sub_task in enumerate(sub_tasks.content.split('\\n')):
        solve_instruction = (
            f"Solve this sub-task: {sub_task}. "
            "Use previous solutions if relevant."
        )
        thinking, answer = solving_agent(context, solve_instruction)
        context.extend([thinking, answer])

    # The last answer is the solution to the most complex sub-task,
    # i.e., the original task
    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Tree-of-Thoughts creates a tree-like search of multiple reasoning "
            "paths, improving search and planning capabilities for complex "
            "problem-solving."
        ),
        name="Tree-of-Thoughts",
        code="""
def forward(self, taskInfo):
    def expand_node(node, depth):
        if depth == 0:
            return node

        instruction = (
            f"Given the current thought: '{node}', generate three possible "
            "next steps in reasoning."
        )
        expansion_agent = LLMAgentBase(['step1', 'step2', 'step3'], 'Expansion Agent')
        steps = expansion_agent([taskInfo, Info('node', 'Tree', node, 0)], instruction)

        children = [expand_node(step.content, depth-1) for step in steps]
        return (node, children)

    root_instruction = "Provide an initial thought about how to approach this task."
    root_agent = LLMAgentBase(['initial_thought'], 'Root Agent')
    root_thought = root_agent([taskInfo], root_instruction)[0].content

    tree = expand_node(root_thought, depth=2)

    evaluation_instruction = (
        "Evaluate the following reasoning path and provide a final answer "
        "based on it."
    )
    evaluation_agent = LLMAgentBase(['evaluation', 'answer'], 'Evaluation Agent')

    def evaluate_path(path):
        eval_input = (
            [taskInfo] + [Info('step', 'Tree', step, i)
                         for i, step in enumerate(path)]
        )
        evaluation, answer = evaluation_agent(eval_input, evaluation_instruction)
        return answer

    def dfs_paths(tree, path=[]):
        node, children = tree[0], tree[1] if len(tree) > 1 else []
        path = path + [node]
        if not children:
            yield path
        for child in children:
            yield from dfs_paths(child, path)

    best_answer = None
    best_score = float('-inf')

    for path in dfs_paths(tree):
        answer = evaluate_path(path)
        # Assuming answer format is "score|actual_answer"
        score = float(answer.content.split('|')[0])
        if score > best_score:
            best_score = score
            best_answer = answer.content.split('|')[1]

    return Info('answer', 'Tree-of-Thoughts Agent', best_answer, 0)
""",
    ),
    AgentTechnique(
        thought=(
            "Program-of-Thoughts generates programming code as reasoning steps, "
            "excelling in mathematical and programming tasks by leveraging "
            "code-based reasoning."
        ),
        name="Program-of-Thoughts",
        code="""
def forward(self, taskInfo):
    code_generation_agent = LLMAgentBase(['code'], 'Code Generation Agent')
    execution_agent = LLMAgentBase(['result'], 'Code Execution Agent')
    interpretation_agent = LLMAgentBase(
        ['interpretation', 'answer'], 'Interpretation Agent'
    )

    gen_instruction = (
        "Generate Python code to solve the given task. Include print "
        "statements to show intermediate steps."
    )
    code = code_generation_agent([taskInfo], gen_instruction)[0]

    exec_instruction = "Execute the following code and provide the output:"
    result = execution_agent([taskInfo, code], exec_instruction)[0]

    interp_instruction = (
        "Interpret the code execution results and provide a final answer "
        "to the original task."
    )
    interpretation, answer = interpretation_agent(
        [taskInfo, code, result], interp_instruction
    )

    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Prompt Chaining uses multiple prompts in succession to handle "
            "complex multi-step tasks, allowing for a more structured approach "
            "to problem-solving."
        ),
        name="Prompt Chaining",
        code="""
def forward(self, taskInfo):
    chain = [
        ('understand', "Understand and restate the given task."),
        ('plan', "Create a step-by-step plan to solve the task."),
        ('execute', "Execute each step of the plan."),
        ('review', "Review the execution and provide a final answer.")
    ]

    context = [taskInfo]
    for step, instruction in chain:
        agent = LLMAgentBase([step], f'{step.capitalize()} Agent')
        result = agent(context, instruction)[0]
        context.append(result)

    return context[-1]  # The last result is the final answer
""",
    ),
    AgentTechnique(
        thought=(
            "Emotion Prompting incorporates emotional phrases to potentially "
            "improve performance on benchmarks by leveraging psychological "
            "relevance."
        ),
        name="Emotion Prompting",
        code="""
def forward(self, taskInfo):
    emotion_agent = LLMAgentBase(['emotion'], 'Emotion Selection Agent')
    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Emotional Solving Agent')

    emotion_instruction = (
        "Based on the given task, suggest an appropriate emotion that might "
        "help in solving it effectively."
    )
    emotion = emotion_agent([taskInfo], emotion_instruction)[0]

    solving_instruction = (
        f"Imagine you're feeling {emotion.content}. With this emotional "
        "state in mind, solve the given task. Show your reasoning and "
        "provide an answer."
    )
    thinking, answer = solving_agent([taskInfo, emotion], solving_instruction)

    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Self-Ask prompts the model to ask and answer follow-up questions, "
            "improving problem decomposition and solving through self-questioning."
        ),
        name="Self-Ask",
        code="""
def forward(self, taskInfo):
    question_agent = LLMAgentBase(['question'], 'Question Agent')
    answer_agent = LLMAgentBase(['answer'], 'Answer Agent')

    context = [taskInfo]
    max_iterations = 5

    for _ in range(max_iterations):
        q_instruction = (
            "Based on the current context, what's the next question we should "
            "ask to solve the task? If no more questions are needed, respond "
            "with 'DONE'."
        )
        question = question_agent(context, q_instruction)[0]

        if question.content == 'DONE':
            break

        a_instruction = f"Answer the following question: {question.content}"
        answer = answer_agent(context + [question], a_instruction)[0]

        context.extend([question, answer])

    final_answer_agent = LLMAgentBase(['final_answer'], 'Final Answer Agent')
    final_instruction = (
        "Based on all the questions and answers, provide the final answer "
        "to the original task."
    )
    final_answer = final_answer_agent(context, final_instruction)[0]

    return final_answer
""",
    ),
    AgentTechnique(
        thought=(
            "Contrastive Chain-of-Thought includes both correct and incorrect "
            "explanations to enhance reasoning by showing what not to do, "
            "leveraging contrast learning."
        ),
        name="Contrastive Chain-of-Thought",
        code="""
def forward(self, taskInfo):
    cot_agent = LLMAgentBase(
        ['correct_thinking', 'incorrect_thinking'], 'CoT Generation Agent'
    )
    contrast_agent = LLMAgentBase(['analysis', 'answer'], 'Contrast Analysis Agent')

    cot_instruction = (
        "Provide both a correct and an incorrect chain of thought for "
        "solving the given task."
    )
    correct_thinking, incorrect_thinking = cot_agent(
        [taskInfo], cot_instruction
    )

    contrast_instruction = (
        "Analyze the correct and incorrect reasoning paths. Explain the "
        "differences and provide the final correct answer."
    )
    analysis, answer = contrast_agent(
        [taskInfo, correct_thinking, incorrect_thinking],
        contrast_instruction
    )

    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Memory-of-Thought uses unlabeled data to build Few-Shot "
            "Chain-of-Thought prompts, improving performance on various "
            "reasoning tasks through dynamic example retrieval."
        ),
        name="Memory-of-Thought",
        code="""
def forward(self, taskInfo):
    memory_bank = [...]  # Assume this is a large list of unlabeled examples

    retrieval_agent = LLMAgentBase(['relevant_examples'], 'Retrieval Agent')
    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Solving Agent')

    retrieval_instruction = (
        "Select the top 3 most relevant examples from the memory bank for "
        "solving the given task."
    )
    relevant_examples = retrieval_agent(
        [taskInfo, Info('memory', 'Memory Bank', str(memory_bank), 0)],
        retrieval_instruction
    )[0]

    solving_instruction = (
        "Use the provided relevant examples as a guide to solve the given "
        "task. Show your reasoning."
    )
    thinking, answer = solving_agent([taskInfo, relevant_examples], solving_instruction)

    return answer
""",
    ),
    AgentTechnique(
        thought=(
            "Choice Annealing gradually narrows down a large set of initial "
            "ideas to a final, refined solution. It starts with high randomness "
            "and gradually reduces it while refining ideas, useful for "
            "open-ended problems or creative tasks."
        ),
        name="Choice Annealing",
        code="""
async def run(self):
    annealing_module = self.agent.create_sub_agent(['ideas'], 'Annealing Module')

    temperature = 1.0
    num_ideas = 20
    final_plan = []

    while temperature > 0.05 and len(final_plan) < 20:
        if len(final_plan) == 0:
            instruction = (
                f"Generate {num_ideas} diverse initial ideas for solving this "
                "problem."
            )
        else:
            instruction = (
                f"Consider the previous ideas and assessment. Generate "
                f"{num_ideas} ideas that expand on parts {len(final_plan)+1} to "
                f"{min(len(final_plan)+3, 20)} of the plan."
            )

        ideas = await annealing_module.llm_response(
            instruction, temperature=temperature
        )
        ideas = ideas.content.split('\\n')

        assessment_instruction = (
            "Critique these ideas, find the best parts, consolidate them, "
            "and write a final assessment."
        )
        assessment = await annealing_module.llm_response(
            assessment_instruction, temperature=temperature
        )

        final_plan.append(assessment.content)
        temperature -= 0.05
        num_ideas -= 1

    return '\\n'.join(final_plan)
""",
    ),
    AgentTechnique(
        thought=(
            "Evolutionary I-beam Tournament evolves and refines ideas through "
            "cycles of mutation, selection, and recombination. It's useful "
            "when you have an initial solution or idea that you want to "
            "improve upon through iterative refinement and combination of "
            "features."
        ),
        name="Evolutionary I-beam Tournament",
        code="""
async def run(self):
    evolution_module = self.agent.create_sub_agent(['ideas'], 'Evolution Module')
    test_module = self.agent.create_sub_agent(['scores'], 'Test Module')

    initial_idea = self.task
    population_size = 8
    num_generations = 5

    for generation in range(num_generations):
        if generation == 0:
            instruction = (
                f"Mutate the initial idea into {population_size} different "
                "permutations."
            )
            population_response = await evolution_module.llm_response(instruction)
            population = population_response.content.split('\\n')
        else:
            # Mutate winners
            winner_mutations = []
            for winner in winners:
                instruction = (
                    f"Mutate this idea into 3 new variants: {winner}"
                )
                mutations_response = await evolution_module.llm_response(instruction)
                mutations = mutations_response.content.split('\\n')
                winner_mutations.extend(mutations)

            # Recombine losers
            loser_recombination_instruction = (
                "Analyze these ideas and combine them in a unique way to create "
                "2 new permutations that synergize their best qualities:"
            )
            loser_recombinations_response = await evolution_module.llm_response(
                loser_recombination_instruction + '\\n' + '\\n'.join(losers)
            )
            loser_recombinations = loser_recombinations_response.content.split('\\n')

            population = winner_mutations + loser_recombinations

        # Test ideas
        test_instruction = (
            "Score each of these ideas from 1-10 based on their potential "
            "to solve the problem."
        )
        scores_response = await test_module.llm_response(
            test_instruction + '\\n' + '\\n'.join(population)
        )
        scores = scores_response.content.split('\\n')

        # Select winners and losers
        sorted_population = sorted(
            zip(population, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )
        winners = [idea for idea, _ in sorted_population[:2]]
        losers = [idea for idea, _ in sorted_population[2:]]

    return winners[0]  # Return the best idea
""",
    ),
    AgentTechnique(
        thought=(
            "Exploration Module alternates between conservative and creative "
            "thinking to explore an idea space thoroughly. It's useful for "
            "brainstorming sessions or when you need to generate a range of "
            "ideas that balance practicality and creativity."
        ),
        name="Exploration Module",
        code="""
async def run(self):
    explore_module = self.agent.create_sub_agent(['ideas'], 'Explore Module')
    critique_module = self.agent.create_sub_agent(['critique'], 'Critique Module')

    temperature = 0.05
    ideas = []

    for i in range(5):  # 5 iterations to get 10 ideas
        if i == 0:
            instruction = (
                "Generate the most conservative small step towards exploring "
                "this idea."
            )
        else:
            instruction = (
                f"Using the previous critique as context, generate 2 ideas to "
                "explore this concept."
            )

        new_ideas_response = await explore_module.llm_response(
            instruction, temperature=temperature
        )
        new_ideas = new_ideas_response.content.split('\\n')
        ideas.extend(new_ideas)

        critique_temp = 1 - temperature
        critique_instruction = (
            "Act as an enthusiastic and creative critic. Think bigger and "
            "offer ways to move forward dynamically."
        )
        critique_response = await critique_module.llm_response(
            critique_instruction + '\\n' + '\\n'.join(new_ideas),
            temperature=critique_temp
        )

        temperature = min(temperature + 0.1, 0.5)

    return '\\n'.join(ideas)
""",
    ),
]


class ChainOfThought:
    @staticmethod
    def apply(prompt: str, intermediate_steps: int = 3) -> str:
        """Apply the Chain of Thought technique to a given prompt.

        :param prompt: The original prompt
        :param intermediate_steps: Number of intermediate reasoning steps
        :return: A modified prompt that encourages step-by-step reasoning
        """
        return f"""
        {prompt}

        Let's approach this step-by-step:

        1. [First step of reasoning]
        2. [Second step of reasoning]
        3. [Third step of reasoning]

        Now, based on these steps, let's formulate the final answer.
        """


class TreeOfThoughts:
    @staticmethod
    def apply(prompt: str, branches: int = 3, depth: int = 2) -> list[str]:
        """Apply the Tree of Thoughts technique to a given prompt.

        :param prompt: The original prompt
        :param branches: Number of alternative thoughts at each step
        :param depth: Number of levels in the thought tree
        :return: A list of prompts representing different paths of reasoning
        """
        thought_tree = [prompt]

        for _ in range(depth):
            new_level = []
            for thought in thought_tree:
                for i in range(branches):
                    new_thought = f"""
                    {thought}

                    Let's explore a new line of thinking:

                    [Alternative thought {i + 1}]

                    Based on this, we can reason further:
                    """
                    new_level.append(new_thought)
            thought_tree = new_level

        return thought_tree


# Add more advanced reasoning techniques here as needed
