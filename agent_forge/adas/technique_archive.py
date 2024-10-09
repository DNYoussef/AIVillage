# agent_forge/adas/technique_archive.py

PROMPT_TECHNIQUE_ARCHIVE = [
    {
        "thought": "Zero-shot prompting provides direct instructions without examples, allowing for quick task performance without prior demonstrations. This approach leverages the model's pre-trained knowledge to generalize to new tasks.",
        "name": "Zero-Shot Prompting",
        "code": """
def forward(self, taskInfo):
    instruction = "Based on the given task, provide a direct answer without any additional examples or explanation."
    zero_shot_agent = LLMAgentBase(['answer'], 'Zero-Shot Agent')
    answer = zero_shot_agent([taskInfo], instruction)[0]
    return answer
"""
    },
    {
        "thought": "Few-shot prompting includes a small number of examples before the main task, enabling the model to learn from demonstrations. This technique can improve performance with minimal example overhead.",
        "name": "Few-Shot Prompting",
        "code": """
def forward(self, taskInfo):
    few_shot_examples = [
        {"input": "Example input 1", "output": "Example output 1"},
        {"input": "Example input 2", "output": "Example output 2"},
    ]
    instruction = "Here are a few examples of how to approach this type of task:\\n"
    for example in few_shot_examples:
        instruction += f"Input: {example['input']}\\nOutput: {example['output']}\\n\\n"
    instruction += "Now, solve the following task:"
    
    few_shot_agent = LLMAgentBase(['answer'], 'Few-Shot Agent')
    answer = few_shot_agent([taskInfo, Info('examples', 'ADAS', str(few_shot_examples), 0)], instruction)[0]
    return answer
"""
    },
    {
        "thought": "Chain-of-Thought (CoT) prompting encourages the model to show its reasoning steps, enhancing problem-solving capabilities for complex tasks that require multi-step reasoning.",
        "name": "Chain-of-Thought",
        "code": """
def forward(self, taskInfo):
    instruction = "Think through this step-by-step:\\n1. Understand the question\\n2. Identify key information\\n3. Reason through the problem\\n4. Formulate your answer\\nNow, solve the task."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
    thinking, answer = cot_agent([taskInfo], instruction)
    return answer
"""
    },
    {
        "thought": "Zero-Shot Chain-of-Thought adds a thought-inducing phrase to prompt step-by-step reasoning without examples, improving reasoning capabilities in a zero-shot setting.",
        "name": "Zero-Shot Chain-of-Thought",
        "code": """
def forward(self, taskInfo):
    instruction = "Let's approach this step-by-step:\\n1. Understand the question\\n2. Identify key information\\n3. Reason through the problem\\n4. Formulate your answer\\nNow, solve the task."
    zero_shot_cot_agent = LLMAgentBase(['thinking', 'answer'], 'Zero-Shot CoT Agent')
    thinking, answer = zero_shot_cot_agent([taskInfo], instruction)
    return answer
"""
    },
    {
        "thought": "Self-Consistency generates multiple Chain-of-Thought paths and selects the majority answer, increasing answer reliability through an ensemble method.",
        "name": "Self-Consistency",
        "code": """
def forward(self, taskInfo):
    instruction = "Solve the given task using step-by-step reasoning. Provide your final answer at the end."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'CoT Agent', temperature=0.7)
    
    num_iterations = 5
    answers = []
    for _ in range(num_iterations):
        thinking, answer = cot_agent([taskInfo], instruction)
        answers.append(answer.content)
    
    from collections import Counter
    most_common_answer = Counter(answers).most_common(1)[0][0]
    
    return Info('answer', 'Self-Consistency Agent', most_common_answer, 0)
"""
    },
    {
        "thought": "Least-to-Most Prompting breaks complex problems into simpler sub-problems, solving them sequentially. This approach helps in handling complex tasks more effectively by decomposing them into manageable parts.",
        "name": "Least-to-Most Prompting",
        "code": """
def forward(self, taskInfo):
    decomposition_agent = LLMAgentBase(['sub_tasks'], 'Decomposition Agent')
    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Solving Agent')
    
    decomp_instruction = "Break down the given task into a list of simpler sub-tasks, ordered from least to most complex."
    sub_tasks = decomposition_agent([taskInfo], decomp_instruction)[0]
    
    context = [taskInfo]
    for i, sub_task in enumerate(sub_tasks.content.split('\\n')):
        solve_instruction = f"Solve this sub-task: {sub_task}. Use previous solutions if relevant."
        thinking, answer = solving_agent(context, solve_instruction)
        context.extend([thinking, answer])
    
    return answer  # The last answer is the solution to the most complex sub-task, i.e., the original task
"""
    },
    {
        "thought": "Tree-of-Thoughts creates a tree-like search of multiple reasoning paths, improving search and planning capabilities for complex problem-solving.",
        "name": "Tree-of-Thoughts",
        "code": """
def forward(self, taskInfo):
    def expand_node(node, depth):
        if depth == 0:
            return node
        
        instruction = f"Given the current thought: '{node}', generate three possible next steps in reasoning."
        expansion_agent = LLMAgentBase(['step1', 'step2', 'step3'], 'Expansion Agent')
        steps = expansion_agent([taskInfo, Info('node', 'Tree', node, 0)], instruction)
        
        children = [expand_node(step.content, depth-1) for step in steps]
        return (node, children)

    root_instruction = "Provide an initial thought about how to approach this task."
    root_agent = LLMAgentBase(['initial_thought'], 'Root Agent')
    root_thought = root_agent([taskInfo], root_instruction)[0].content

    tree = expand_node(root_thought, depth=2)

    evaluation_instruction = "Evaluate the following reasoning path and provide a final answer based on it."
    evaluation_agent = LLMAgentBase(['evaluation', 'answer'], 'Evaluation Agent')

    def evaluate_path(path):
        eval_input = [taskInfo] + [Info('step', 'Tree', step, i) for i, step in enumerate(path)]
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
        score = float(answer.content.split('|')[0])  # Assuming answer format is "score|actual_answer"
        if score > best_score:
            best_score = score
            best_answer = answer.content.split('|')[1]

    return Info('answer', 'Tree-of-Thoughts Agent', best_answer, 0)
"""
    },
    {
        "thought": "Program-of-Thoughts generates programming code as reasoning steps, excelling in mathematical and programming tasks by leveraging code-based reasoning.",
        "name": "Program-of-Thoughts",
        "code": """
def forward(self, taskInfo):
    code_generation_agent = LLMAgentBase(['code'], 'Code Generation Agent')
    execution_agent = LLMAgentBase(['result'], 'Code Execution Agent')
    interpretation_agent = LLMAgentBase(['interpretation', 'answer'], 'Interpretation Agent')
    
    gen_instruction = "Generate Python code to solve the given task. Include print statements to show intermediate steps."
    code = code_generation_agent([taskInfo], gen_instruction)[0]
    
    exec_instruction = "Execute the following code and provide the output:"
    result = execution_agent([taskInfo, code], exec_instruction)[0]
    
    interp_instruction = "Interpret the code execution results and provide a final answer to the original task."
    interpretation, answer = interpretation_agent([taskInfo, code, result], interp_instruction)
    
    return answer
"""
    },
    {
        "thought": "Prompt Chaining uses multiple prompts in succession to handle complex multi-step tasks, allowing for a more structured approach to problem-solving.",
        "name": "Prompt Chaining",
        "code": """
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
"""
    },
    {
        "thought": "Role Prompting assigns a specific role to the AI, potentially improving task-specific outputs by leveraging persona-based reasoning.",
        "name": "Role Prompting",
        "code": """
def forward(self, taskInfo):
    role_selection_agent = LLMAgentBase(['role'], 'Role Selection Agent')
    role_instruction = "Based on the task, select the most appropriate role from: Mathematician, Physicist, Computer Scientist, or General Problem Solver."
    selected_role = role_selection_agent([taskInfo], role_instruction)[0].content

    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Role-based Solving Agent')
    solve_instruction = f"As a {selected_role}, approach this task step-by-step and provide your answer."
    thinking, answer = solving_agent([taskInfo, Info('role', 'ADAS', selected_role, 0)], solve_instruction)

    return answer
"""
    },
    {
        "thought": "Emotion Prompting incorporates emotional phrases to potentially improve performance on benchmarks by leveraging psychological relevance.",
        "name": "Emotion Prompting",
        "code": """
def forward(self, taskInfo):
    emotion_agent = LLMAgentBase(['emotion'], 'Emotion Selection Agent')
    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Emotional Solving Agent')
    
    emotion_instruction = "Based on the given task, suggest an appropriate emotion that might help in solving it effectively."
    emotion = emotion_agent([taskInfo], emotion_instruction)[0]
    
    solving_instruction = f"Imagine you're feeling {emotion.content}. With this emotional state in mind, solve the given task. Show your reasoning and provide an answer."
    thinking, answer = solving_agent([taskInfo, emotion], solving_instruction)
    
    return answer
"""
    },
    {
        "thought": "Self-Ask prompts the model to ask and answer follow-up questions, improving problem decomposition and solving through self-questioning.",
        "name": "Self-Ask",
        "code": """
def forward(self, taskInfo):
    question_agent = LLMAgentBase(['question'], 'Question Agent')
    answer_agent = LLMAgentBase(['answer'], 'Answer Agent')
    
    context = [taskInfo]
    max_iterations = 5
    
    for _ in range(max_iterations):
        q_instruction = "Based on the current context, what's the next question we should ask to solve the task? If no more questions are needed, respond with 'DONE'."
        question = question_agent(context, q_instruction)[0]
        
        if question.content == 'DONE':
            break
        
        a_instruction = f"Answer the following question: {question.content}"
        answer = answer_agent(context + [question], a_instruction)[0]
        
        context.extend([question, answer])
    
    final_answer_agent = LLMAgentBase(['final_answer'], 'Final Answer Agent')
    final_instruction = "Based on all the questions and answers, provide the final answer to the original task."
    final_answer = final_answer_agent(context, final_instruction)[0]
    
    return final_answer
"""
    },
    {
        "thought": "Contrastive Chain-of-Thought includes both correct and incorrect explanations to enhance reasoning by showing what not to do, leveraging contrast learning.",
        "name": "Contrastive Chain-of-Thought",
        "code": """
def forward(self, taskInfo):
    cot_agent = LLMAgentBase(['correct_thinking', 'incorrect_thinking'], 'CoT Generation Agent')
    contrast_agent = LLMAgentBase(['analysis', 'answer'], 'Contrast Analysis Agent')
    
    cot_instruction = "Provide both a correct and an incorrect chain of thought for solving the given task."
    correct_thinking, incorrect_thinking = cot_agent([taskInfo], cot_instruction)
    
    contrast_instruction = "Analyze the correct and incorrect reasoning paths. Explain the differences and provide the final correct answer."
    analysis, answer = contrast_agent([taskInfo, correct_thinking, incorrect_thinking], contrast_instruction)
    
    return answer
"""
    },
    {
        "thought": "Memory-of-Thought uses unlabeled data to build Few-Shot Chain-of-Thought prompts, improving performance on various reasoning tasks through dynamic example retrieval.",
        "name": "Memory-of-Thought",
        "code": """
def forward(self, taskInfo):
    memory_bank = [...]  # Assume this is a large list of unlabeled examples
    
    retrieval_agent = LLMAgentBase(['relevant_examples'], 'Retrieval Agent')
    solving_agent = LLMAgentBase(['thinking', 'answer'], 'Solving Agent')
    
    retrieval_instruction = "Select the top 3 most relevant examples from the memory bank for solving the given task."
    relevant_examples = retrieval_agent([taskInfo, Info('memory', 'Memory Bank', str(memory_bank), 0)], retrieval_instruction)[0]
    
    solving_instruction = "Use the provided relevant examples as a guide to solve the given task. Show your reasoning."
    thinking, answer = solving_agent([taskInfo, relevant_examples], solving_instruction)
    
    return answer
"""
    },
]