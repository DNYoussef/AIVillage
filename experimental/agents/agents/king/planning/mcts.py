import asyncio
from collections import defaultdict
import math
import random


class MCTSNode:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


class MCTS:
    def __init__(self, exploration_weight=1.0, max_depth=10) -> None:
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.stats = defaultdict(lambda: {"visits": 0, "value": 0})

    async def search(self, task, problem_analyzer, plan_generator, iterations=1000):
        root = MCTSNode(task)

        for _ in range(iterations):
            node = self.select(root)
            if node.visits < 1 or len(node.children) == 0:
                child = await self.expand(node, problem_analyzer)
            else:
                child = self.best_uct_child(node)
            result = await self.simulate(child, plan_generator)
            self.backpropagate(child, result)

        return self.best_child(root).state

    def select(self, node):
        path = []
        while True:
            path.append(node)
            if not node.children or len(path) > self.max_depth:
                return node
            unexplored = [child for child in node.children if child.visits == 0]
            if unexplored:
                return random.choice(unexplored)
            node = self.best_uct_child(node)

    async def expand(self, node, problem_analyzer):
        if problem_analyzer:
            new_states = await problem_analyzer.generate_possible_states(node.state)
        else:
            new_states = [
                node.state
            ]  # Placeholder for when problem_analyzer is not provided
        for state in new_states:
            if state not in [child.state for child in node.children]:
                new_node = MCTSNode(state, parent=node)
                node.children.append(new_node)
                return new_node
        return random.choice(node.children)

    async def simulate(self, node, plan_generator):
        return await plan_generator.evaluate(node.state)

    def backpropagate(self, node, result) -> None:
        while node:
            self.stats[node.state]["visits"] += 1
            self.stats[node.state]["value"] += result
            node.visits += 1
            node.value += result
            node = node.parent

    def best_uct_child(self, node):
        log_n_visits = math.log(self.stats[node.state]["visits"])
        return max(
            node.children,
            key=lambda c: (self.stats[c.state]["value"] / self.stats[c.state]["visits"])
            + self.exploration_weight
            * math.sqrt(log_n_visits / self.stats[c.state]["visits"]),
        )

    def best_child(self, node):
        return max(node.children, key=lambda c: self.stats[c.state]["visits"])

    async def update(self, task, result) -> None:
        # Update MCTS statistics based on task execution results
        self.stats[task]["visits"] += 1
        self.stats[task]["value"] += result

    async def prune(self, node, threshold) -> None:
        node.children = [
            child
            for child in node.children
            if self.stats[child.state]["visits"] > threshold
        ]
        for child in node.children:
            await self.prune(child, threshold)

    async def parallel_search(
        self, task, problem_analyzer, plan_generator, iterations=1000, num_workers=4
    ):
        root = MCTSNode(task)
        semaphore = asyncio.Semaphore(num_workers)

        async def worker() -> None:
            async with semaphore:
                node = self.select(root)
                if node.visits < 1 or len(node.children) == 0:
                    child = await self.expand(node, problem_analyzer)
                else:
                    child = self.best_uct_child(node)
                result = await self.simulate(child, plan_generator)
                self.backpropagate(child, result)

        await asyncio.gather(*[worker() for _ in range(iterations)])
        return self.best_child(root).state
