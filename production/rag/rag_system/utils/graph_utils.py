from typing import Dict, List


def distance_sensitive_linearization(graph: Dict[str, List[str]], anchor: str) -> List[str]:
    distances = {anchor: 0}
    queue = [(anchor, 0)]
    linearized = []

    while queue:
        node, dist = queue.pop(0)
        linearized.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    return sorted(linearized, key=lambda x: distances[x], reverse=True)
