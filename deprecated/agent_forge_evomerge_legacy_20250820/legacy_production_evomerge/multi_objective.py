def calculate_pareto_front(objectives: list[list[float]]) -> list[int]:
    """Calculate the Pareto front from a set of objective values.
    Returns the indices of solutions in the Pareto front.
    """
    pareto_front = []
    for i, obj in enumerate(objectives):
        if all(not dominates(other_obj, obj) for j, other_obj in enumerate(objectives) if i != j):
            pareto_front.append(i)
    return pareto_front


def dominates(a: list[float], b: list[float]) -> bool:
    """Check if solution a dominates solution b."""
    return all(a_val >= b_val for a_val, b_val in zip(a, b, strict=False)) and any(
        a_val > b_val for a_val, b_val in zip(a, b, strict=False)
    )


def calculate_crowding_distance(objectives: list[list[float]]) -> list[float]:
    """Calculate crowding distance for solutions in objective space."""
    n_solutions = len(objectives)
    n_objectives = len(objectives[0])
    crowding_distances = [0.0] * n_solutions

    for m in range(n_objectives):
        sorted_indices = sorted(range(n_solutions), key=lambda i: objectives[i][m])
        crowding_distances[sorted_indices[0]] = float("inf")
        crowding_distances[sorted_indices[-1]] = float("inf")

        obj_range = objectives[sorted_indices[-1]][m] - objectives[sorted_indices[0]][m]
        if obj_range == 0:
            continue

        for i in range(1, n_solutions - 1):
            crowding_distances[sorted_indices[i]] += (
                objectives[sorted_indices[i + 1]][m] - objectives[sorted_indices[i - 1]][m]
            ) / obj_range

    return crowding_distances


def nsga2_select(population: list, objectives: list[list[float]], n_select: int) -> tuple[list, list[list[float]]]:
    """Perform NSGA-II selection."""
    fronts = non_dominated_sort(objectives)
    selected = []
    selected_objectives = []

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend([population[i] for i in front])
            selected_objectives.extend([objectives[i] for i in front])
        else:
            crowding_distances = calculate_crowding_distance([objectives[i] for i in front])
            sorted_front = sorted(front, key=lambda i: crowding_distances[front.index(i)], reverse=True)
            n_remaining = n_select - len(selected)
            selected.extend([population[i] for i in sorted_front[:n_remaining]])
            selected_objectives.extend([objectives[i] for i in sorted_front[:n_remaining]])
            break

    return selected, selected_objectives


def non_dominated_sort(objectives: list[list[float]]) -> list[list[int]]:
    """Perform non-dominated sorting of solutions."""
    n_solutions = len(objectives)
    domination_counts = [0] * n_solutions
    dominated_solutions = [[] for _ in range(n_solutions)]
    fronts = [[]]

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            if dominates(objectives[i], objectives[j]):
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_solutions[j].append(i)
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for solution in fronts[i]:
            for dominated in dominated_solutions[solution]:
                domination_counts[dominated] -= 1
                if domination_counts[dominated] == 0:
                    next_front.append(dominated)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove the empty last front
