import random


def mutate_config(cfg: dict) -> dict:
    new = cfg.copy()
    if random.random() < 0.2:
        new['num_layers'] = max(1, new.get('num_layers', 1) + random.choice([-1, 1]))
    if random.random() < 0.3:
        new['hidden_size'] = int(new.get('hidden_size', 128) * random.uniform(0.9, 1.1))
    if random.random() < 0.1:
        new['num_experts'] = max(1, new.get('num_experts', 1) + random.choice([-1, 1]))
    return new


def select_best(parent, candidates, eval_fn):
    parent_score = eval_fn(parent)
    best = parent
    best_score = parent_score
    for c in candidates:
        s = eval_fn(c)
        if s > best_score:
            best = c
            best_score = s
    return best
