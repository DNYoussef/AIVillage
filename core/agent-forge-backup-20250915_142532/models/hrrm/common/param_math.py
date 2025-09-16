# packages/hrrm/common/param_math.py
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_tiny_params(model, lo=48_000_000, hi=55_000_000):
    n = count_params(model)
    if not (lo <= n <= hi):
        raise ValueError(f"Param count {n} outside [{lo},{hi}]")
