BASELINE_REGISTRY = {}


def register_baseline(name):
    def decorator(obj):
        BASELINE_REGISTRY[name] = obj
        return obj
    return decorator


def get_baseline(name, **kwargs):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Baseline '{name}' is not registered")
    return BASELINE_REGISTRY[name](**kwargs)


def list_baselines():
    return list(BASELINE_REGISTRY.keys())
