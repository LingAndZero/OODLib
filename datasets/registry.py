DATASET_REGISTRY = {}


def register_dataset(name):
    def decorator(obj):
        DATASET_REGISTRY[name] = obj
        return obj
    return decorator


def get_dataset(name, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered")
    return DATASET_REGISTRY[name](**kwargs)


def list_datasets():
    return list(DATASET_REGISTRY.keys())
