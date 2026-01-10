MODEL_REGISTRY = {}


def register_model(name):
    def decorator(obj):
        MODEL_REGISTRY[name] = obj
        return obj
    return decorator


def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    return list(MODEL_REGISTRY.keys())
