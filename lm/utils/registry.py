from typing import Callable, Any
opts = {}

class Registry(object):
    def __init__(self, key: str):
        self.key = key
    
    def __call__(self, func: Callable[[], Any]) -> Callable[[], Any]:
        def wrapper(*args, **kwargs) -> Any:
            key = args[0].__class__.__name__
            x = func(*args, **kwargs)
            opts[key] = x
            return x
        return wrapper
