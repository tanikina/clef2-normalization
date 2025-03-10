from typing import Any

class Model:
    """
    The base class for all models.
    
    Args:
        name (str): The name of the model.
        max_tokens (int): The maximum number of new tokens.
    """
    def __init__(self, name: str, max_tokens: int, **kwargs):
        self.max_tokens = max_tokens

    def infere(self, prompt: str):
        raise NotImplementedError
    
    
    def __call__(self, **kwargs: Any) -> Any:
        return self.infere(**kwargs)
