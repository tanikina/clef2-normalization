from src.prompts.prompt import Prompt

def prompt_factory(prompt_type: str, **kwargs) -> Prompt:
    return {
        'prompt': Prompt,
    }[prompt_type](**kwargs)
