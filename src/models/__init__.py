from src.models.model import Model
from src.models.hf_model import HFModel
#from src.models.vllm_model import VLLMModel

def model_factory(model_type: str, **kwargs) -> Model:
    Model = {
        'huggingface': HFModel,
        #'vllm': VLLMModel
    }[model_type]
    
    return Model(**kwargs)
