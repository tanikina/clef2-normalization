import logging
from typing import List, Union
from src.models.model import Model
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMModel(Model):
    def __init__(
        self,
        name: str,
        max_tokens: int = 512,
        do_sample: bool = False,
        device_map: str = "auto",
        quantization: bool = False,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(name='VLLMModel', max_tokens=max_tokens)
        self.name = name
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.device_map = device_map
        self.quantization = quantization
        self.temperature = temperature
        
        self.system_prompt = kwargs.get("system_prompt", None)
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llm = None
        self.load()
        
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        
        if self.quantization:
            self.llm = LLM(
                model=self.name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            )
        else:
            self.llm = LLM(
                model=self.name,
                trust_remote_code=True,
            )
            
    def _is_chat(self) -> bool:
        return hasattr(self.tokenizer, 'chat_template')
    
    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
    
    def _get_system_role(self) -> str:
        """
        Get the system role.
        """
        if 'gemma' in self.name:
            return None
        else:
            return 'system'
    
    def infere(self, prompt: Union[str, List[str]], max_tokens: int = None) -> Union[str, List[str]]:
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        if self.llm is None:
            self.load()
            
        is_list = isinstance(prompt, list)
        if not is_list:
            prompt = [prompt]
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        
        answers = []
        for p in tqdm(prompt, desc="Generating text", total=len(prompt)):
            if self._is_chat():
                if self.system_prompt and self._get_system_role() == "system":
                    messages = [
                        { "role": "system", "content": self.system_prompt },
                        { "role": "user", "content": p }
                    ]
                elif self.system_prompt:
                    messages = [
                        { "role": "user", "content": f'{self.system_prompt}\n\n{p}' }
                    ]
                else:
                    messages = [
                        { "role": "user", "content": p }
                    ]
                    
                print(messages)
                
                output = self.llm.chat(messages, sampling_params)
                print(output)
                output = output[0].outputs[0].text    
            else:
                if self.system_prompt is not None:
                    p = f'{self.system_prompt}\n\n{p}'
                
                output = self.llm.generate(p, sampling_params)
                print(output)
                output = output[0].outputs[0].text
        
            answers.append(output.strip())
        
        if not is_list:
            answers = answers[0]
        
        return answers
