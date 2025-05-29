from typing import List, Union
from src.models.model import Model
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFModel(Model):
    """
    A class to represent a Hugging Face model.
    
    Attributes:
        name (str): The name of the model.
        max_tokens (int): The maximum number of tokens to generate.
        do_sample (bool): Whether to use sampling.
        device_map (str): The device map.
        quantization (bool): Whether to load the model in 4-bit.
        offload_folder (str): The offload folder.
        offload_state_dict (bool): Whether to offload the state dict.
        max_memory (Any): The maximum memory to use.
        system_prompt (str): The system prompt to use.    
    """
    def __init__(
        self, 
        name: str = 'meta-llama/Llama-3.1-8B-Instruct',
        max_tokens: int = 128,
        do_sample: bool = False, 
        device_map: str = 'auto', 
        quantization: bool = False,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(name='HFModel', max_tokens=max_tokens)
        self.model_name = name
        self.tokenizer = None
        self.model = None
        self.device_map = eval(device_map) if '{' in device_map else device_map
        self.do_sample = do_sample
        self.quantization = quantization
        self.temperature = temperature
        if 'offload_folder' not in kwargs:
            self.offload_folder = None
            self.offload_state_dict = None
        else:
            self.offload_folder = kwargs['offload_folder']
            self.offload_state_dict = kwargs['offload_state_dict']
            
        if 'max_memory' not in kwargs:
            self.max_memory = None
        else:
            self.max_memory = eval(kwargs['max_memory']) if kwargs['max_memory'] else None
        self.system_prompt = kwargs['system_prompt'] if kwargs['system_prompt'] != 'None' else None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.enable_thinking = kwargs.get('enable_thinking', True)
        self.top_k = kwargs.get('top_k', 50)
        self.top_p = kwargs.get('top_p', 1.0)
        self.min_p = kwargs.get('min_p', None)
        self.load()
        
    def _load_model(self) -> None:
        """
        Load the model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map = self.device_map,
            offload_folder = self.offload_folder,
            offload_state_dict = self.offload_state_dict,
            max_memory = self.max_memory
        )
        
    def load_quantized_model(self) -> None:
        """
        Load the quantized model.
        """
        print(f'Loading quantized model - {self.model_name}')
        if 'gemma-2-27b-it' in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.quantization,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map=self.device_map, 
                quantization_config=quantization_config
            )

    def load(self) -> 'HFModel':
        """
        Load the Hugging Face model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.quantization:
            self.load_quantized_model()
        else:
            self._load_model()

        logging.log(
            logging.INFO, f'Loaded model and tokenizer from {self.model_name}')
        
    def _is_chat(self) -> bool:
        """
        Check if the model is a chat model.
        """
        return hasattr(self.tokenizer, 'chat_template')
    
    def _get_system_role(self) -> str:
        """
        Get the system role.
        """
        # if 'gemma' in self.model_name or 'mistral' in self.model_name:
        #     return None
        # else:
        return 'system'
    
    def _terminators(self) -> List[int]:
        """
        Get the terminators.
        """
        if 'Llama-3' in self.model_name:
            return [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            return [
                self.tokenizer.eos_token_id
            ]
            
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt.
        
        Args:
            system_prompt (str): The system prompt to set.
        """
        self.system_prompt = system_prompt

    def infere(self, prompt: Union[str, List[str]], max_tokens: int = None) -> Union[str, List[str]]:
        """
        Generate text based on the prompt.
        
        Args:
            prompt (Union[str, List[str]]): The prompt to generate text from.
            
        Returns:
            str: The generated text.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        is_list = True
        if isinstance(prompt, str):
            prompt = [prompt]
            is_list = False
        
        answers = []
        for p in tqdm(prompt, desc="Generating text", total=len(prompt)):
            if self._is_chat():
                if self.system_prompt and self._get_system_role() == 'system':
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": p}
                    ]
                elif self.system_prompt:
                    messages = [
                        {"role": "user", "content": f'{self.system_prompt}\n\n{p}'}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": p}
                    ]
                
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    return_tensors='pt',
                    enable_thinking=self.enable_thinking,
                ).to(self.device)

            else:
                if 'falcon-40b' in self.model_name:
                    if self.system_prompt is not None:
                        p = f'{self.system_prompt}\nUser: {p}\nFalcon:'
                    else:
                        p = f'User: {p}\nFalcon:'
                    
                elif self.system_prompt is not None:
                    p = f'{self.system_prompt}\n\n{p}'

                inputs = self.tokenizer(
                    p, 
                    return_tensors='pt'
                ).to(self.device)['input_ids']
            
            generated_output = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_tokens,
                eos_token_id=self._terminators(),
                return_dict_in_generate=True,
                output_logits=True,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
            )
            generated_ids = generated_output.sequences
            
            decoded_input = self.tokenizer.batch_decode(
                inputs, 
                skip_special_tokens=True
            )[0]
            
            decoded = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            decoded = decoded[len(decoded_input):]
            answers.append(decoded.strip())
        
        if not is_list:
            answers = answers[0]
        
        return answers
