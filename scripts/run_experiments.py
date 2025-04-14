import os
import yaml

# set VLLM_WORKER_MULTIPROC_METHOD=spawn: https://github.com/vllm-project/vllm/issues/8893
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

MODELS = [
    {
        "name": "meta-llama/Llama-3.1-70B-Instruct",
        "quantization": True
    },
    {
        "name": "google/gemma-3-27b-it",
        "quantization": True
    },
    {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "quantization": True
    },
    {
        "name": "utter-project/EuroLLM-9B-Instruct",
        "quantization": False
    },
    {
        "name": "mistralai/Mistral-Large-Instruct-2411",
        "quantization": True
    },
    {
        "name": "google/gemma-3-12b-it",
        "quantization": True
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "quantization": False
    },
    {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization": False
    },
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "quantization": False
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": False
    },
    {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "quantization": False
    },
]

LANGUAGES = [
    'ara',
    'deu',
    'eng',
    'fra',
    'hi',
    'mr',
    'msa',
    'pa',
    'pol',
    'por',
    'spa',
    'ta',
    'tha'
]

def main():
    for model in MODELS:
        with open('./configs/baseline.yaml', 'r') as f:
            config = yaml.safe_load(f)
                
        config['model']['name'] = model['name']
        config['model']['quantization'] = model['quantization']
        
        print(config['prompt']['template'])
        
        model_name = model['name'].split('/')[1]
        with open(f'./configs/baseline-{model_name}.yaml', 'w') as f:
            yaml.dump(config, f)
                
        for language in LANGUAGES:           
            os.system(f'python -m scripts.inference --config ./configs/baseline-{model_name}.yaml --data_path ./data/dev/dev-{language}.csv --prompt_name baseline')

        os.remove(f'./configs/baseline-{model_name}.yaml')

if __name__ == '__main__':
    main()