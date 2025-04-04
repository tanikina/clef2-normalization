import os
import yaml

# set VLLM_WORKER_MULTIPROC_METHOD=spawn: https://github.com/vllm-project/vllm/issues/8893
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

MODELS = [
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": False
    },
    {
        "name": "google/gemma-3-4b-it",
        "quantization": False
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
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
        print(f'./configs/baseline-{model_name}.yaml')
        with open(f'./configs/baseline-{model_name}.yaml', 'w') as f:
            yaml.dump(config, f)
                
        for language in LANGUAGES:           
            os.system(f'python -m scripts.inference --config ./configs/baseline-{model_name}.yaml --data_path ./data/train/train-{language}.csv --prompt_name baseline')

        os.remove(f'./configs/baseline-{model_name}.yaml')

if __name__ == '__main__':
    main()