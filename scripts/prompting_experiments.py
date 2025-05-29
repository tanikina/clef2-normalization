import os
import yaml

# set VLLM_WORKER_MULTIPROC_METHOD=spawn: https://github.com/vllm-project/vllm/issues/8893
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

MODELS = [
    {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "quantization": True,
        "model_type": "vllm"
    },
    {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "quantization": True,
        "model_type": "vllm"
    },
    {
        "name": "google/gemma-3-27b-it",
        "quantization": True,
        "model_type": "vllm"
    },
    {
        "name": "Qwen/Qwen3-32B",
        "quantization": True,
        "model_type": "huggingface"
    },
    {
        "name": "Qwen/Qwen3-8B",
        "quantization": False,
        "model_type": "huggingface"
    },
]

DEV_LANGUAGES = [
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

TEST_LANGUAGES = [
    'bn',
    'ces',
    'ell',
    'kor',
    'nld',
    'ron',
    'te',
]

CONFIGS = [
    {
        'config_path': './configs/final/zero-shot.yaml',
        'prompt_name': 'zero-shot',
        'num_shots': 0,
    },    
    {
        'config_path': './configs/final/fewshot.yaml',
        'prompt_name': 'fewshot-1',
        'num_shots': 1,
    },
    {
        'config_path': './configs/final/fewshot.yaml',
        'prompt_name': 'fewshot-2',
        'num_shots': 2,
    },
    {
        'config_path': './configs/final/fewshot.yaml',
        'prompt_name': 'fewshot-5',
        'num_shots': 5,
    },
    {
        'config_path': './configs/final/fewshot.yaml',
        'prompt_name': 'fewshot-10',
        'num_shots': 10,
    },
]

TRANSLATED_CONFIGS = [
    {
        'config_path': './configs/final/translated_zs/',
        'prompt_name': 'translated-zero-shot',
        'num_shots': 0,
    },
    {
        'config_path': './configs/final/translated_fs/',
        'prompt_name': 'translated-fewshot-1',
        'num_shots': 1,
    },
    {
        'config_path': './configs/final/translated_fs/',
        'prompt_name': 'translated-fewshot-2',
        'num_shots': 2,
    },
    {
        'config_path': './configs/final/translated_fs/',
        'prompt_name': 'translated-fewshot-5',
        'num_shots': 5,
    },
    {
        'config_path': './configs/final/translated_fs/',
        'prompt_name': 'translated-fewshot-10',
        'num_shots': 10,
    },
]



def main():
    for model in MODELS:
        for con in CONFIGS:
            with open(con['config_path'], 'r') as f:
                config = yaml.safe_load(f)
                    
            config['model']['name'] = model['name']
            config['model']['quantization'] = model['quantization']
            config['model']['model_type'] = model['model_type']
            
            model_name = model['name'].split('/')[1]
            with open(f'./configs/final/{con["prompt_name"]}-{model_name}.yaml', 'w', encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True)
            
            for split in ['dev', 'test']:
                # original data
                os.system(f'python -m scripts.inference_all --config ./configs/final/{con["prompt_name"]}-{model_name}.yaml --prompt_name {con["prompt_name"]} --num_shots {con["num_shots"]} --lang_type multi --data_type ./data --data_split {split}')
                # filtered data
                if con["num_shots"] != 0:
                    os.system(f'python -m scripts.inference_all --config ./configs/final/{con["prompt_name"]}-{model_name}.yaml --prompt_name filtered-{con["prompt_name"]} --num_shots {con["num_shots"]} --lang_type multi --data_type ./filtered --data_split {split}')

            os.remove(f'./configs/final/{con["prompt_name"]}-{model_name}.yaml')

        for con in TRANSLATED_CONFIGS:
            for language in DEV_LANGUAGES + TEST_LANGUAGES:
                if 'translated_zs' in con['config_path']:
                    config_path = con['config_path'] + f'zero-shot-{language}'
                else:
                    config_path = con['config_path'] + f'fewshot-{language}'
                    
                with open(f'{config_path}.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                        
                config['model']['name'] = model['name']
                config['model']['quantization'] = model['quantization']
                config['model']['model_type'] = model['model_type']
                
                model_name = model['name'].split('/')[1]
                with open(f'{config_path}-{model_name}.yaml', 'w', encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True)
                
                for split in ['dev', 'test']:
                    if os.path.exists(f'./results/{model_name}/{con["prompt_name"]}/{split}-{language}_generated.csv'):
                        print(f'File already exists: ./results/{model_name}/{con["prompt_name"]}/{split}-{language}_generated.csv')
                        os.remove(f'{config_path}-{model_name}.yaml')
                        continue
                    
                    if split == 'test' and language in DEV_LANGUAGES:
                        continue
                    # original data
                    os.system(f'python -m scripts.inference --config {config_path}-{model_name}.yaml --data_path ./data/{split}/{split}-{language}.csv --data_split {split} --prompt_name {con["prompt_name"]} --num_shots {con["num_shots"]} --lang_type multi')
                    # filtered data
                    if con["num_shots"] != 0:
                        os.system(f'python -m scripts.inference --config {config_path}-{model_name}.yaml --data_path ./filtered/{split}/{split}-{language}.csv --data_split {split} --prompt_name filtered-{con["prompt_name"]} --num_shots {con["num_shots"]} --lang_type multi')


if __name__ == '__main__':
    main()