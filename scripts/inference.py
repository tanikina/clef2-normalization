import argparse
from src.models import model_factory
from src.prompts import prompt_factory  
import os
import pandas as pd
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--data_path', type=str, help='Path to data')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model = model_factory(
        **config['model']
    )
    # Load data
    df = pd.read_csv(args.data_path)
    posts = df['post'].tolist()

    # Load prompt template
    prompt = prompt_factory(
        **config['prompt']
    )
    posts = [
        prompt(post=post)
        for post in posts
    ]
    generated_outputs = model.infere(posts)
    
    # Save generated outputs
    df['generated_output'] = generated_outputs
    
    output_path = args.data_path.split('/')[-1].replace('.csv', '_generated.csv')
    model_name = config['model']['name'].split('/')[-1]
    os.makedirs(f'./results/{model_name}', exist_ok=True)
    df = df.to_csv(f'./results/{model_name}/{output_path}', index=False)
    

if __name__ == '__main__':
    main()