import argparse
from src.models import model_factory
from src.prompts import prompt_factory  
from src.evaluation import evaluate_outputs
import os
import pandas as pd
import yaml
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--data_path', type=str, help='Path to data')
    parser.add_argument('--prompt_name', type=str, help='Prompt name')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # initialize wandb
    wandb.init(
        project='clef2025',
        config=config
    )
    
    # name the wandb run with the model name prompt name and language
    # get the language from ./data/train/train-{language}.csv data path
    language = args.data_path.split('/')[-1].split('-')[-1].replace('.csv', '')
    wandb.run.name = f"{config['model']['name'].split('/')[-1]}_{args.prompt_name}_{language}"
        
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
    df['prompt'] = posts
    
    output_path = args.data_path.split('/')[-1].replace('.csv', '_generated.csv')
    model_name = config['model']['name'].split('/')[-1]
    os.makedirs(f'./results/{model_name}', exist_ok=True)
    os.makedirs(f'./results/{model_name}/{args.prompt_name}', exist_ok=True)
    df.to_csv(f'./results/{model_name}/{args.prompt_name}/{output_path}', index=False)
    
    # save dataframe to wandb
    wandb.log({'generated_outputs': wandb.Table(dataframe=df)})
    
    # evaluate the results
    metrics = evaluate_outputs(
        df['generated_output'].tolist(),
        df['normalized claim'].tolist()
    )
    
    # log metrics
    wandb.log(metrics)
    
    print(metrics)
    # save metrics
    with open(f'./results/{model_name}/{args.prompt_name}/metrics.txt', 'w') as f:
        f.write(str(metrics))
        
    wandb.finish()
    

if __name__ == '__main__':
    main()