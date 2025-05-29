import argparse
from src.models import model_factory
from src.prompts import prompt_factory  
import os
import pandas as pd
import yaml
import glob
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--data_split', type=str, help='Data split')
    parser.add_argument('--prompt_name', type=str, help='Prompt name')
    parser.add_argument('--language', type=str, help='Language')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots')
    parser.add_argument('--lang_type', type=str, help='"multi" for multilingual or "sep" for separate languages')
    parser.add_argument('--data_type', type=str, default='./data', help='Path to data type (default: ./data)')
    
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
    
    paths = glob.glob(f'./{args.data_type}/{args.data_split}/*.csv')
    
    for path in paths:
        model_name = config['model']['name'].split('/')[-1]
        language = path.split('/')[-1].split('-')[-1].replace('.csv', '')
        
        if os.path.exists(f'./results/{model_name}/{args.prompt_name}/{args.data_split}-{language}_generated.csv'):
            print(f'File already exists: ./results/{model_name}/{args.prompt_name}/{args.data_split}-{language}_generated.csv')
            continue
    
        # Load data
        df = pd.read_csv(path)
        if 'clean' in path:
            flag = "clean post"
            data_type = './clean_data/'
        elif 'filtered' in path:
            flag = "post"
            data_type = './filtered/'
        else:
            flag = "post"
            data_type = './data/'
            
        if args.lang_type == 'multi':
            sim_type = 'similarities multi'
        else:
            sim_type = 'similarities'

        # Load prompt template
        prompt = prompt_factory(
            **config['prompt']
        )
        
        demo_posts = []
        if args.num_shots != 0:
            for csv_file in sorted(glob.glob(data_type+"train/"+"*.csv")): 
                with open(csv_file, "r") as g:
                    df_train = pd.read_csv(csv_file)
                    demo_posts.extend(zip(df_train[flag].tolist(), df_train['normalized claim'].tolist()))
        
        posts = []
        
        if args.lang_type != 'multi' and args.num_shots != 0:
            df_train = pd.read_csv(args.data_path.replace(f'{args.data_split}', 'train'))
            demo_posts = list(zip(df_train[flag].tolist(), df_train['normalized claim'].tolist()))
    
        with open(args.data_path.replace('.csv', '.jsonlines'), "r") as f:
            for line in f.readlines():
                line = json.loads(line)
                curr_post = line[flag]
                sims = line[sim_type]
                
                exs = ""
                for i in range(args.num_shots):
                    sim_id = sims[i]
                    sim_post, sim_norm = demo_posts[sim_id]
                    exs += "\n\nPost: "+sim_post+"\n\nNormalized claim: "+sim_norm
                    
                if args.num_shots == 0:
                    posts.append(prompt(post=curr_post))
                else:
                    posts.append(prompt(examples=exs, post=curr_post))
        
        generated_outputs = model.infere(posts)
        
        # Save generated outputs
        df['generated_output'] = generated_outputs
        df['prompt'] = posts
        
        output_path = path.split('/')[-1].replace('.csv', '_generated.csv')
        model_name = config['model']['name'].split('/')[-1]
        os.makedirs(f'./results/{model_name}', exist_ok=True)
        os.makedirs(f'./results/{model_name}/{args.prompt_name}', exist_ok=True)
        df.to_csv(f'./results/{model_name}/{args.prompt_name}/{output_path}', index=False)
    

if __name__ == '__main__':
    main()