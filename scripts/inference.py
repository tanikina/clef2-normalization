import argparse
from src.models import model_factory
from src.prompts import prompt_factory 
import os
import pandas as pd
import yaml
import json
import evaluate
import glob
import random

def evaluate_outputs(predictions, references):
    meteor = evaluate.load('meteor')
    return meteor.compute(predictions=predictions, references=references)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--data_type', type=str, help='Type of data: "clean" or "original"')
    parser.add_argument('--num_demos', type=str, help='Number of demonstrations')
    parser.add_argument('--partition', type=str, help='Dev or test partition')
    parser.add_argument('--lang', type=str, help='"Multi" for multilingual or "sep" for separate languages')
    
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

    # Load prompt template
    prompt = prompt_factory(
        **config['prompt']
    )

    # Load data and do inference
    num_demos = int(args.num_demos)

    data_type = args.data_type
    partition = args.partition
    lang_type = args.lang

    if data_type == 'clean':
        data_type = './clean_data/'
        data_folder = data_type+partition+'/'
        flag = "clean post"
    else:
        data_type = './data/'
        data_folder = data_type+partition+'/'
        flag = "post"

    if lang_type == 'multi':
        sim_type = 'similarities multi'
    else:
        sim_type = 'similarities'

    # make global reference list to extract demonstrations from all the languages in train
    demo_posts = []
    for csv_file in sorted(glob.glob(data_type+"train/"+"*.csv")): 
        with open(csv_file, "r") as g:
            df = pd.read_csv(csv_file)
            demo_posts.extend(zip(df[flag].tolist(), df['normalized claim'].tolist()))
            
    for data_path in glob.glob(data_folder+"*.*"):

        lang = data_path.split('-')[-1].split('.')[0].strip()

        if num_demos == 0:
            # zero-shot, use .csv files
            if data_path.endswith('.jsonlines'):
                continue
            df = pd.read_csv(data_path)
            posts = df[flag].tolist()
            gold_claims = df['normalized claim'].tolist()

            posts = [
                prompt(post=post)
                for post in posts
            ]

        else:
            # few-shot, use .jsonlines, as they contain info about most similar examples
            if data_path.endswith('.csv'):
                continue
            df = pd.read_csv(data_path.replace('.jsonlines', '.csv'))
            posts = []
            gold_claims = []

            # make reference dict from train partition (same lang) to extract demonstrations data_type+"/train/train-"
            data_path_train = data_type+"train/train-"+data_path.replace('.jsonlines', '.csv').split("-")[-1] # csv
            
            if lang_type != 'multi':
                # overwrite demo_posts with post for the current language
                with open(data_path_train, "r") as g:
                    #print(data_path, data_path_train)
                    df = pd.read_csv(data_path_train)
                    demo_posts = list(zip(df[flag].tolist(), df['normalized claim'].tolist()))

            # create prompts with demonstrations
            with open(data_path, "r") as f:
                for line in f.readlines():
                    line = json.loads(line)
                    gold_claims.append(line['normalized claim'])
                    curr_post = line[flag]
                    sims = line[sim_type] # use either from the same language ("similarities") or global ("similarities multi")

                    exs = ""
                    for i in range(num_demos):
                        sim_id = sims[i] # use most similar demonstrations; int
                        #sim_id = random.sample(sims, 1)[0] # use random demonstrations
                        sim_post, sim_norm = demo_posts[sim_id] # use either local similar posts or global ones
                        exs += "\n\nSocial media post: "+sim_post+"\n\nNormalized claim: "+sim_norm

                    posts.append(prompt(examples=exs, post=curr_post))
                    #print("PROMPT:", posts[-1])

        print('LANGUAGE:', lang)
        print()

        repeat = 3 # repeat N times, calculate average METEOR

        best_generated_outputs = []
        best_meteor = 0
        meteors = []

        for k in range(repeat):

            generated_outputs = model.infere(posts)

            # evaluate
            meteor = evaluate_outputs(generated_outputs, gold_claims)['meteor']
            meteors.append(meteor)
            print("RUN", str(k+1))
            print("METEOR:", meteor)
            print()
            if meteor > best_meteor:
                best_meteor = meteor
                best_generated_outputs = generated_outputs

        print("Avg METEOR: {:.3f}".format(sum(meteors)/len(meteors)))
        print()
        
        # Save generated outputs
        df['generated_output'] = best_generated_outputs
        
        output_path = data_path.split('/')[-1].replace('.jsonlines', '.csv')
        model_name = config['model']['name'].split('/')[-1]
        if 'clean_data' in data_path:
            os.makedirs(f'./results/{model_name}/clean_data', exist_ok=True)
            output_path = f'{num_demos}-shot.sim.{output_path}'
            df = df.to_csv(f'./results/{model_name}/clean_data/{output_path}', index=False)
        else:
            os.makedirs(f'./results/{model_name}/orig_data', exist_ok=True)
            output_path = f'{num_demos}-shot.sim.{output_path}'
            df = df.to_csv(f'./results/{model_name}/orig_data/{output_path}', index=False)
    

if __name__ == '__main__':
    main()