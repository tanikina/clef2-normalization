import glob
import os
import json
import torch
import pandas as pd
from scripts.cleanup_utils import *
import sentence_transformers 
from sentence_transformers import SentenceTransformer

def cleanup_files(input_path_to_folder, output_path_to_folder):

    if not os.path.exists(output_path_to_folder):
        os.makedirs(output_path_to_folder)

    for csv_file in glob.glob(input_path_to_folder+"*.csv"):
        file_name = csv_file.split("/")[-1]
        lang = csv_file.split(".")[0].split("-")[-1]
        df = pd.read_csv(csv_file)
        #print(df)
        clean_posts = []
        for post in df["post"]:
            # 1) remove repetitions, emojis, etc.
            post = clean_text(post, lang)
            # 2) Extract meaningful tokens from weblinks
            post = extract_meaningful_tokens_from_weblinks(post)
            # 3) Process hashtags
            post = convert_hashtags(post)
            post = post.replace('""', '"')
            clean_posts.append(post)
        df["clean post"] = clean_posts

        # save dataframe
        df.to_csv(output_path_to_folder+file_name, index=False) 


def calculate_similarities(path_to_folder):

    # collect all (multilingual) original or clean posts to calculate multilingual similarities
    multi_posts = []

    # sort the files to keep the same order as during the inference!
    data_type = path_to_folder.split("/")[1] # "data" or "clean_data"
    if data_type == "clean_data":
        flag = "clean post"
        data_type = "./clean_data"
    else:
        flag = "post"
        data_type = "./data"

    # take train partitions
    for csv_file in sorted(glob.glob(data_type+"/train/"+"*.csv")):
        df = pd.read_csv(csv_file)
        multi_posts.extend(df[flag].tolist())

    print(len(multi_posts))

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    train_embeddings_multi = embedder.encode(multi_posts, convert_to_tensor=True, show_progress_bar=True)
    #print(train_embeddings_multi)

    for csv_file in glob.glob(path_to_folder+"*.csv"):
        print(csv_file)
        file_name = csv_file.split("/")[-1].split(".")[0].strip() # e.g., dev-eng
        df = pd.read_csv(csv_file)
        posts = df[flag]

        # calculate similarities between posts        
        curr_embeddings = embedder.encode(posts, convert_to_tensor=True)

        path_to_train = data_type+"/train/train-"+file_name.split("-")[-1]+".csv" # find similar always among train docs
        
        try:
            train_df = pd.read_csv(path_to_train)
            train_embeddings = embedder.encode(train_df[flag], convert_to_tensor=True)        
            similarity = torch.argsort(embedder.similarity(curr_embeddings, train_embeddings), dim=-1, descending=True).tolist() # a 2D list
        except FileNotFoundError:
            similarity = [[]]*df.shape[0]
            
        similarity_multi = torch.argsort(embedder.similarity(curr_embeddings, train_embeddings_multi), dim=-1, descending=True).tolist() # a 2D list
        
        # store as json
        with open(path_to_folder+file_name+".jsonlines", "w") as output_file:
            column_names = df.columns.tolist()
            for i, _ in df.iterrows():
                row = dict()
                for name in column_names:
                    row[name] = df[name][i] # copy info
                row["id"] = i
                row["similarities"] = similarity[i]
                row["similarities multi"] = similarity_multi[i]
                    
                output_file.write(json.dumps(row))
                output_file.write('\n')


if __name__ == "__main__":

    # clean up the data first, update the dataframes for each partition and language
    partitions = ["train/", "dev/", "test/"]
    for partition in partitions:

        input_data_path = "./data/"+partition
        output_data_path = "./clean_data/"+partition

        cleanup_files(input_data_path, output_data_path)

    # calculate similarities between the posts
    partitions = ["dev/", "test/"]
    data_types = ["./data/", "./clean_data/"]
    for data_type in data_types:
        for partition in partitions:
            folder = data_type+partition
            calculate_similarities(folder)