import argparse
import os
import random

import numpy as np
import pandas as pd

random.seed(1024)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def prepare_data(target_lang: str):

    file2samples = dict()
    for fname in os.listdir(target_lang):
        filepath = f"{target_lang}/{fname}"
        print(fname)
        file2samples[fname] = list(pd.read_csv(filepath)["normalized claim"])

    new_claims = []
    cnt = 0
    for idx, sample in enumerate(file2samples[f"task2_{target_lang}.csv"]):
        # compute the claim centroid
        samples = [file2samples[fname][idx] for fname in file2samples.keys()]
        sample_embeddings = model.encode(samples)
        claim_centroid = np.mean(sample_embeddings, axis=0)
        similarity = model.similarity(claim_centroid, sample_embeddings)[0].numpy()
        selected_idx = np.argmax(similarity)
        new_claims.append(samples[selected_idx])
        cnt += 1
        if cnt % 100 == 0:
            print(samples[selected_idx])

    df_new = pd.DataFrame.from_dict({"normalized claim": new_claims})
    df_new.to_csv(f"task2_ensemble_{target_lang}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation parameters.")
    parser.add_argument("--setting", type=str, choices=["monolingual", "zeroshot", "all"])
    args = parser.parse_args()
    zeroshot_langs = ["kor", "ces", "ron", "ell", "te", "nld", "bn"]
    monolingual_langs = [
        "pol",
        "msa",
        "deu",
        "tha",
        "ara",
        "ta",
        "pa",
        "mr",
        "hi",
        "por",
        "spa",
        "fra",
        "eng",
    ]
    if args.setting == "monolingual":
        languages = monolingual_langs
    elif args.setting == "zeroshot":
        languages = zeroshot_langs
    else:
        languages = monolingual_langs + zeroshot_langs
    # this assumes that we have a separate folder for each language
    # that contains the annotated test set, each file is an output of a different model/approach
    for lang in languages:
        prepare_data(lang)
