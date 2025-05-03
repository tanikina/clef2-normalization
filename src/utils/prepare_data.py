import argparse
import random
from pathlib import Path

import pandas as pd
from ftlangdetect import detect

random.seed(1024)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

VALID_LANG_NAMES = [
    "deu",
    "pol",
    "hi",
    "mr",
    "ara",
    "msa",
    "fra",
    "pa",
    "por",
    "spa",
    "ta",
    "tha",
    "eng",
]
TARGET_LANG_CODE = {
    "deu": "de",
    "pol": "pl",
    "hi": "hi",
    "mr": "mr",
    "ara": "ar",
    "msa": "id",
    "fra": "fr",
    "pa": "pa",
    "por": "pt",
    "spa": "es",
    "ta": "ta",
    "tha": "th",
    "eng": "en",
}


def filter_posts_and_claims(
    input_filename: str,
    target_lang: str,
    similarity_threshold: float = 0.05,
    verbose: bool = False,
):
    """Filter out the claims and posts that:
    (1) have duplicates in the given file;
    (2) claim and post have different languages, and one of them is English with high probability;
    (3) cosine similarity between the claim and the post is less than the min threshold (e.g. 0.05).
    """
    df = pd.read_csv(input_filename)
    filtered_posts_with_claims = []
    processed_posts = []
    if "train" in input_filename:
        print("original posts:", len(df["post"]))
    for post, claim in zip(list(df["post"]), list(df["normalized claim"])):
        single_line_post = post.replace("\n", "")
        single_line_claim = claim.replace("\n", "")
        # check that claim and post do not mix English with another language
        # note that we cannot just check if the language is the same and
        # corresponds to the target language, e.g. Marathi is often misidentified as Hindi
        # Also, Indonesian is sometimes misidentified as English, thus we need to check
        # that the language identification score is high enough.
        res_post = detect(text=single_line_post, low_memory=True)
        score_post = res_post["score"]
        res_claim = detect(text=single_line_claim, low_memory=True)
        score_claim = res_claim["score"]
        mixed_with_english = res_post["lang"] != res_claim["lang"] and (
            (res_post["lang"] == "en" and score_post > 0.95)
            or (res_claim["lang"] == "en" and score_claim > 0.95)
        )
        # check post-claim similarity
        post_embeddings = model.encode([single_line_post.lower()])
        claim_embeddings = model.encode([single_line_claim.lower()])
        similarity = model.similarity(post_embeddings, claim_embeddings).item()
        claim_post_similarity = similarity > similarity_threshold
        if verbose:
            if mixed_with_english:
                print(f"Mixed with English:\nClaim: {claim[:50]}\nPost: {post[:50]}\n")
            elif not claim_post_similarity:
                print(f"Low similarity:\nClaim: {claim[:50]}\nPost: {post[:50]}\n")
        if post not in processed_posts and not mixed_with_english and claim_post_similarity:
            filtered_posts_with_claims.append((post, claim))
        # keep track for duplicates (within each split)
        processed_posts.append(post)
    return filtered_posts_with_claims, processed_posts


def prepare_data(target_lang: str, similarity_threshold: float, verbose: bool):
    # read original train and dev data
    train_posts_with_claims, train_posts = filter_posts_and_claims(
        f"data/train/train-{target_lang}.csv", target_lang, similarity_threshold, verbose
    )
    dev_posts_with_claims, dev_posts = filter_posts_and_claims(
        f"data/dev/dev-{target_lang}.csv", target_lang, similarity_threshold, verbose
    )
    # save the "filtered" train data but leave the dev set as it is
    new_train_posts = []
    new_train_claims = []
    for post, claim in train_posts_with_claims:
        if post not in dev_posts:
            new_train_posts.append(post)
            new_train_claims.append(claim)
    df_train_new = pd.DataFrame.from_dict(
        {"post": new_train_posts, "normalized claim": new_train_claims}
    )
    save_path_filtered = f"data/filtered/train/train-{target_lang}.csv"
    Path(save_path_filtered).parent.absolute().mkdir(parents=True, exist_ok=True)
    df_train_new.to_csv(save_path_filtered, index=False)
    if verbose:
        print(
            f"{target_lang}: {len(new_train_posts)} filtered posts in the train set (with the original dev set)"
        )
    # combine the "filtered" train and dev data,
    # use 10% of the combined data as a new dev split
    new_train_dev_posts = []
    new_train_dev_claims = []
    # copy the filtered posts and claims from the train data
    for post, claim in train_posts_with_claims:
        new_train_dev_posts.append(post)
        new_train_dev_claims.append(claim)
    # add dev data w/o duplicates and overlaps with the train data
    for post, claim in dev_posts_with_claims:
        if post not in train_posts:
            new_train_dev_posts.append(post)
            new_train_dev_claims.append(claim)
    new_train_posts = []
    new_train_claims = []
    new_dev_posts = []
    new_dev_claims = []
    total_samples_combined = len(new_train_dev_claims)
    # randomly select 10% as a new dev set
    indices = [idx for idx in range(total_samples_combined)]
    dev_indices = random.choices(indices, k=round(0.1 * total_samples_combined))
    for idx in range(total_samples_combined):
        if idx in dev_indices:
            new_dev_posts.append(new_train_dev_posts[idx])
            new_dev_claims.append(new_train_dev_claims[idx])
        else:
            new_train_posts.append(new_train_dev_posts[idx])
            new_train_claims.append(new_train_dev_claims[idx])
    # save new splits into the files
    df_train_new = pd.DataFrame.from_dict(
        {"post": new_train_posts, "normalized claim": new_train_claims}
    )
    save_path_filtered_combined_train = f"data/combined_filtered/train/train-{target_lang}.csv"
    Path(save_path_filtered_combined_train).parent.absolute().mkdir(parents=True, exist_ok=True)
    df_train_new.to_csv(save_path_filtered_combined_train, index=False)

    df_dev_new = pd.DataFrame.from_dict(
        {"post": new_dev_posts, "normalized claim": new_dev_claims}
    )
    save_path_filtered_combined_dev = f"data/combined_filtered/dev/dev-{target_lang}.csv"
    Path(save_path_filtered_combined_dev).parent.absolute().mkdir(parents=True, exist_ok=True)
    df_dev_new.to_csv(save_path_filtered_combined_dev, index=False)
    # print some statistics
    if verbose:
        print(
            f"{target_lang}: {len(new_dev_posts)} filtered posts in the dev set and {len(new_train_posts)} filtered posts in the train set (dev set is 10% of the combined data)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation parameters.")
    parser.add_argument("--target_lang", type=str, choices=VALID_LANG_NAMES + ["all"])
    parser.add_argument("--similarity_threshold", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.target_lang == "all":
        for target_lang in VALID_LANG_NAMES:
            prepare_data(target_lang, args.similarity_threshold, args.verbose)
    else:
        prepare_data(args.target_lang, args.similarity_threshold, args.verbose)
