import os
import argparse
import pandas as pd

def main(split, balanced):
    # threshold is 50 for dev and 500 for train
    if split == "dev":
        threshold = 50
    elif split == "train":
        threshold = 500
    else:
        threshold = None

    input_dir = f"data/{split}"

    if balanced:
        output_path_combined = f"data/{split}/{split}-balanced-all.csv"
    else:
        output_path_combined = f"data/{split}/{split}-all.csv" 

    posts = []
    normalized_claims = []

    for fname in os.listdir(input_dir):
        if "-all" in fname:
            continue
        df = pd.read_csv(f"{input_dir}/{fname}")
        posts_per_lang = list(df["post"])
        print(fname, len(posts_per_lang))
        if balanced and threshold is not None:
            posts_per_lang = posts_per_lang[:threshold]
        posts.extend(posts_per_lang)
        claims_per_lang = list(df["normalized claim"])
        if balanced and threshold is not None:
            claims_per_lang = claims_per_lang[:threshold]
        normalized_claims.extend(claims_per_lang)

    df = pd.DataFrame(data={"post": posts, "normalized claim": normalized_claims})
    df.to_csv(output_path_combined, index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or evaluation parameters.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev"])
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args.split, args.balanced)
