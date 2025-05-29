import argparse

import pandas as pd
from unsloth import FastLanguageModel, FastModel

HF_TOKEN = ""  # HuggingFace access token

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


def is_valid_claim(generated_claim, post, target_lang):
    # less than 3 tokens
    if len(set(generated_claim.split())) <= 3 and target_lang != "tha":
        return False
    # less than 5 characters
    if len(set(generated_claim)) <= 5:
        return False
    # exclude links
    if "http" in generated_claim:
        return False
    return True


def main(model_name: str, instruction_type: str, target_lang: str, topk: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    test_data = pd.read_csv(f"data/test/test-{target_lang}.csv")
    posts = list(test_data["post"])

    generated_claims = []
    post2gen = dict()
    for idx in range(topk):
        post2gen[idx] = []
    for idx, post in enumerate(posts):
        # remove all duplicated sentences in the post!
        split_post = post.split(".")
        checked = set()
        new_sent = []
        for sent in split_post:
            if sent not in checked:
                new_sent.append(sent)
            checked.add(sent)
        post = ".".join(new_sent)

        gen_per_post = 0
        while gen_per_post < topk:
            generated_claim = ""
            counter = 0
            invalid_gen_claim = True
            while invalid_gen_claim and counter < topk:
                prepared_post = post
                if instruction_type == "concise_instruction":
                    prepared_post = (
                        "You are an expert in misinformation detection and fact-checking. Your task is to identify the central claim in the given post while preserving its original language. Post: "
                        + prepared_post
                    )
                elif instruction_type == "verbose_instruction":
                    prepared_post = (
                        "You are an expert in misinformation detection and fact-checking. Your task is to identify the central claim in the given post while preserving its original language.\n\nThe central claim should meet the following criteria:\n- **Verifiable**: It must be a factual assertion that can be checked against evidence.\n- **Concise**: It should be a single, clear sentence that captures the main claim of the post.\n- **Socially impactful**: It should be a statement that could influence public opinion, health, or policy.\n- **Free from rhetorical elements**: Do not include opinions, rhetorical questions, or unnecessary context.\n- **Preserve Original Language**: The output should be in the same language as the input post.\n\nOutput only the central claim without additional explanation or formatting.\nPost: "
                        + prepared_post
                    )

                if "gemma" in model_name.lower():
                    post_entry = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prepared_post,
                            }
                        ],
                    }
                elif "qwen" in model_name.lower():
                    post_entry = {"role": "user", "content": prepared_post}
                val_messages = [post_entry]
                if "gemma" in model_name.lower():
                    val_text = tokenizer.apply_chat_template(
                        val_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    outputs = model.generate(
                        **tokenizer(
                            [val_text],
                            return_tensors="pt",
                        ).to("cuda"),
                        max_new_tokens=64,
                        temperature=1.0,
                        top_p=0.95,
                        top_k=64,
                    )
                elif "qwen" in model_name.lower():
                    val_text = tokenizer.apply_chat_template(
                        val_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    outputs = model.generate(
                        **tokenizer([val_text], return_tensors="pt").to("cuda"),
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                    )
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                if "gemma" in model_name.lower():
                    generated_claim = res.split("\nmodel\n")[1]
                elif "qwen" in model_name.lower():
                    generated_claim = (
                        res.split("assistant\n")[1]
                        .replace("<think>", "")
                        .replace("</think>", "")
                        .replace("\n", " ")
                        .strip()
                    )
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                print()
                print(idx, post[:50], ">>>", generated_claim)
                print()
                if is_valid_claim(generated_claim, post, target_lang):
                    generated_claims.append(generated_claim)
                    invalid_gen_claim = False
                else:
                    print(f"Invalid claim: {generated_claim}")
                counter += 1
            post2gen[gen_per_post].append(generated_claim)
            gen_per_post += 1
    df = pd.DataFrame.from_dict({"normalized claim": generated_claims})
    df.to_csv(f"outputs/task2_{target_lang}.csv", index=False)

    df_multiple = pd.DataFrame.from_dict(post2gen)
    df_multiple.to_csv(f"outputs/task2_{target_lang}_multiple.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference parameters.")
    parser.add_argument("--model_name", type=str)
    parser.add_argument(
        "--instruction_type",
        type=str,
        choices=["no_instruction", "concise_instruction", "verbose_instruction"],
    )
    parser.add_argument("--target_lang", type=str, choices=VALID_LANG_NAMES)
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()
    main(args.model_name, args.instruction_type, args.target_lang, args.topk)
