# this code is mostly based on https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb
import argparse
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from nltk.tokenize import RegexpTokenizer
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
from trl import SFTConfig, SFTTrainer # this needs to be imported after unsloth!

# Important: to access the gated models you need to specify HF_TOKEN
# Also, for logging with W&B you need to add the wandb key, e.g.:
# export WANDB_API_KEY="YOUR_KEY"
# wandb login

# It is also recommended to set `use_cache` to False in the model's config
# otherwise we may get "AttributeError: 'HybridCache' object has no attribute 'float'"
# see also https://github.com/unslothai/unsloth/issues/2052


def create_dataset_split(path_to_split: str):
    # e.g. path_to_split="data/train/train-pol.csv"
    # see https://docs.unsloth.ai/basics/datasets-guide
    dataset_split = pd.read_csv(path_to_split)
    dataset_split = Dataset.from_pandas(dataset_split.sample(frac=1).reset_index(drop=True))
    new_dataset_split = []
    for el in dataset_split:
        new_dataset_split.append(
            [
                {"from": "human", "value": el["post"]},
                {"from": "gpt", "value": el["normalized claim"]},
            ]
        )
    return Dataset.from_dict({"conversations": new_dataset_split})


def main(
    model_name: str = "unsloth/gemma-3-4b-it",
    max_seq_length: int = 2048,
    train_language_code: str = "deu",
    val_language_code: str = "deu",
    num_epochs: int = 10,
    verbose: bool = False,
):
    # loading the model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,  # e.g. 1024 or 2048
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # slightly more accurate, uses 2x memory
        full_finetuning=False,
    )

    # loading adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # not needed for text inputs
        finetune_language_layers=True,  # always on
        finetune_attention_modules=True,  # attention is good for GRPO
        finetune_mlp_modules=True,  # always on
        r=8,  # larger r can give higher accuracy, but also overfit
        lora_alpha=8,  # at least alpha == r
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # preparing the data
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    path_to_train_data = f"data/train/train-{train_language_code}.csv"
    path_to_validation_data = f"data/dev/dev-{train_language_code}.csv"

    train_split = create_dataset_split(path_to_train_data)
    val_split = create_dataset_split(path_to_validation_data)

    train_split = standardize_data_formats(train_split)
    val_split = standardize_data_formats(val_split)

    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"])
        return {"text": texts}

    train_split = train_split.map(apply_chat_template, batched=True)
    val_split = val_split.map(apply_chat_template, batched=True)

    # training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_split,
        eval_dataset=val_split,
        bf16=False,
        args=SFTConfig(
            eval_steps=100,
            eval_strategy="steps",
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # using GA to approximate the batch size
            warmup_steps=5,
            num_train_epochs=num_epochs,
            learning_rate=2e-4,  # maybe to 2e-5 for longer training
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("Starting training!")
    with torch.cuda.amp.autocast():
        trainer_stats = trainer.train()
    print(f"Statistics: {trainer_stats}")

    os.makedirs("saved_models", exist_ok=True)
    save_name = "saved_models/" + model_name.split("/")[1] + "-" + train_language_code
    model.save_pretrained(save_name)  # saving tuned adapters and configs
    tokenizer.save_pretrained(save_name)  # saving the tokenizer

    def tokenize_sentence(arg):
        encoded_arg = tokenizer(arg)
        return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

    # evaluation
    meteor_metric = evaluate.load("meteor", tokenize=tokenize_sentence)

    val_data = pd.read_csv(path_to_validation_data)
    posts = list(val_data["post"])
    normalized_gold_claims = list(val_data["normalized claim"])

    generated_claims = []
    for idx, post in enumerate(posts):
        post_entry = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": post,
                }
            ],
        }
        val_messages = [post_entry]
        val_text = tokenizer.apply_chat_template(
            val_messages,
            add_generation_prompt=True,  # needed for generation
        )
        outputs = model.generate(
            **tokenizer([val_text], return_tensors="pt").to("cuda"),
            max_new_tokens=64,  # could be longer
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_claim = res.split("\nmodel\n")[1]
        generated_claims.append(generated_claim)

        if verbose:
            print(f"{idx}: generated: {generated_claim}")
            print(f"{idx}: gold: {normalized_gold_claims[idx]}")
            print()

    # copied this part from the official baseline
    text_preds = [
        (txt if txt.endswith(("!", "！", "?", "？", "。")) else txt + "。") for txt in generated_claims
    ]
    text_labels = [
        (txt if txt.endswith(("!", "！", "?", "？", "。")) else txt + "。")
        for txt in normalized_gold_claims
    ]
    sent_tokenizer_jp = RegexpTokenizer("[^!！?？。]*[!！?？。]")
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(txt))) for txt in text_preds]
    text_labels = [
        "\n".join(np.char.strip(sent_tokenizer_jp.tokenize(txt))) for txt in text_labels
    ]

    # compute METEOR score
    meteor_score = meteor_metric.compute(
        predictions=text_preds,
        references=text_labels,
    )

    print("Training language:", train_language_code)
    print("Validation language:", val_language_code)
    print("METEOR:", meteor_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning parameters.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-4b-it")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--train_language_code", type=str, default="deu")
    parser.add_argument("--val_language_code", type=str, default="deu")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)
    main(
        args.model_name,
        args.max_seq_length,
        args.train_language_code,
        args.val_language_code,
        args.num_epochs,
        args.verbose,
    )
