# Based on the following code:
# Gemma3 fine-tuning with Unsloth: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=yqxqAZ7KJ4oL
# Qwen3 fine-tuning with Unsloth: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb#scrollTo=kR3gIAX-SM2q

import argparse
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from nltk.tokenize import RegexpTokenizer
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)

HF_TOKEN = ""  # HuggingFace access token


def create_dataset_split(path_to_split: str, model_name: str, instruction_type: str):
    # e.g. data/train/train-pol.csv
    dataset_split = pd.read_csv(path_to_split)
    dataset_split = Dataset.from_pandas(dataset_split.sample(frac=1).reset_index(drop=True))
    new_dataset_split = []
    for el in dataset_split:
        prepared_post = el["post"]
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
            new_dataset_split.append(
                [
                    {"from": "human", "value": prepared_post},
                    {"from": "gpt", "value": el["normalized claim"]},
                ]
            )
        elif "qwen" in model_name.lower():
            new_dataset_split.append(
                [
                    {"role": "user", "content": prepared_post},
                    {"role": "assistant", "content": el["normalized claim"]},
                ]
            )
        else:
            raise ValueError(f"Unknown model {model_name}")
    return Dataset.from_dict({"conversations": new_dataset_split})


def main(
    model_name: str = "unsloth/gemma-3-4b-it",
    max_seq_length: int = 2048,
    train_language_code: str = "deu",
    val_language_code: str = "deu",
    input_folder: str = "data",
    num_epochs: int = 10,
    learning_rate: float = 2e-4,
    lr_scheduler: str = "linear",
    instruction_type: str = "no_instruction",
    save_adapters_folder: str = "saved_adapters",
):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False,
        full_finetuning=False,
        token=HF_TOKEN,
    )

    # loading adapters
    if "gemma" in model_name.lower():
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=8,
            lora_alpha=8,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
    elif "qwen" in model_name.lower():
        model = FastModel.get_peft_model(
            model,
            r=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # preparing the data
    def apply_chat_template(examples):
        if "qwen" in model_name.lower():
            texts = tokenizer.apply_chat_template(
                examples["conversations"], tokenize=False
            )  # , add_generation_prompt=True, enable_thinking=False)
        elif "gemma" in model_name.lower():
            texts = tokenizer.apply_chat_template(
                examples["conversations"], tokenize=False, add_generation_prompt=True
            )
        else:
            raise ValueError(f"Unknown model name {model_name}")
        return {"text": texts}

    path_to_train_data = f"{input_folder}/train/train-{train_language_code}.csv"
    dev_language_code = (
        train_language_code.replace("-translated", "")
        .replace("_with_all_balanced", "")
        .replace("_with_best_translated", "")
    )
    path_to_validation_data = f"{input_folder}/dev/dev-{dev_language_code}.csv"

    train_split = create_dataset_split(path_to_train_data, model_name, instruction_type)
    val_split = create_dataset_split(path_to_validation_data, model_name, instruction_type)

    train_split = standardize_data_formats(train_split)
    val_split = standardize_data_formats(val_split)
    train_split = train_split.map(apply_chat_template, batched=True)
    val_split = val_split.map(apply_chat_template, batched=True)

    # training!
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
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=lr_scheduler,
            seed=3407,
            report_to="wandb",
        ),
    )

    if "gemma" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
    elif "qwen" in model_name.lower():
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    print("Training!")
    with torch.cuda.amp.autocast():
        trainer_stats = trainer.train()
    print(trainer_stats)

    save_name = save_adapters_folder + "/" + model_name.split("/")[1] + "-" + train_language_code
    Path(save_name).parent.absolute().mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)

    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)

    load_for_eval = False
    if load_for_eval:
        model, tokenizer = FastModel.from_pretrained(
            model_name=save_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )

    def tokenize_sentence(arg):
        encoded_arg = tokenizer(arg)
        return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

    meteor_metric = evaluate.load("meteor", tokenize=tokenize_sentence)

    val_data = pd.read_csv(path_to_validation_data)
    posts = list(val_data["post"])
    normalized_gold_claims = list(val_data["normalized claim"])

    generated_claims = []
    for idx, post in enumerate(posts):
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
                **tokenizer([val_text], return_tensors="pt").to("cuda"),
                max_new_tokens=64,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        elif "qwen" in model_name.lower():
            val_text = tokenizer.apply_chat_template(
                val_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
        print(idx, generated_claim, ">>>", normalized_gold_claims[idx])
        print()
        generated_claims.append(generated_claim)

    text_preds = [
        (p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in generated_claims
    ]
    text_labels = [
        (l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in normalized_gold_claims
    ]
    sent_tokenizer_jp = RegexpTokenizer("[^!！?？。]*[!！?？。]")
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]

    # compute METEOR score with custom tokenization
    meteor_score = meteor_metric.compute(
        predictions=text_preds,
        references=text_labels,
    )

    print("TRAIN LANG:", train_language_code)
    print("VAL LANG:", val_language_code)
    print("METEOR:", meteor_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning parameters.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-4b-it")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--train_language_code", type=str, default="deu")
    parser.add_argument("--val_language_code", type=str, default="deu")
    parser.add_argument("--input_folder", type=str, default="train")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--instruction_type", type=str, default="no_instruction")
    parser.add_argument("--save_adapters_folder", type=str, default="saved_adapters")

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)
    main(
        args.model_name,
        args.max_seq_length,
        args.train_language_code,
        args.val_language_code,
        args.input_folder,
        args.num_epochs,
        args.learning_rate,
        args.lr_scheduler,
        args.instruction_type,
        args.save_adapters_folder,
    )
