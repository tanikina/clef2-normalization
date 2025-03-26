import argparse
import copy
import os

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb

if torch.cuda.is_available():
    print("GPU is enabled.")
    print(
        "device count: {}, current device: {}".format(
            torch.cuda.device_count(), torch.cuda.current_device()
        )
    )
else:
    print("GPU is not enabled.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # make sure GPU is enabled.

import accelerate
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class Baseline:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.metric = evaluate.load("meteor", tokenize=self.tokenize_sentence)
        self.model_checkpoint = model_checkpoint

        config = AutoConfig.from_pretrained(
            model_checkpoint,
            max_length=128,
            length_penalty=0.6,
            no_repeat_ngram_size=2,
            num_beams=15,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=config).to(
            device
        )

    def tokenize_sample_data(self, data):
        # Max token size is set to 1024 and 128 for inputs and labels, respectively.
        input_feature = self.tokenizer(data["post"], truncation=True, max_length=1024)
        label = self.tokenizer(data["normalized claim"], truncation=True, max_length=128)
        return {
            "input_ids": input_feature["input_ids"],
            "attention_mask": input_feature["attention_mask"],
            "labels": label["input_ids"],
        }

    def tokenize_sentence(self, arg):
        encoded_arg = self.tokenizer(arg)
        return self.tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

    def metrics_func(self, eval_arg):
        preds, labels = eval_arg
        # Replace -100
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Convert id tokens to text
        text_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        text_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Insert a line break (\n) in each sentence for scoring
        text_preds = [
            (p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds
        ]
        text_labels = [
            (l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels
        ]
        sent_tokenizer_jp = RegexpTokenizer("[^!！?？。]*[!！?？。]")
        text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
        text_labels = [
            "\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels
        ]
        # compute METEOR score with custom tokenization
        return self.metric.compute(
            predictions=text_preds,
            references=text_labels,
            # tokenizer=_tokenize_sentence
        )


def train_baseline(
    language: str = "English",
    model_checkpoint: str = "google/umt5-base",
    train_data_path: str = "../../data/train/train-eng.csv",
    val_data_path: str = "../../data/dev/dev-eng.csv",
    max_epochs: int = 20,
    learning_rate: float = 5e-4,
    seed: int = 42,
):
    wandb.init(tags=["clef2-baseline", language])

    model_code = (
        model_checkpoint.split("/")[-1] + "-lr-" + str(learning_rate) + "-seed-" + str(seed)
    )

    baseline = Baseline(model_checkpoint)

    data_collator = DataCollatorForSeq2Seq(
        baseline.tokenizer, model=baseline.model, return_tensors="pt"
    )

    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)

    ds_original = DatasetDict(
        {"train": Dataset.from_pandas(train_data), "validation": Dataset.from_pandas(val_data)}
    )

    # add 10% of the original
    splitted_train = ds_original["train"].train_test_split(test_size=0.1, seed=42)
    # TODO: check if there is a more efficient way of combining the two splits
    val_combined = copy.deepcopy(ds_original["validation"])
    subsampled_val = splitted_train["test"]
    for item in subsampled_val:
        val_combined = val_combined.add_item(item)
    ds = DatasetDict({"train": splitted_train["train"], "validation": val_combined})

    tokenized_ds = ds.map(
        baseline.tokenize_sample_data,
        remove_columns=["post", "normalized claim"],
        batched=True,
        batch_size=1,
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=5)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"saved-models-{model_code}",
        num_train_epochs=max_epochs,  # epochs
        seed=seed,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=90,
        optim="adamw_hf",
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=100,
        eval_steps=100,
        predict_with_generate=True,
        generation_max_length=128,
        eval_strategy="steps",
        logging_steps=10,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=baseline.model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=baseline.metrics_func,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=baseline.tokenizer,
        callbacks=[early_stop],
    )

    trainer.train()

    os.makedirs(f"{model_code}/finetuned_{model_code}", exist_ok=True)

    if hasattr(trainer.model, "module"):
        trainer.model.module.save_pretrained(f"./{model_code}/finetuned_{model_code}")
    else:
        trainer.model.save_pretrained(f"./{model_code}/finetuned_{model_code}")

    print("Training done")

    # Evaluation on the development set (both original and subsampled)
    print("Evaluation on both validation splits")
    print(trainer.evaluate())
    tokenized_validation_original = ds_original["validation"].map(
        baseline.tokenize_sample_data,
        remove_columns=["post", "normalized claim"],
        batched=True,
        batch_size=1,
    )
    trainer.eval_dataset = tokenized_validation_original
    print("Evaluation on the official validation split")
    print(trainer.evaluate())
    tokenized_validation_subsampled = subsampled_val.map(
        baseline.tokenize_sample_data,
        remove_columns=["post", "normalized claim"],
        batched=True,
        batch_size=1,
    )
    trainer.eval_dataset = tokenized_validation_subsampled
    print("Evaluation on the validation split subsampled from the training set")
    print(trainer.evaluate())


def run_inference(model_checkpoint, learning_rate, seed, input_texts):
    # Inference
    model_code = (
        model_checkpoint.split("/")[-1] + "-lr-" + str(learning_rate) + "-seed-" + str(seed)
    )
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(f"./{model_code}/finetuned_{model_code}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.eval()  # Set the model to evaluation mode

    for text in input_texts:
        # Tokenize the Input Text
        inputs = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():  # Disable gradient calculation
            generated_ids = model.generate(
                inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True
            )

        # Decode the Generated Output
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Print the Output
        print(f"Generated Output: {output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters.")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--model_checkpoint", type=str, default="google/umt5-base")
    parser.add_argument("--train_data_path", type=str, default="../../data/train/train-eng.csv")
    parser.add_argument("--val_data_path", type=str, default="../../data/dev/dev-eng.csv")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)

    train_baseline(
        args.language,
        args.model_checkpoint,
        args.train_data_path,
        args.val_data_path,
        args.max_epochs,
        args.learning_rate,
        args.seed,
    )
