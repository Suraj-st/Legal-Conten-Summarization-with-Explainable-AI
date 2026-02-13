import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    # TrainingArguments,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate
from optuna import Trial
from transformers import Seq2SeqTrainer

# Load dataset (JSON with fields: text, summary)

dataset = load_dataset(
    "json",
    data_files={"train": "legal_dataset_hf_src.json"},
    streaming=False
)

# Load tokenizer & model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Preprocess function
max_input_length = 256 #512
max_target_length = 64 # 128

def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,     
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text", "summary"],
    num_proc=4   # 🚀 speed boost
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Metric
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }

# Hyperparameter tuning with Optuna
def model_init():
    return BartForConditionalGeneration.from_pretrained(model_name)

def hp_space(trial: Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-finetuned-legal",

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    eval_accumulation_steps=1,

    predict_with_generate=True,   

    bf16=torch.cuda.is_available(),
    fp16=False,

    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
)

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
eval_dataset  = tokenized_datasets["train"].shuffle(seed=123).select(range(500))

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Run hyperparameter search
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=10  # increase for better results
)
# enable Weights & Biases
import wandb; wandb.init(project="legal-doc-summarizer-hptune", name="bart-base-train")

print("Best hyperparameters:", best_run.hyperparameters)
