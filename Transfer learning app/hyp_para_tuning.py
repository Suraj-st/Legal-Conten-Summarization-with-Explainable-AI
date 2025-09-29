import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import optuna

# 1. Load dataset

dataset = load_dataset("json", data_files="legal_dataset_hf.json")

# Split dataset into train (90%) and validation (10%)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]


# 2. Load tokenizer & model

model_checkpoint = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["summary"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary"])


# 3. Metric (ROUGE)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    # result already returns floats like {"rouge1": 0.32, "rouge2": 0.15, "rougeL": 0.28, ...}
    return {k: v for k, v in result.items()}



# 4. Hyperparameter tuning with Optuna

def model_init():
    return BartForConditionalGeneration.from_pretrained(model_checkpoint)

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=None),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    return eval_results["eval_rougeL"]


# 5. Run Optuna study

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print("Best trial:")
trial = study.best_trial
print(f"  RougeL: {trial.value}")
print("  Best hyperparameters:", trial.params)