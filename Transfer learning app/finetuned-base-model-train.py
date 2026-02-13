# 2. TRAIN MODEL (BART-base, fp16)
import numpy as np
import evaluate
from datasets      import load_dataset
from transformers  import (AutoTokenizer, AutoModelForSeq2SeqLM,
                           Seq2SeqTrainingArguments, Seq2SeqTrainer,
                           DataCollatorForSeq2Seq)


raw_data   = load_dataset("json", data_files="legal_dataset_hf.json")["train"]
dataset    = raw_data.train_test_split(test_size=0.1)

model_name = "facebook/bart-base"      # ← smaller, fits 8 GB easily
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    mod_inputs = examples["text"]
    targets = examples["summary"]

    inputs = tokenizer(
        mod_inputs,
        max_length=1024,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs



rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)

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

tokenized_ds = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["text", "summary"]
    # num_proc=4   # 🚀 speed boost
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()          # save VRAM

training_args = Seq2SeqTrainingArguments(
    output_dir             = "./results_legal",
    evaluation_strategy    = "steps",
    save_steps             = 500,
    eval_steps             = 500,
    logging_steps          = 100,
    learning_rate          = 2e-5,
    per_device_train_batch_size = 1,   # 1 sample × 1024 tokens fits 8 GB
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 8,   # effective batch = 8
    num_train_epochs       = 4,
    warmup_steps           = 200,
    predict_with_generate  = True,
    fp16                   = True,     # mixed precision
    load_best_model_at_end = True,
    metric_for_best_model  = "eval_loss",
    greater_is_better      = False,
    generation_max_length  = 256,
    generation_num_beams   = 3,        # faster than 5
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer       = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset =tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# OPTIONAL: enable Weights & Biases
import wandb; wandb.init(project="legal-doc-summarizer-new1", name="bart-base-train")

trainer.train()

best_ckpt = trainer.state.best_model_checkpoint
print(f"🏆 Best checkpoint: {best_ckpt}")

# Save final artefacts
model.save_pretrained("legal-summarizer-final")
tokenizer.save_pretrained("legal-summarizer-final")
print("✅ Model + tokenizer saved → legal-summarizer-final/")
