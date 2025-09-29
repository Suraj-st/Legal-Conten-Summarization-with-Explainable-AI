# 2. TRAIN MODEL (BART-base)

from datasets      import load_dataset
from transformers  import (AutoTokenizer, AutoModelForSeq2SeqLM,
                           Seq2SeqTrainingArguments, Seq2SeqTrainer,
                           DataCollatorForSeq2Seq)

raw_data   = load_dataset("json", data_files="legal_dataset_hf.json")["train"]
dataset    = raw_data.train_test_split(test_size=0.1)

model_name = "facebook/bart-base"      # ← smaller, fits 8 GB GPU easily
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    # INPUTS: up to 1024 tokens
    inputs = tokenizer(examples["text"], max_length=1024,
                       truncation=True, padding="max_length")
    # LABELS: up to 256 tokens
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=256,
                           truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_ds = dataset.map(preprocess, batched=True, remove_columns=["text", "summary"])

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

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
)

# OPTIONAL: enable Weights & Biases
import wandb; wandb.init(project="legal-doc-summarizer", name="bart-base-train")

trainer.train()

best_ckpt = trainer.state.best_model_checkpoint
print(f"🏆 Best checkpoint: {best_ckpt}")

# Save final artefacts
model.save_pretrained("legal-summarizer-final")
tokenizer.save_pretrained("legal-summarizer-final")
print("✅ Model + tokenizer saved → legal-summarizer-final/")