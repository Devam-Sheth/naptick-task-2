from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
import os
import traceback
from huggingface_hub import HfApi, login

jsonl_data_files = [
    "sleep_coach_multichat_1000.jsonl",
    "evolving_advice_200.jsonl"
]

model_checkpoint = "EleutherAI/pythia-410m"

hub_model_id = "devam-sheth-bits/sleep-ai-evolving" # Using a distinct name

training_output_dir = "./training_results_combined_evolving"
logging_dir = "./training_logs_combined_evolving"
max_seq_length = 512
LOGGING_STEPS = 20
SAVE_STEPS = 100 

print("Verifying Hugging Face CLI login status...")

print(f"Loading tokenizer for base model: {model_checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded. Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")

print(f"Attempting to load combined dataset from: {jsonl_data_files}")

dataset = load_dataset("json", data_files=jsonl_data_files, split="train")
print(f"Successfully loaded combined dataset. Total examples: {len(dataset)}")

def format_and_tokenize(example):
    formatted_text = ""
    if 'conversation' in example and isinstance(example['conversation'], list):
        for turn in example['conversation']:
             if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
                 formatted_text += f"{turn['role']}: {turn['content']}{tokenizer.eos_token}"
             else:
                 formatted_text += f"malformed_turn: {str(turn)}{tokenizer.eos_token}"
    else:
        return {'input_ids': [], 'attention_mask': []}
    tokenized_output = tokenizer(
        formatted_text,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    return tokenized_output

print("Tokenizing combined dataset...")
tokenized_dataset = dataset.map(format_and_tokenize, batched=False, remove_columns=dataset.column_names)
print("Tokenization complete.")

original_count = len(tokenized_dataset)
tokenized_dataset = tokenized_dataset.filter(lambda example: len(example['input_ids']) > 0)
filtered_count = len(tokenized_dataset)
print(f"Filtered {original_count - filtered_count} empty examples. Remaining: {filtered_count}")

if filtered_count == 0:
    print("ERROR: No valid examples remaining after tokenization.")
    exit()

# --- Split Dataset ---
if filtered_count > 10:
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Split combined data into {len(train_dataset)} training and {len(eval_dataset)} evaluation examples.")
else:
    train_dataset = tokenized_dataset
    eval_dataset = None
    print(f"Using all {len(train_dataset)} examples from combined data for training (evaluation disabled).")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(f"Loading BASE model for fine-tuning on combined data: {model_checkpoint}")
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model '{model_checkpoint}' loaded onto device: {device}")

print("Initializing Training Arguments...")

training_args = TrainingArguments(
        output_dir=training_output_dir,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=1, # Adjust if needed
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        report_to="none",
        push_to_hub=True,
        hub_model_id=hub_model_id, # Pushing to the NEW ID
        hub_strategy="checkpoint",
    )
print("TrainingArguments initialized successfully.")

# --- Initialize Trainer ---
print("Initializing Trainer...")

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
print("Trainer Initialized.")
print(f"Starting model training on COMBINED data... Model will be pushed to: https://huggingface.co/{hub_model_id}")

train_result = trainer.train()
print("Training finished.")
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

final_local_dir = f"./{hub_model_id.split('/')[-1]}_final"
print(f"Saving final model locally to: {final_local_dir}")
os.makedirs(final_local_dir, exist_ok=True)
trainer.save_model(final_local_dir)
tokenizer.save_pretrained(final_local_dir)
print("Final model saved locally.")

print(f"Pushing final model and tokenizer to Hub repository: {hub_model_id}")

trainer.push_to_hub(commit_message="End of training on combined sleep data")
print("Model pushed successfully.")
