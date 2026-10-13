"""
train.py  (farm agent version)
==============================
LoRA fine-tune a small English LLM so that

    English command + farm state  ->  JSON plan of whitelisted skills

Run on Kaggle (GPU on), from the repo folder, AFTER `python make_farm_data.py`:
    python train.py

Needs: HF login (HF_TOKEN secret) because the last step pushes the adapter to the Hub.
Everything runs in ONE command, so it also works with "Save & Run All (Commit)".
"""

import json
import os

import torch
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments,
)

# ----------------------------
# Config
# ----------------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HF_MODEL_REPO = "MRlionman6/farm-agent-qwen-lora"     # NEW repo: old Thai model is untouched
TRAIN_FILE = "data/farm_train.jsonl"
OUTPUT_DIR = "farm_lora_out"
MAX_LEN = 512
NUM_EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-4
PUSH_TO_HUB = True

SEP = "### Plan:\n"      # everything up to and including this is the PROMPT


# ----------------------------
# Data: loss only on the plan, not on the command/state
# ----------------------------
class PlanDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.items = []
        eos = tokenizer.eos_token
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                prompt, plan = text.split(SEP, 1)
                prompt += SEP
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                a_ids = tokenizer(plan + eos, add_special_tokens=False)["input_ids"]
                ids = (p_ids + a_ids)[:MAX_LEN]
                labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
                self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        login(token=token)
    elif PUSH_TO_HUB:
        raise SystemExit("No HF_TOKEN found. Add the Kaggle secret or set PUSH_TO_HUB=False.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp32 weights + fp16 autocast: works on T4 (no bf16 there)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to("cuda")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    train_ds = PlanDataset(TRAIN_FILE, tokenizer)
    print("training rows:", len(train_ds), "| longest:", max(len(x["input_ids"]) for x in train_ds.items))

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("saved adapter to", OUTPUT_DIR)

    if PUSH_TO_HUB:
        model.push_to_hub(HF_MODEL_REPO)
        tokenizer.push_to_hub(HF_MODEL_REPO)
        print("pushed to https://huggingface.co/" + HF_MODEL_REPO)


if __name__ == "__main__":
    main()
